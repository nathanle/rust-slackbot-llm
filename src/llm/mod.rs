mod utils;

use super::routes::pages::send_user_message;
use super::routes::SlackOAuthToken;
use log::error;
use reqwest::{header::AUTHORIZATION, multipart};
use sqlx::SqlitePool;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot};
use utils::Model;


const SAMPLE_LEN: usize = 1000;
const TEMPERATURE: Option<f64> = Some(0.8);
const TOP_P: Option<f64> = None;
const SEED: Option<u64> = None;
const REPEAT_PENALTY: f32 = 1.1;
const REPEAT_LAST_N: i32 = 64;

struct LlmRequest {
    prompt: String,
    pre_prompt_tokens: Vec<u32>,
    response: oneshot::Sender<Result<(Vec<u32>, String), String>>,
}

/// Starts a background worker that pulls tasks from the DB queue, runs the LLM,
/// persists session state, and replies in Slack.
pub fn start_llm_worker(db_pool: SqlitePool, slack_oauth_token: SlackOAuthToken) {
    let (llm_tx, mut llm_rx) = mpsc::channel::<LlmRequest>(1);

    // Keep the blocking LLM model on a dedicated thread so inference does not
    // stall the async runtime's worker threads.
    thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build LLM thread runtime");

        rt.block_on(async move {
            thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min)
                .unwrap_or_default();

            let mut llm_model = match Model::start_model(
                TEMPERATURE,
                TOP_P,
                SEED,
                SAMPLE_LEN,
                REPEAT_PENALTY,
                REPEAT_LAST_N.try_into().unwrap(),
            )
            .await
            {
                Ok(model) => model,
                Err(e) => {
                    error!("Failed to start LLM model: {e}");
                    return;
                }
            };

            while let Some(request) = llm_rx.recv().await {
                let result = llm_model
                    .run_model_iteraction(request.prompt, request.pre_prompt_tokens)
                    .map_err(|e| format!("Model interaction failed: {e}"));
                let _ = request.response.send(result);
            }
        });
    });

    tokio::spawn(async move {
        loop {
            if let Err(e) = run_worker_loop(&db_pool, &slack_oauth_token, &llm_tx).await {
                error!("LLM worker loop exited with error: {e:#}, restarting in 5 seconds");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    });
}

async fn run_worker_loop(
    db_pool: &SqlitePool,
    slack_oauth_token: &SlackOAuthToken,
    llm_tx: &mpsc::Sender<LlmRequest>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    loop {
        let (task_id, prompt_str, channel, thread_ts) = get_next_task(db_pool).await?;
        let pre_prompt_tokens =
            get_session_state(db_pool, &channel, &thread_ts, slack_oauth_token).await?;

        let (response_tx, response_rx) = oneshot::channel();
        llm_tx
            .send(LlmRequest {
                prompt: prompt_str,
                pre_prompt_tokens,
                response: response_tx,
            })
            .await
            .map_err(|_| "LLM worker thread stopped")?;

        let (next_pre_prompt_tokens, generated_text) = response_rx
            .await
            .map_err(|_| "LLM worker dropped response channel")??;

        save_session_and_dequeue(
            db_pool,
            task_id,
            &channel,
            &thread_ts,
            &next_pre_prompt_tokens,
        )
        .await?;

        let reply = trim_generated_text(&generated_text);
        if let Err(e) = send_user_message(slack_oauth_token, channel, thread_ts, reply).await {
            error!("Failed to send Slack reply: {e:?}");
        }
    }
}

async fn save_session_and_dequeue(
    db_pool: &SqlitePool,
    task_id: i64,
    channel: &str,
    thread_ts: &str,
    next_pre_prompt_tokens: &[u32],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let encoded = bincode::serialize(next_pre_prompt_tokens)
        .map_err(|e| format!("Failed to encode model state: {e}"))?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("System clock error: {e:?}"))?
        .as_secs() as i64;

    sqlx::query("DELETE FROM queue WHERE id = $1;")
        .bind(task_id)
        .execute(db_pool)
        .await?;

    sqlx::query(
        "INSERT INTO sessions
        (channel, thread_ts, created_at, updated_at, model_state)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (channel, thread_ts)
        DO UPDATE SET
        model_state = EXCLUDED.model_state,
        updated_at = EXCLUDED.updated_at;",
    )
    .bind(channel)
    .bind(thread_ts)
    .bind(now)
    .bind(now)
    .bind(encoded)
    .execute(db_pool)
    .await?;

    Ok(())
}

fn trim_generated_text(generated_text: &str) -> String {
    if generated_text.len() > 5 {
        generated_text[1..generated_text.len() - 4].to_owned()
    } else {
        generated_text.to_owned()
    }
}

async fn get_next_task(
    db_pool: &SqlitePool,
) -> Result<(i64, String, String, String), Box<dyn std::error::Error + Send + Sync>> {
    loop {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("System clock error: {e:?}"))?
            .as_secs() as i64;

        let mut tx = db_pool.begin().await?;
        match sqlx::query_as::<_, (i64, String, String, String)>(
            "
            SELECT id, text, channel, thread_ts FROM queue
            WHERE leased_at <= $1
            ORDER BY created_at ASC
            LIMIT 1
            ",
        )
        .bind(now - 600)
        .fetch_one(&mut *tx)
        .await
        {
            Ok((task_id, prompt_str, channel, thread_ts)) => {
                if sqlx::query(
                    "
                    UPDATE queue SET leased_at = $1
                    WHERE id = $2
                    ",
                )
                .bind(now)
                .bind(task_id)
                .execute(&mut *tx)
                .await
                .is_ok()
                    && tx.commit().await.is_ok()
                {
                    return Ok((task_id, prompt_str, channel, thread_ts));
                }
            }
            Err(_) => tokio::time::sleep(Duration::from_secs(1)).await,
        }
    }
}

async fn get_session_state(
    db_pool: &SqlitePool,
    channel: &str,
    thread_ts: &str,
    slack_oauth_token: &SlackOAuthToken,
) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
    let mut initial_message = "Running LLM ".to_owned();

    let pre_prompt_tokens = match sqlx::query_as::<_, (Vec<u8>,)>(
        r#"SELECT model_state FROM sessions WHERE channel = $1 AND thread_ts = $2;"#,
    )
    .bind(channel)
    .bind(thread_ts)
    .fetch_optional(db_pool)
    .await?
    {
        Some((model_state,)) => {
            initial_message.push_str("reusing section. ");
            bincode::deserialize(&model_state).unwrap_or_default()
        }
        None => {
            initial_message.push_str("with new section. ");
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| format!("System clock error: {e:?}"))?
                .as_secs() as i64;

            sqlx::query(
                r#"INSERT OR IGNORE INTO
                sessions (channel, thread_ts, created_at, updated_at)
                VALUES ($1, $2, $3, $4);"#,
            )
            .bind(channel)
            .bind(thread_ts)
            .bind(timestamp)
            .bind(timestamp)
            .execute(db_pool)
            .await?;

            Vec::new()
        }
    };

    let reqw_client = reqwest::Client::new();
    let form = multipart::Form::new()
        .text("text", initial_message)
        .text("channel", channel.to_owned())
        .text("thread_ts", thread_ts.to_owned());

    if let Err(e) = reqw_client
        .post("https://slack.com/api/chat.postMessage")
        .header(AUTHORIZATION, format!("Bearer {}", slack_oauth_token.0))
        .multipart(form)
        .send()
        .await
    {
        error!("Failed to post Slack status message: {e}");
    }

    Ok(pre_prompt_tokens)
}

