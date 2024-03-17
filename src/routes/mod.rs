mod error_handling;
pub(crate) mod pages;
use super::llm::start_llm_worker;
use axum::{
    extract::FromRef,
    routing::{get, post},
    Router,
};
use dotenvy::var;
use error_handling::AppError;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use sqlx::{migrate::MigrateDatabase, Sqlite};

#[derive(Clone, FromRef)]
pub struct SlackOAuthToken(pub String);

#[derive(Clone, FromRef)]
pub struct SlackSigningSecret(pub String);

#[derive(Clone, FromRef)]
pub struct AppState {
    pub db_pool: SqlitePool,
    pub slack_oauth_token: SlackOAuthToken,
    pub slack_signing_secret: SlackSigningSecret,
}

pub async fn create_routes() -> Result<Router, Box<dyn std::error::Error>> {
    let slack_oauth_token = var("SLACK_OAUTH_TOKEN")
        .map_err(|_| "Expected SLACK_OAUTH_TOKEN in the environment or .env file")?;
    let slack_oauth_token = SlackOAuthToken(slack_oauth_token);
    let slack_signing_secret = var("SLACK_SIGNING_SECRET")
        .map_err(|_| "Expected SLACK_SIGNING_SECRET in the environment or .env file")?;
    let slack_signing_secret = SlackSigningSecret(slack_signing_secret);
    let database_url = var("DATABASE_URL").unwrap_or("sqlite://db/db.sqlite3".to_owned());
    if !Sqlite::database_exists(database_url.as_str()).await.unwrap_or(false) {
        println!("Creating database {}", database_url);
        match Sqlite::create_database(database_url.as_str()).await {
            Ok(_) => println!("Create db success"),
            Err(error) => panic!("error: {}", error),
        }

        let db = SqlitePool::connect(database_url.as_str()).await.unwrap();
        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let migrations = std::path::Path::new(&crate_dir).join("./migrations");
        let migration_results = sqlx::migrate::Migrator::new(migrations)
            .await
            .unwrap()
            .run(&db)
            .await;
        match migration_results {
            Ok(_) => println!("Migration success"),
            Err(error) => {
                panic!("error: {}", error);
            }
        }
        sqlx::Pool::close(&db).await;
        println!("migration: {:?}", migration_results);
        
    } else {
        println!("Database exists");
    }
    let db_pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect(database_url.as_str())
        .await
        .map_err(|e| format!("DB connection failed: {}", e))?;
    sqlx::query("UPDATE queue SET leased_at = 0")
        .execute(&db_pool)
        .await
        .unwrap_or_default();

    start_llm_worker(db_pool.clone(), slack_oauth_token.clone()).await;

    let app_state = AppState {
        db_pool,
        slack_oauth_token,
        slack_signing_secret,
    };

    let api = Router::new()
        .route("/slack_events", post(pages::receive_slack_events))
        .route("/", get(pages::index))
        .with_state(app_state.clone());

    Ok(Router::new()
        .nest("/v1", api)
        .fallback(get(pages::not_found_json)))
}
