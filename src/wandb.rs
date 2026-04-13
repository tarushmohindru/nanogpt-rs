use serde_json::{Value, json};

pub struct WandbRun {
    pub run_id: String,
    pub entity: String,
    pub project: String,
    api_key: String,
    client: reqwest::blocking::Client,
}

impl WandbRun {
    pub fn init(entity: &str, project: &str, config: Value) -> Self {
        let api_key = std::env::var("WANDB_API_KEY").expect("WANDB_API_KEY not set");
        let client = reqwest::blocking::Client::new();

        // Create the run
        let res: Value = client
            .post(format!(
                "https://api.wandb.ai/api/v1/runs/{entity}/{project}"
            ))
            .bearer_auth(&api_key)
            .json(&json!({ "config": config }))
            .send()
            .unwrap()
            .json()
            .unwrap();

        let run_id = res["name"].as_str().unwrap().to_string();
        println!("wandb run: https://wandb.ai/{entity}/{project}/runs/{run_id}");

        Self {
            run_id,
            entity: entity.to_string(),
            project: project.to_string(),
            api_key,
            client,
        }
    }

    pub fn log(&self, metrics: Value, step: usize) {
        let mut payload = metrics.clone();
        payload["_step"] = json!(step);

        self.client
            .post(format!(
                "https://api.wandb.ai/api/v1/runs/{}/{}/{}/history",
                self.entity, self.project, self.run_id
            ))
            .bearer_auth(&self.api_key)
            .json(&json!({ "history": [payload] }))
            .send()
            .ok(); // Don't crash training on logging failure
    }

    pub fn finish(&self) {
        self.client
            .patch(format!(
                "https://api.wandb.ai/api/v1/runs/{}/{}/{}",
                self.entity, self.project, self.run_id
            ))
            .bearer_auth(&self.api_key)
            .json(&json!({ "state": "finished" }))
            .send()
            .ok();
    }
}
