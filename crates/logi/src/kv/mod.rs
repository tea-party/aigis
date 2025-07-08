use serde_json::json;

pub struct Memory {}

trait KeyValue {
    fn get_global_mem(&self) -> Option<Memory>;
    fn get_user_mem(&self) -> String;
}
