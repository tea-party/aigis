use anyhow::{anyhow, Error};
use calc::Context;
use serde_json::Value;

use crate::tools::AiTool;

/// Example tool that performs basic math operations.
pub struct MathTool;

#[async_trait::async_trait]
impl AiTool for MathTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        r#"A powerful calculator tool that can evaluate mathematical expressions.
Supports basic arithmetic, bitwise operations, shifts, and functions like sin, cos, tan, etc.
Do not use this tool for complex logic, programming, or string concatenation tasks, it is strictly for mathematical calculations.

order of ops:
    ( ) unary: - ! shifts/exp: << >>> >> < bitwise: & | ^ mult/div: * / // % add/sub: + -

unary ops:
    - : neg
    ! : bitwise not

infix ops:
    + : add
    - : sub
    * / // % : mult div trunc%
    ** : exp
    << >> <<< >>>: shifts
    & | ^ : bitwise

functions:
    use like sin(123)
    abs ceil floor round
    trig (radians only): sin cos tan sinh cosh tanh
    inv: asin acos atan
    inv_hyp: asinh acosh atanh
    conv: rad dec
    roots/logs: sqrt cbrt log lg ln exp

constants:
    e pi π

Example usage:
tool_name: calculator
tool_args: { "expr": "round(12345 / 543)" }
"#
    }

    async fn execute(&self, args: &Value) -> Result<Value, Error> {
        let expr = args
            .get("expr")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'expr' parameter"))?;

        let mut ctx: Context<f64> = Context::default();

        match ctx.evaluate(expr) {
            Ok(result) => Ok(serde_json::json!({ "result": result })),
            Err(e) => Err(anyhow!("Error evaluating expression '{}': {}", expr, e)),
        }
    }
}
