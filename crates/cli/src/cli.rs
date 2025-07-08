use anyhow::Result;
use colored::*;
use futures_util::StreamExt;
use genai::chat::ChatMessage;
use genai::chat::ToolResponse;
use logi::llm::{AiService, LLMService};
use logi::tools::calc::MathTool;
use logi::tools::search::DDGSearchTool;
use logi::tools::website::WebsiteTool;
use regex::Regex;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use termimad::MadSkin;

/// Replace Markdown links with OSC 8 hyperlinks for supported terminals.
fn add_osc8_hyperlinks(input: &str) -> String {
    let re = Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap();
    re.replace_all(input, |caps: &regex::Captures| {
        let text = &caps[1];
        let url = &caps[2];
        format!("\x1b]8;;{}\x1b\\{}\x1b]8;;\x1b\\", url, text)
    })
    .to_string()
}

fn clear_line() {
    // ANSI escape code to clear the entire line
    print!("\r\x1b[2K");
}

const TOOL_CALL_TIMES: usize = 3; // Maximum number of repeated tool calls allowed

/// Streams and prints the assistant's response, returning the accumulated response string.
async fn print_assistant_response_stream(
    llm_service: &LLMService,
    messages: &Vec<ChatMessage>,
) -> String {
    let mut response_accum = String::new();
    let stream = llm_service.generate_response_stream(messages, None).await;
    let mut is_spinner_at_end = false;
    let spinner_frames = ['✴', '✦', '✶', '✺', '✶', '✦', '✴'];
    let mut spinner_index = 0;
    let mut last_spinner_update = Instant::now();

    let skin = MadSkin::default();

    match stream {
        Ok(mut stream) => {
            print!("{}", "Assistant:".green().bold());
            print!(" ");
            io::stdout().flush().unwrap();
            let mut is_thinking = false;
            let mut in_code_block = false;
            let mut in_list = false;
            let mut in_table = false;
            let mut block_buffer = String::new();
            let mut line_buffer = String::new();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => match chunk {
                        genai::chat::ChatStreamEvent::Chunk(stream_chunk) => {
                            if is_thinking {
                                print!("\r{}", " ".repeat(40)); // Clear spinner line
                                print!("\r{}", "--- Done!\n".green().bold());
                                is_thinking = false;
                            }
                            response_accum.push_str(&stream_chunk.content);

                            // Buffer and process lines
                            for c in stream_chunk.content.chars() {
                                line_buffer.push(c);
                                if c == '\n' {
                                    let line = line_buffer.as_str();
                                    let trimmed = line.trim_start();

                                    // Detect block starts/ends
                                    let is_code = trimmed.starts_with("```");
                                    let is_list = trimmed.starts_with("- ")
                                        || trimmed.starts_with("* ")
                                        || trimmed.starts_with("+ ")
                                        || (trimmed
                                            .chars()
                                            .next()
                                            .map(|c| c.is_ascii_digit())
                                            .unwrap_or(false)
                                            && trimmed.contains(". "));
                                    let is_table = line.contains('|') && line.contains("---");

                                    // Code block logic
                                    if is_code {
                                        block_buffer.push_str(line);
                                        in_code_block = !in_code_block;
                                        if !in_code_block {
                                            // Closed code block: render and print
                                            let with_links = add_osc8_hyperlinks(&block_buffer);
                                            let rendered = skin.term_text(&with_links);
                                            clear_line();
                                            print!("{}", rendered);
                                            io::stdout().flush().unwrap();
                                            block_buffer.clear();
                                        }
                                    }
                                    // List logic
                                    else if is_list {
                                        block_buffer.push_str(line);
                                        if !in_list {
                                            in_list = true;
                                        }
                                    } else if in_list && trimmed.is_empty() {
                                        // End of list: render and print
                                        block_buffer.push_str(line);
                                        in_list = false;
                                        let with_links = add_osc8_hyperlinks(&block_buffer);
                                        let rendered = skin.term_text(&with_links);
                                        clear_line();
                                        print!("{}", rendered);
                                        io::stdout().flush().unwrap();
                                        block_buffer.clear();
                                    }
                                    // Table logic
                                    else if is_table {
                                        block_buffer.push_str(line);
                                        if !in_table {
                                            in_table = true;
                                        }
                                    } else if in_table && trimmed.is_empty() {
                                        // End of table: render and print
                                        block_buffer.push_str(line);
                                        in_table = false;
                                        let with_links = add_osc8_hyperlinks(&block_buffer);
                                        let rendered = skin.term_text(&with_links);
                                        clear_line();
                                        print!("{}", rendered);
                                        io::stdout().flush().unwrap();
                                        block_buffer.clear();
                                    }
                                    // Paragraph/normal text
                                    else if !in_code_block
                                        && !in_list
                                        && !in_table
                                        && trimmed.is_empty()
                                    {
                                        // End of paragraph: render and print
                                        block_buffer.push_str(line);
                                        let with_links = add_osc8_hyperlinks(&block_buffer);
                                        let rendered = skin.term_text(&with_links);
                                        clear_line();
                                        print!("{}", rendered);
                                        io::stdout().flush().unwrap();
                                        block_buffer.clear();
                                    } else {
                                        block_buffer.push_str(line);
                                    }

                                    line_buffer.clear();
                                }
                            }
                            // Only show spinner if waiting for content and not about to print new content
                            if block_buffer.trim().is_empty() {
                                if is_spinner_at_end {
                                    clear_line();
                                    is_spinner_at_end = false;
                                }
                                // Print content only, no spinner prefix
                                clear_line();
                                print!("{}", block_buffer);
                                io::stdout().flush().unwrap();
                            } else {
                                // Animate spinner only while waiting for next token
                                if last_spinner_update.elapsed() >= Duration::from_millis(100) {
                                    clear_line();
                                    print!(
                                        "{}",
                                        spinner_frames[spinner_index % spinner_frames.len()]
                                    );
                                    io::stdout().flush().unwrap();
                                    spinner_index += 1;
                                    last_spinner_update = Instant::now();
                                }
                                is_spinner_at_end = true;
                            }
                        }
                        genai::chat::ChatStreamEvent::ReasoningChunk(_stream_chunk) => {
                            if !is_thinking {
                                print!("\r{}", "--- Thinking... ".yellow().bold());
                                io::stdout().flush().unwrap();
                                is_thinking = true;
                                spinner_index = 0;
                                last_spinner_update = Instant::now();
                            }
                            // Animate spinner every ~100ms
                            if last_spinner_update.elapsed() >= Duration::from_millis(100) {
                                print!(
                                    "\r{} Thinking...",
                                    spinner_frames[spinner_index % spinner_frames.len()]
                                );
                                io::stdout().flush().unwrap();
                                spinner_index += 1;
                                last_spinner_update = Instant::now();
                            }
                        }
                        _ => (),
                    },
                    Err(e) => {
                        println!("{}", format!("\nError: {}", e).red().bold());
                        break;
                    }
                }
            }
            if !block_buffer.trim().is_empty() {
                let with_links = add_osc8_hyperlinks(&block_buffer);
                let rendered = skin.term_text(&with_links);
                print!("{}", rendered);
            }
            if is_thinking {
                print!("\r{}", " ".repeat(40)); // Clear spinner line
                print!("\r{}", "--- Done!\n".green().bold());
            }
            println!();
        }
        Err(e) => {
            println!(
                "{}: {e}",
                "Error: Failed to get response stream from LLMService"
                    .red()
                    .bold()
            );
        }
    }
    response_accum
}

/// Runs the CLI mode for interacting with the LLMService.
pub async fn run_cli() -> Result<()> {
    // get formatted current time (to provide to the LLMService)
    let current_time =
        time::OffsetDateTime::now_utc().format(&time::format_description::well_known::Rfc3339)?;

    let default_prompt = format!(
        "You are a helpful assistant. Keep your thoughts short and sweet, and take extra special note of previous responses, especially for tool use. The current time is {}",
        current_time
    );

    // load the system prompt from 'prompt.txt' if it exists
    let prompt_path: &str = "prompt_cli.txt";
    let prompt_string = std::fs::read_to_string(prompt_path).ok();
    let mut system_prompt: Option<&str> = prompt_string.as_deref();

    if let None = system_prompt {
        system_prompt = Some(&default_prompt);
    }

    // Initialize LLMService with tools
    let mut llm_service = LLMService::new(
        system_prompt,
        vec![
            Box::new(MathTool),
            Box::new(DDGSearchTool),
            Box::new(WebsiteTool),
        ],
        "DeepSeek-R1-0528",
    )?;

    println!("Welcome to the Aigis CLI!");
    println!(
        "Type your messages below. Type 'exit' to quit or use slash commands (e.g., /command) to manage settings."
    );
    println!("Available tools:");
    for tool in llm_service.list_tools() {
        println!("- {}", tool);
    }
    println!();

    let mut messages = vec![];

    loop {
        // Read user input
        print!("{}", "You:".cyan().bold());
        print!(" ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        if input.starts_with('/') {
            let command_input = input.trim_start_matches('/').trim();

            match command_input {
                "help" => {
                    println!("{}", "Available commands:".magenta().bold());
                    println!(
                        "{}",
                        "  /set_prompt <new_prompt> - Change the system prompt.".magenta()
                    );
                    println!("{}", "  /list_tools - List all available tools.".magenta());
                    println!("{}", "  /exit - Exit the CLI.".magenta());
                }
                cmd if cmd.starts_with("set_prompt ") => {
                    let new_prompt = cmd.trim_start_matches("set_prompt ").to_string();
                    llm_service.set_system_prompt(new_prompt);
                    println!("{}", "System prompt updated.".magenta());
                }
                "list_tools" => {
                    println!("Available tools:");
                    for tool in llm_service.list_tools() {
                        println!("- {}", tool);
                    }
                }
                "exit" => {
                    println!("Goodbye!");
                    break;
                }
                _ => {
                    println!(
                        "{}",
                        "Unknown command. Type '/help' for a list of commands."
                            .magenta()
                            .bold()
                    );
                }
            }
        } else {
            // Add user message to conversation
            messages.push(ChatMessage::user(input.to_string()));

            // Streaming response with tool call support
            use logi::tools::{execute_tool_calls, parse_tool_calls};

            // Use the helper function for initial assistant response
            let stream_messages = messages.clone();
            let mut response_accum =
                print_assistant_response_stream(&llm_service, &stream_messages).await;

            // Tool call detection after streaming, now allowing possibly infinite chaining
            let mut last_tool_call: Option<(String, serde_json::Value)> = None;
            let mut last_tool_call_times = 0;
            loop {
                let tool_calls = parse_tool_calls(&response_accum);
                if !tool_calls.is_empty() {
                    // Check for repeated tool call (name + args) for the first tool call only
                    let first_call = &tool_calls[0];
                    if let Some((last_name, last_args)) = &last_tool_call {
                        if last_name == &first_call.tool_name && last_args == &first_call.tool_args
                        {
                            if last_tool_call_times >= TOOL_CALL_TIMES {
                                println!(
                                    "{}",
                                    "! error ! Too many repeated tool calls, breaking to avoid infinite loop."
                                        .red()
                                        .bold()
                                );
                                break;
                            }
                            last_tool_call_times += 1;
                        } else {
                            last_tool_call_times = 1; // Reset count for new tool call
                        }
                    }
                    last_tool_call =
                        Some((first_call.tool_name.clone(), first_call.tool_args.clone()));

                    // Execute all tool calls in order
                    let tool_results = futures::executor::block_on(execute_tool_calls(
                        &tool_calls,
                        &llm_service.tools,
                    ));
                    for (tool_name, result) in &tool_results {
                        match result {
                            Ok(tool_result) => {
                                println!(
                                    "\n{}",
                                    format!("[Tool `{}` returned: {}]", tool_name, tool_result)
                                        .yellow()
                                        .bold()
                                );
                                // Feed tool result back into conversation and stream follow-up
                                messages.push(ChatMessage::assistant(response_accum.clone()));
                                messages.push(
                                    ToolResponse::new(tool_name.clone(), tool_result.to_string())
                                        .into(),
                                );
                            }
                            Err(e) => {
                                println!(
                                    "{}",
                                    format!(
                                        "! error ! Error executing tool `{}`: {}",
                                        tool_name, e
                                    )
                                    .red()
                                    .bold()
                                );
                                // Feed error result back into conversation
                                messages.push(ChatMessage::assistant(response_accum.clone()));
                                messages.push(
                                    ToolResponse::new(tool_name.clone(), format!("Error: {}", e))
                                        .into(),
                                );
                            }
                        }
                    }
                    // Use the helper function for follow-up assistant response
                    let followup_accum =
                        print_assistant_response_stream(&llm_service, &messages).await;
                    messages.push(ChatMessage::assistant(followup_accum.clone()));
                    // Prepare for next loop iteration
                    response_accum = followup_accum;
                    continue;
                } else {
                    // No tool call, just add the response to the conversation
                    messages.push(ChatMessage::assistant(response_accum));
                    break;
                }
            }
        }
    }

    Ok(())
}
