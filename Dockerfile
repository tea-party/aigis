FROM rust:latest AS base

FROM base AS planner
WORKDIR /app
COPY . .
RUN cargo install --locked cargo-chef
RUN cargo chef prepare --recipe-path recipe.json

FROM base AS builder
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
RUN cargo install --locked cargo-chef
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo build --release

FROM debian:bookworm-slim AS runner
RUN apt-get update && apt-get install -y openssl
COPY --from=builder /app/target/release/aigis /usr/local/bin/aigis
COPY --from=builder /app/prompt.txt /prompt.txt
CMD ["aigis"]
