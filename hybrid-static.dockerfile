# Builder-only Dockerfile for producing a static `buildex` binary (hybrid crate).
#
# No GDAL dependency — the hybrid crate uses pure-Rust I/O (rusqlite for
# GeoPackage, las crate for point clouds).  Like tyler-static.dockerfile,
# this uses Alpine (musl-native) for a fully self-contained binary.
#
# Usage:
#   docker build --output type=local,dest=. -f hybrid-static.dockerfile .
#
# This writes ./buildex directly to the current directory.

FROM rust:1.93-alpine AS builder

# Install corporate CA certificates so TLS works behind corporate proxies.
# To use: place .crt/.pem files in a certs/ directory at the repo root.
# If certs/ is empty or absent, this step is a no-op.
COPY certs/ /usr/local/share/ca-certificates/
RUN apk add --no-cache ca-certificates && update-ca-certificates

RUN apk add --no-cache \
    build-base \
    cmake \
    pkgconf

WORKDIR /usr/src/lod2-rs

# Copy workspace manifests first for Docker layer caching.
COPY Cargo.toml Cargo.lock ./
COPY lod2-common/Cargo.toml   lod2-common/Cargo.toml
COPY hybrid/Cargo.toml         hybrid/Cargo.toml

# Stub out other workspace members so cargo can resolve the workspace.
# Only hybrid and lod2-common sources are needed.
COPY plane-extrude/Cargo.toml  plane-extrude/Cargo.toml
COPY arrangement/Cargo.toml    arrangement/Cargo.toml
COPY graph-cut/Cargo.toml      graph-cut/Cargo.toml
COPY footprint-extract/Cargo.toml footprint-extract/Cargo.toml
COPY footprint-merge/Cargo.toml   footprint-merge/Cargo.toml
COPY footprint-repair/Cargo.toml  footprint-repair/Cargo.toml
COPY eval/Cargo.toml           eval/Cargo.toml

# Copy source trees needed for the build.
COPY lod2-common/src   lod2-common/src
COPY hybrid/src        hybrid/src

# Create stub lib.rs for workspace members we don't actually compile,
# so cargo doesn't complain about missing sources.
RUN for d in plane-extrude arrangement graph-cut footprint-extract \
             footprint-merge footprint-repair eval; do \
        mkdir -p "$d/src" && echo "fn main(){}" > "$d/src/main.rs"; \
    done

RUN cargo build --release --package hybrid --target x86_64-unknown-linux-musl && \
    cp target/x86_64-unknown-linux-musl/release/hybrid /usr/local/bin/buildex

# Verify the binary is static.
RUN file /usr/local/bin/buildex && ldd /usr/local/bin/buildex 2>&1 || true

# Minimal final stage — just the binary, for easy extraction.#
FROM scratch
COPY --from=builder /usr/local/bin/buildex /buildex
