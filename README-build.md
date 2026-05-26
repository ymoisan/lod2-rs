# Building lod2-rs

## Quick start (native Linux with GDAL)

The full workspace (all binaries) requires GDAL development headers.
On Ubuntu/Debian:

```bash
sudo apt-get install libgdal-dev libclang-dev
cargo build --release
```

Binaries land in `target/release/`.

---

## Static MUSL executable (`buildex`)

The `hybrid-static.dockerfile` produces a fully-static, musl-linked binary
called `buildex` from the `hybrid` workspace crate.  It runs on any x86_64
Linux system with **no runtime dependencies** — no GDAL, no shared libraries.

This is possible because `hybrid` uses pure-Rust I/O:
- **GeoPackage** — `rusqlite` with bundled SQLite + WKB geometry parsing
- **LAS/LAZ** — the `las` crate (pure Rust)

### Prerequisites

- Docker (or Podman with Docker CLI compatibility)

### Build

From the repository root:

```bash
docker build --output type=local,dest=. -f hybrid-static.dockerfile .
```

This writes `./buildex` directly to the current directory.

### Corporate proxies / custom CA certificates

Place `.crt` or `.pem` files in a `certs/` directory at the repo root before
building.  The Dockerfile installs them into the Alpine CA bundle so that
`cargo` and `rustup` can reach crates.io behind TLS-intercepting proxies.
If `certs/` is empty or absent the step is a no-op.

### Verify the binary

```bash
file ./buildex          # should report "ELF 64-bit ... statically linked"
./buildex --help
```

---

## Building just `hybrid` without GDAL

Because `hybrid` uses pure-Rust I/O, you can build it on any platform
without installing GDAL:

```bash
cargo build --release --package hybrid
```

The other workspace binaries (`arrangement`, `graph-cut`, `plane-extrude`)
still require GDAL.

---

## Cross-platform builds (Windows / macOS)

### `hybrid` only (no GDAL needed)

```bash
cargo build --release --package hybrid
```

Works on any platform with a Rust toolchain.

### Full workspace (needs GDAL)

#### Windows

1. Install GDAL via [OSGeo4W](https://trac.osgeo.org/osgeo4w/) or
   [vcpkg](https://vcpkg.io/) (`vcpkg install gdal`).
2. Set the `GDAL_HOME` environment variable to the GDAL prefix.
3. `cargo build --release`

#### macOS

```bash
brew install gdal
cargo build --release
```

#### WSL

```bash
sudo apt-get install libgdal-dev libclang-dev
cargo build --release
```

### Nix

If you have a Nix dev-shell (e.g. from a companion repo):

```bash
nix develop
cargo build --release
```
