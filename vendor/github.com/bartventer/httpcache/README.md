# httpcache

[![Go Reference](https://pkg.go.dev/badge/github.com/bartventer/httpcache.svg)](https://pkg.go.dev/github.com/bartventer/httpcache)
[![Go Report Card](https://goreportcard.com/badge/github.com/bartventer/httpcache)](https://goreportcard.com/report/github.com/bartventer/httpcache)
[![Test](https://github.com/bartventer/httpcache/actions/workflows/default.yml/badge.svg)](https://github.com/bartventer/httpcache/actions/workflows/default.yml)
[![codecov](https://codecov.io/github/bartventer/httpcache/graph/badge.svg?token=pnpoA3t4EE)](https://codecov.io/github/bartventer/httpcache)

**httpcache** is a Go package that provides a standards-compliant [http.RoundTripper](https://pkg.go.dev/net/http#RoundTripper) for transparent HTTP response caching, following [RFC 9111 (HTTP Caching)](https://www.rfc-editor.org/rfc/rfc9111).

> **Note:** This package is intended for use as a **private (client-side) cache**. It is **not** a shared or proxy cache. It is designed to be used with an HTTP client to cache responses from origin servers, improving performance and reducing load on those servers.

## Features

- **Plug-and-Play**: Just swap in as your HTTP client's transport; no extra configuration needed. [^1]
- **RFC 9111 Compliance**: Handles validation, expiration, and revalidation ([see details](#rfc-9111-compliance-matrix)).
- **Cache Control**: Supports all required HTTP cache control directives, as well as extensions like `stale-while-revalidate`, `stale-if-error`, and `immutable` ([view details](#field-definitions-details)).
- **Cache Backends**: Built-in support for file system and memory caches, with the ability to implement custom backends (see [Cache Backends](#cache-backends)).
- **Cache Maintenance API**: Optional REST endpoints for listing, retrieving, and deleting cache entries (see [Cache Maintenance API](#cache-maintenance-api-debug-only)).
- **Extensible**: Options for logging, transport and timeouts (see [Options](#options)).
- **Debuggable**: Adds a cache status header to every response (see [Cache Status Headers](#cache-status-headers)).
- **Zero Dependencies**: No external dependencies, pure Go implementation.

![Made with VHS](https://vhs.charm.sh/vhs-3WOBtYTZzzXggFGYRudHTV.gif)

*Demonstration of HTTP caching in action. See [_examples/app](_examples/app/app.go) for code.*

## Installation

To install the package, run:

```bash
go get github.com/bartventer/httpcache
```

## Quick Start

To get started, create a new HTTP client with the `httpcache` transport, specifying a cache backend DSN. You'll need to register the desired cache backend before using it. Here's an example using the built-in file system cache:

```go
package main

import (
    "log/slog"
    "net/http"
    "time"

    "github.com/bartventer/httpcache"
    // Register the file system cache backend
    _ "github.com/bartventer/httpcache/store/fscache"
)

func main() {
    // Example DSN for the file system cache backend
    dsn := "fscache://?appname=myapp"
    client := &http.Client{
        Transport: httpcache.NewTransport(
            dsn,
            httpcache.WithSWRTimeout(10*time.Second),
            httpcache.WithLogger(slog.Default()),
        ),
    }
    // ... Use the client as usual
}
```

> **Note:** The DSN format and options depend on the cache backend you choose. Refer to the [Cache Backends](#cache-backends) section for details on available backends and their DSN formats.

## Cache Backends

The following built-in cache backends are available:

| Backend                                                                         | DSN Example                | Description                                                                                                                                                            |
| ------------------------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`fscache`](https://pkg.go.dev/github.com/bartventer/httpcache/store/fscache)   | `fscache://?appname=myapp` | File system cache, stores responses on disk. Suitable for persistent caching across restarts. Supports context cancellation, as well as optional `AES-GCM` encryption. |
| [`memcache`](https://pkg.go.dev/github.com/bartventer/httpcache/store/memcache) | `memcache://`              | In-memory cache, suitable for ephemeral caching. Does not persist across restarts.                                                                                     |

Consult the documentation for each backend for specific configuration options and usage details.

### Custom Cache Backends

To implement a custom cache backend, create a type that satisfies the [`store/driver.Conn`](https://pkg.go.dev/github.com/bartventer/httpcache/store/driver#Conn) interface, then register it using the [`store.Register`](https://pkg.go.dev/github.com/bartventer/httpcache/store#Register) function. Refer to the built-in backends for examples of how to implement this interface.

### Cache Maintenance API (Debug Only)

A REST API is available for cache inspection and maintenance, intended for debugging and development use only. **Do not expose these endpoints in production.**

**Endpoints:**
- `GET    /debug/httpcache`           — List cache keys (if supported)
- `GET    /debug/httpcache/{key}`     — Retrieve a cache entry
- `DELETE /debug/httpcache/{key}`     — Delete a cache entry

All endpoints require a `dsn` query parameter to select the cache backend.

**Usage Example:**
```go
import (
    "net/http"
    "github.com/bartventer/httpcache/store/expapi"
)

func main() {
    expapi.Register()
    http.ListenAndServe(":8080", nil)
}
```

To use a custom [ServeMux](https://pkg.go.dev/net/http#ServeMux), pass `expapi.WithServeMux(mux)` to `expapi.Register()`.

## Options

| Option                            | Description                                         | Default Value                   |
| --------------------------------- | --------------------------------------------------- | ------------------------------- |
| `WithUpstream(http.RoundTripper)` | Set a custom transport for upstream/origin requests | `http.DefaultTransport`         |
| `WithSWRTimeout(time.Duration)`   | Set the stale-while-revalidate timeout              | `5 * time.Second`               |
| `WithLogger(*slog.Logger)`        | Set a logger for debug output                       | `slog.New(slog.DiscardHandler)` |

## Cache Status Headers

This package sets a cache status header on every response:

- `X-Httpcache-Status`: The primary, detailed cache status header (always set).
- `X-From-Cache`: (Legacy) Provided for compatibility with [`gregjones/httpcache`](https://github.com/gregjones/httpcache). Only set for cache hits/stale/revalidated responses.

### Header Value Mapping

| X-Httpcache-Status | X-From-Cache | Description                        |
| ------------------ | ------------ | ---------------------------------- |
| HIT                | 1            | Served from cache                  |
| STALE              | 1            | Served from cache but stale        |
| REVALIDATED        | 1            | Revalidated with origin            |
| MISS               | *(not set)*  | Served from origin                 |
| BYPASS             | *(not set)*  | Bypassed cache, served from origin |

### Example: Stale cache hit

```http
HTTP/1.1 200 OK
X-Httpcache-Status: STALE
X-From-Cache: 1
Content-Type: application/json
```

## Limitations

- **Range Requests & Partial Content:**
  This cache does **not** support HTTP range requests or partial/incomplete responses (e.g., status code 206, `Range`/`Content-Range` headers). All requests with a `Range` header are bypassed, and 206 responses are not cached. For example:

  ```http
  GET /example.txt HTTP/1.1
  Host: example.com
  Range: bytes=0-99
  ```

  The above request will bypass the cache and fetch the response directly from the origin server. See [RFC 9111 §3.3-3.4](https://www.rfc-editor.org/rfc/rfc9111#section-3.3) for details.


## RFC 9111 Compliance Matrix

[![RFC 9111](https://img.shields.io/badge/RFC%209111-Compliant-brightgreen)](https://www.rfc-editor.org/rfc/rfc9111)

| §   | Title                                         | Requirement | Implemented | Notes                                      |
| --- | --------------------------------------------- | :---------: | :---------: | ------------------------------------------ |
| 1.  | Introduction                                  |     N/A     |     N/A     | Nothing to implement                       |
| 2.  | Overview of Cache Operation                   |     N/A     |     N/A     | Nothing to implement                       |
| 3.  | Storing Responses in Caches                   |  Required   |      ✔️      | [Details](#storing-responses-details)      |
| 4.  | Constructing Responses from Caches            |  Required   |      ✔️      | [Details](#constructing-responses-details) |
| 5.  | Field Definitions                             |  Required   |      ✔️      | [Details](#field-definitions-details)      |
| 6.  | Relationship to Applications and Other Caches |     N/A     |     N/A     | Nothing to implement                       |
| 7.  | Security Considerations                       |     N/A     |     N/A     | Nothing to implement                       |
| 8.  | IANA Considerations                           |     N/A     |     N/A     | Nothing to implement                       |
| 9.  | References                                    |     N/A     |     N/A     | Nothing to implement                       |

**Legend for Requirements:**

| Requirement | Description                                                             |
| ----------- | ----------------------------------------------------------------------- |
| Required    | *Must be implemented for RFC compliance*                                |
| Optional    | *May be implemented, but not required for compliance*                   |
| Obsolete    | *Directive is no longer relevant as per RFC 9111*                       |
| Deprecated  | *Directive is deprecated as per RFC 9111, but can still be implemented* |
| N/A         | *Nothing to implement or not applicable to private caches*              |

<details id="storing-responses-details">
<summary><strong>§3. Storing Responses in Caches (Details)</strong></summary>

| §    | Title                                       | Requirement | Implemented | Notes                                        |
| ---- | ------------------------------------------- | :---------: | :---------: | -------------------------------------------- |
| 3.1. | Storing Header and Trailer Fields           |  Required   |      ✔️      |                                              |
| 3.2. | Updating Stored Header Fields               |  Required   |      ✔️      |                                              |
| 3.3. | Storing Incomplete Responses                |  Optional   |      ❌      | See [Limitations](#limitations)              |
| 3.4. | Combining Partial Content                   |  Optional   |      ❌      | See [Limitations](#limitations)              |
| 3.5. | Storing Responses to Authenticated Requests |     N/A     |     N/A     | Not applicable to private client-side caches |

</details>

<details id="constructing-responses-details">
<summary><strong>§4. Constructing Responses from Caches (Details)</strong></summary>

| §    | Title                                             | Requirement | Implemented | Notes                         |
| ---- | ------------------------------------------------- | :---------: | :---------: | ----------------------------- |
| 4.1. | Calculating Cache Keys with the Vary Header Field |  Required   |      ✔️      |                               |
| 4.2. | Freshness                                         |  Required   |      ✔️      | [Details](#freshness-details) |

<details id="freshness-details">
<summary><em>§4.2. Freshness (Subsections)</em></summary>

| §      | Title                           | Requirement | Implemented | Notes |
| ------ | ------------------------------- | :---------: | :---------: | ----- |
| 4.2.1. | Calculating Freshness Lifetime  |  Required   |      ✔️      |       |
| 4.2.2. | Calculating Heuristic Freshness |  Required   |      ✔️      |       |
| 4.2.3. | Calculating Age                 |  Required   |      ✔️      |       |
| 4.2.4. | Serving Stale Responses         |  Required   |      ✔️      |       |

</details>

| §    | Title      | Requirement | Implemented | Notes                          |
| ---- | ---------- | :---------: | ----------- | ------------------------------ |
| 4.3. | Validation |  Required   | ✔️           | [Details](#validation-details) |

<details id="validation-details">
<summary><em>§4.3. Validation (Subsections)</em></summary>

|   §    | Title                                       | Requirement | Implemented | Notes                                                                                                                                  |
| :----: | ------------------------------------------- | :---------: | :---------: | -------------------------------------------------------------------------------------------------------------------------------------- |
| 4.3.1. | Sending a Validation Request                |  Required   |      ✔️      |                                                                                                                                        |
| 4.3.2. | Handling Received Validation Request        |     N/A     |     N/A     | Not applicable to private client-side caches                                                                                           |
| 4.3.3. | Handling a Validation Response              |  Required   |      ✔️      |                                                                                                                                        |
| 4.3.4. | Freshening Stored Responses upon Validation |  Required   |      ✔️      |                                                                                                                                        |
| 4.3.5. | Freshening Responses with HEAD              |  Optional   |      ❌      | Pointless, rather use conditional GETs; see [RFC 9110 §13.2.1 last para](https://datatracker.ietf.org/doc/html/rfc9110#section-13.2.1) |

</details>

| §    | Title                         | Requirement | Implemented | Notes |
| ---- | ----------------------------- | :---------: | :---------: | ----- |
| 4.4. | Invalidating Stored Responses |  Required   |      ✔️      |       |

</details>

<details id="field-definitions-details">
<summary><strong>§5. Field Definitions (Details)</strong></summary>

| §    | Title         | Requirement | Implemented | Notes                                   |
| ---- | ------------- | :---------: | :---------: | --------------------------------------- |
| 5.1. | Age           |  Required   |      ✔️      |                                         |
| 5.2. | Cache-Control |  Required   |      ✔️      | [Details](#cache-control-directives)    |
| 5.3. | Expires       |  Required   |      ✔️      |                                         |
| 5.4. | Pragma        | Deprecated  |      ❌      | Deprecated by RFC 9111; not implemented |
| 5.5. | Warning       |  Obsolete   |      ❌      | Obsoleted by RFC 9111; not implemented  |

<details id="cache-control-directives">
<summary><em>§5.2. Cache-Control Directives</em></summary>

| §      | Title              | Requirement | Implemented | Notes                                  |
| ------ | ------------------ | :---------: | :---------: | -------------------------------------- |
| 5.2.1. | Request Directives |  Optional   |      ✔️      | [Details](#request-directives-details) |

<details id="request-directives-details">
<summary><em>§5.2.1. Request Directives (Details)</em></summary>

| §        | Title/Directive  | Requirement | Implemented | Notes                                                          |
| -------- | ---------------- | :---------: | :---------: | -------------------------------------------------------------- |
| 5.2.1.1. | `max-age`        |  Optional   |      ✔️      |                                                                |
| 5.2.1.2. | `max-stale`      |  Optional   |      ✔️      |                                                                |
| 5.2.1.3. | `min-fresh`      |  Optional   |      ✔️      |                                                                |
| 5.2.1.4. | `no-cache`       |  Optional   |      ✔️      |                                                                |
| 5.2.1.5. | `no-store`       |  Optional   |      ✔️      |                                                                |
| 5.2.1.6. | `no-transform`   |  Optional   |      ✔️      | Compliant by default - implementation never transforms content |
| 5.2.1.7. | `only-if-cached` |  Optional   |      ✔️      |                                                                |

</details>

| Section                    | Requirement | Implemented | Notes                                   |
| -------------------------- | :---------: | :---------: | --------------------------------------- |
| 5.2.2. Response Directives |  Required   |      ✔️      | [Details](#response-directives-details) |

<details id="response-directives-details">
<summary><em>§5.2.2. Response Directives (Details)</em></summary>

| §         | Title/Directive    | Requirement | Implemented | Notes                                                          |
| --------- | ------------------ | :---------: | :---------: | -------------------------------------------------------------- |
| 5.2.2.1.  | `max-age`          |  Required   |      ✔️      |                                                                |
| 5.2.2.2.  | `must-revalidate`  |  Required   |      ✔️      |                                                                |
| 5.2.2.3.  | `must-understand`  |  Required   |      ✔️      |                                                                |
| 5.2.2.4.  | `no-cache`         |  Required   |      ✔️      | Both qualified and unqualified forms supported                 |
| 5.2.2.5.  | `no-store`         |  Required   |      ✔️      |                                                                |
| 5.2.2.6.  | `no-transform`     |  Required   |      ✔️      | Compliant by default - implementation never transforms content |
| 5.2.2.7.  | `private`          |     N/A     |     N/A     | Intended for shared caches; not applicable to private caches   |
| 5.2.2.8.  | `proxy-revalidate` |     N/A     |     N/A     | Intended for shared caches; not applicable to private caches   |
| 5.2.2.9.  | `public`           |  Optional   |      ✔️      |                                                                |
| 5.2.2.10. | `s-maxage`         |     N/A     |     N/A     | Intended for shared caches; not applicable to private caches   |

</details>

| §      | Title                | Requirement | Implemented | Notes                                    |
| ------ | -------------------- | :---------: | :---------: | ---------------------------------------- |
| 5.2.3. | Extension Directives |  Optional   | *partially* | [Details](#extension-directives-details) |

<details id="extension-directives-details">
<summary><em>§5.2.3. Extension Directives (Details)</em></summary>

The following additional cache control directives are supported, as defined in various RFCs:

| Reference                                                        | Directive                | Notes                                  |
| ---------------------------------------------------------------- | ------------------------ | -------------------------------------- |
| [RFC 5861, §3](https://www.rfc-editor.org/rfc/rfc5861#section-3) | `stale-while-revalidate` | Only applies to responses              |
| [RFC 5861, §4](https://www.rfc-editor.org/rfc/rfc5861#section-4) | `stale-if-error`         | Applies to both requests and responses |
| [RFC 8246, §2](https://www.rfc-editor.org/rfc/rfc8246)           | `immutable`              | Only applies to responses              |

</details>
</details>
</details>

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](LICENSE) file for details.

## Notes

[^1]: No configuration is needed beyond the cache backend DSN. Caching is handled automatically based on HTTP headers and directives. To use a custom upstream transport, pass it with the `WithUpstream` option. This lets you add `httpcache` to your existing HTTP client with minimal changes. See [Options](#options) for details.