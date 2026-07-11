// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

// Package loading provides tools to load a file from http or from a local file system.
//
// # Security
//
// By default, the local loader reads any path the process can access, including absolute
// paths and "file://" URIs (for example "file:///etc/passwd"). Applications that pass
// untrusted input to [LoadFromFileOrHTTP], [JSONDoc] (or to downstream consumers such as
// go-openapi/loads) must confine local loading to a trusted directory.
//
// Use [WithRoot] to do so: it resolves every requested path relative to a chosen directory
// and rejects anything that escapes it, including via symlink. It is built on [os.Root]
// and is therefore safer than passing an [os.DirFS] to [WithFS], which does not block
// symlink escapes.
//
// Remote loading uses a standard [net/http] client.
// By default it follows redirects and performs no destination filtering — exactly like [net/http.DefaultClient].
//
// A caller-controlled URL may therefore reach internal services or cloud metadata endpoints
// (server-side request forgery).
//
// This package does not, and should not, embed a network policy:
// when the URL may derive from untrusted input, supply a restricted client with
// [WithHTTPClient] whose transport rejects unwanted destinations at dial time — which also
// covers redirects and DNS rebinding.
// See the example on [LoadFromFileOrHTTP].
package loading
