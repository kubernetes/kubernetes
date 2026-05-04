// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package swag

import (
	"encoding/json"
	"time"

	"github.com/go-openapi/swag/loading"
)

var (
	// Package-level defaults for the file loading utilities (deprecated).

	// LoadHTTPTimeout the default timeout for load requests.
	//
	// Deprecated: use [loading.WithTimeout] instead.
	LoadHTTPTimeout = 30 * time.Second

	// LoadHTTPBasicAuthUsername the username to use when load requests require basic auth.
	//
	// Deprecated: use [loading.WithBasicAuth] instead.
	LoadHTTPBasicAuthUsername = ""

	// LoadHTTPBasicAuthPassword the password to use when load requests require basic auth.
	//
	// Deprecated: use [loading.WithBasicAuth] instead.
	LoadHTTPBasicAuthPassword = ""

	// LoadHTTPCustomHeaders an optional collection of custom HTTP headers for load requests.
	//
	// Deprecated: use [loading.WithCustomHeaders] instead.
	LoadHTTPCustomHeaders = map[string]string{}
)

// LoadFromFileOrHTTP loads the bytes from a file or a remote http server based on the provided path.
//
// Deprecated: use [loading.LoadFromFileOrHTTP] instead.
func LoadFromFileOrHTTP(pth string, opts ...loading.Option) ([]byte, error) {
	return loading.LoadFromFileOrHTTP(pth, loadingOptionsWithDefaults(opts)...)
}

// LoadFromFileOrHTTPWithTimeout loads the bytes from a file or a remote http server based on the path passed in
// timeout arg allows for per request overriding of the request timeout.
//
// Deprecated: use [loading.LoadFileOrHTTP] with the [loading.WithTimeout] option instead.
func LoadFromFileOrHTTPWithTimeout(pth string, timeout time.Duration, opts ...loading.Option) ([]byte, error) {
	opts = append(opts, loading.WithTimeout(timeout))

	return LoadFromFileOrHTTP(pth, opts...)
}

// LoadStrategy returns a loader function for a given path or URL.
//
// Deprecated: use [loading.LoadStrategy] instead.
func LoadStrategy(pth string, local, remote func(string) ([]byte, error), opts ...loading.Option) func(string) ([]byte, error) {
	return loading.LoadStrategy(pth, local, remote, loadingOptionsWithDefaults(opts)...)
}

// YAMLMatcher matches yaml for a file loader.
//
// Deprecated: use [loading.YAMLMatcher] instead.
func YAMLMatcher(path string) bool { return loading.YAMLMatcher(path) }

// YAMLDoc loads a yaml document from either http or a file and converts it to json.
//
// Deprecated: use [loading.YAMLDoc] instead.
func YAMLDoc(path string) (json.RawMessage, error) {
	return loading.YAMLDoc(path)
}

// YAMLData loads a yaml document from either http or a file.
//
// Deprecated: use [loading.YAMLData] instead.
func YAMLData(path string) (any, error) {
	return loading.YAMLData(path)
}

// loadingOptionsWithDefaults bridges deprecated default settings that use package-level variables,
// with the recommended use of loading.Option.
func loadingOptionsWithDefaults(opts []loading.Option) []loading.Option {
	o := []loading.Option{
		loading.WithTimeout(LoadHTTPTimeout),
		loading.WithBasicAuth(LoadHTTPBasicAuthUsername, LoadHTTPBasicAuthPassword),
		loading.WithCustomHeaders(LoadHTTPCustomHeaders),
	}
	o = append(o, opts...)

	return o
}
