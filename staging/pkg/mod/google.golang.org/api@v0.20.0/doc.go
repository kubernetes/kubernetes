// Copyright 2019 Google LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package api is the root of the packages used to access Google Cloud
// Services. See https://godoc.org/google.golang.org/api for a full list of
// sub-packages.
//
// Within api there exist numerous clients which connect to Google APIs,
// and various utility packages.
//
//
// Client Options
//
// All clients in sub-packages are configurable via client options. These
// options are described here: https://godoc.org/google.golang.org/api/option.
//
//
// Authentication and Authorization
//
// All the clients in sub-packages support authentication via Google
// Application Default Credentials (see
// https://cloud.google.com/docs/authentication/production), or by providing a
// JSON key file for a Service Account. See the authentication examples in
// https://godoc.org/google.golang.org/api/transport for more details.
//
//
// Versioning and Stability
//
// Due to the auto-generated nature of this collection of libraries, complete
// APIs or specific versions can appear or go away without notice. As a result,
// you should always locally vendor any API(s) that your code relies upon.
//
// Google APIs follow semver as specified by
// https://cloud.google.com/apis/design/versioning. The code generator and
// the code it produces - the libraries in the google.golang.org/api/...
// subpackages - are beta.
//
// Note that versioning and stability is strictly not communicated through Go
// modules. Go modules are used only for dependency management.
//
//
// Integers
//
// Many parameters are specified using ints. However, underlying APIs might
// operate on a finer granularity, expecting int64, int32, uint64, or uint32,
// all of whom have different maximum values. Subsequently, specifying an int
// parameter in one of these clients may result in an error from the API
// because the value is too large.
//
// To see the exact type of int that the API expects, you can inspect the API's
// discovery doc. A global catalogue pointing to the discovery doc of APIs can
// be found at https://www.googleapis.com/discovery/v1/apis.
package api
