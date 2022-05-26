// Copyright 2017 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

// This file provides glue for making github work on App Engine.

package github

import (
	"context"
	"net/http"
)

func withContext(ctx context.Context, req *http.Request) *http.Request {
	// No-op because App Engine adds context to a request differently.
	return req
}
