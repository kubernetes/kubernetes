// Copyright 2017 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine

// This file provides glue for making github work without App Engine.

package github

import (
	"context"
	"net/http"
)

func withContext(ctx context.Context, req *http.Request) *http.Request {
	return req.WithContext(ctx)
}
