// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

//go:build !appengine
// +build !appengine

package appengine

import (
	"context"
)

// BackgroundContext returns a context not associated with a request.
//
// Deprecated: App Engine no longer has a special background context.
// Just use context.Background().
func BackgroundContext() context.Context {
	return context.Background()
}
