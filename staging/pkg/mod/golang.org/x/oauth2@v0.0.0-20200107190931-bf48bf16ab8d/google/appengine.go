// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"context"
	"time"

	"golang.org/x/oauth2"
)

// Set at init time by appengine_gen1.go. If nil, we're not on App Engine standard first generation (<= Go 1.9) or App Engine flexible.
var appengineTokenFunc func(c context.Context, scopes ...string) (token string, expiry time.Time, err error)

// Set at init time by appengine_gen1.go. If nil, we're not on App Engine standard first generation (<= Go 1.9) or App Engine flexible.
var appengineAppIDFunc func(c context.Context) string

// AppEngineTokenSource returns a token source that fetches tokens from either
// the current application's service account or from the metadata server,
// depending on the App Engine environment. See below for environment-specific
// details. If you are implementing a 3-legged OAuth 2.0 flow on App Engine that
// involves user accounts, see oauth2.Config instead.
//
// First generation App Engine runtimes (<= Go 1.9):
// AppEngineTokenSource returns a token source that fetches tokens issued to the
// current App Engine application's service account. The provided context must have
// come from appengine.NewContext.
//
// Second generation App Engine runtimes (>= Go 1.11) and App Engine flexible:
// AppEngineTokenSource is DEPRECATED on second generation runtimes and on the
// flexible environment. It delegates to ComputeTokenSource, and the provided
// context and scopes are not used. Please use DefaultTokenSource (or ComputeTokenSource,
// which DefaultTokenSource will use in this case) instead.
func AppEngineTokenSource(ctx context.Context, scope ...string) oauth2.TokenSource {
	return appEngineTokenSource(ctx, scope...)
}
