// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine,!appenginevm

// App Engine hooks.

package oauth2

import (
	"log"
	"net/http"
	"sync"

	"appengine"
	"appengine/urlfetch"
)

var warnOnce sync.Once

func init() {
	registerContextClientFunc(contextClientAppEngine)
}

func contextClientAppEngine(ctx Context) (*http.Client, error) {
	if actx, ok := ctx.(appengine.Context); ok {
		return urlfetch.Client(actx), nil
	}
	// The user did it wrong. We'll log once (and hope they see it
	// in dev_appserver), but stil return (nil, nil) in case some
	// other contextClientFunc hook finds a way to proceed.
	warnOnce.Do(gaeDoingItWrongHelp)
	return nil, nil
}

func gaeDoingItWrongHelp() {
	log.Printf("WARNING: you attempted to use the oauth2 package without passing a valid appengine.Context or *http.Request as the oauth2.Context. App Engine requires that all service RPCs (including urlfetch) be associated with an *http.Request/appengine.Context.")
}
