// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine appenginevm

// App Engine hooks.

package oauth2

import (
	"net/http"

	"github.com/coreos/etcd/Godeps/_workspace/src/golang.org/x/net/context"
	"google.golang.org/appengine/urlfetch"
)

func init() {
	registerContextClientFunc(contextClientAppEngine)
}

func contextClientAppEngine(ctx context.Context) (*http.Client, error) {
	return urlfetch.Client(ctx), nil
}
