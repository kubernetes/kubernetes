// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build appengine
// +build appengine

package internal

import "google.golang.org/appengine/urlfetch"

func init() {
	appengineClientHook = urlfetch.Client
}
