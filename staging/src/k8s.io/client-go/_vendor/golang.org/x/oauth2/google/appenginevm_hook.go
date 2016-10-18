// Copyright 2015 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appenginevm

package google

import "google.golang.org/appengine"

func init() {
	appengineVM = true
	appengineTokenFunc = appengine.AccessToken
}
