// Copyright 2015 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package facebook provides constants for using OAuth2 to access Facebook.
package facebook

import (
	"github.com/coreos/etcd/Godeps/_workspace/src/golang.org/x/oauth2"
)

// Endpoint is Facebook's OAuth 2.0 endpoint.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://www.facebook.com/dialog/oauth",
	TokenURL: "https://graph.facebook.com/oauth/access_token",
}
