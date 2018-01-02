// Copyright 2017 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package amazon provides constants for using OAuth2 to access Amazon.
package amazon

import (
	"golang.org/x/oauth2"
)

// Endpoint is Amazon's OAuth 2.0 endpoint.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://www.amazon.com/ap/oa",
	TokenURL: "https://api.amazon.com/auth/o2/token",
}
