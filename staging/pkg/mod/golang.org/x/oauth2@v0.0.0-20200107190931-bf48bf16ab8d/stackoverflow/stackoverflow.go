// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stackoverflow provides constants for using OAuth2 to access Stack Overflow.
package stackoverflow // import "golang.org/x/oauth2/stackoverflow"

import (
	"golang.org/x/oauth2"
)

// Endpoint is Stack Overflow's OAuth 2.0 endpoint.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://stackoverflow.com/oauth",
	TokenURL: "https://stackoverflow.com/oauth/access_token",
}
