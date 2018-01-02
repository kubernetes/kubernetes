// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package foursquare provides constants for using OAuth2 to access Foursquare.
package foursquare // import "golang.org/x/oauth2/foursquare"

import (
	"golang.org/x/oauth2"
)

// Endpoint is Foursquare's OAuth 2.0 endpoint.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://foursquare.com/oauth2/authorize",
	TokenURL: "https://foursquare.com/oauth2/access_token",
}
