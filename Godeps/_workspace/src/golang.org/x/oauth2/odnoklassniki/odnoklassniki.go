// Copyright 2015 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package odnoklassniki provides constants for using OAuth2 to access Odnoklassniki.
package odnoklassniki

import (
	"golang.org/x/oauth2"
)

// Endpoint is Odnoklassniki's OAuth 2.0 endpoint.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://www.odnoklassniki.ru/oauth/authorize",
	TokenURL: "https://api.odnoklassniki.ru/oauth/token.do",
}
