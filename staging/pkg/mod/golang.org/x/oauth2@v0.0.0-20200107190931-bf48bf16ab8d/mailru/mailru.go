// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mailru provides constants for using OAuth2 to access Mail.Ru.
package mailru // import "golang.org/x/oauth2/mailru"

import (
	"golang.org/x/oauth2"
)

// Endpoint is Mail.Ru's OAuth 2.0 endpoint.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://o2.mail.ru/login",
	TokenURL: "https://o2.mail.ru/token",
}
