// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package slack provides constants for using OAuth2 to access Slack.
package slack // import "golang.org/x/oauth2/slack"

import (
	"golang.org/x/oauth2"
)

// Endpoint is Slack's OAuth 2.0 endpoint.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://slack.com/oauth/authorize",
	TokenURL: "https://slack.com/api/oauth.access",
}
