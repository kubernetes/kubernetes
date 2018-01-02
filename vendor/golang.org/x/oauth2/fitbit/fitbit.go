// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fitbit provides constants for using OAuth2 to access the Fitbit API.
package fitbit // import "golang.org/x/oauth2/fitbit"

import (
	"golang.org/x/oauth2"
)

// Endpoint is the Fitbit API's OAuth 2.0 endpoint.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://www.fitbit.com/oauth2/authorize",
	TokenURL: "https://api.fitbit.com/oauth2/token",
}
