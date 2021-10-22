// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mediamath provides constants for using OAuth2 to access MediaMath.
package mediamath // import "golang.org/x/oauth2/mediamath"

import (
	"golang.org/x/oauth2"
)

// Endpoint is MediaMath's OAuth 2.0 endpoint for production.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://api.mediamath.com/oauth2/v1.0/authorize",
	TokenURL: "https://api.mediamath.com/oauth2/v1.0/token",
}

// SandboxEndpoint is MediaMath's OAuth 2.0 endpoint for sandbox.
var SandboxEndpoint = oauth2.Endpoint{
	AuthURL:  "https://t1sandbox.mediamath.com/oauth2/v1.0/authorize",
	TokenURL: "https://t1sandbox.mediamath.com/oauth2/v1.0/token",
}
