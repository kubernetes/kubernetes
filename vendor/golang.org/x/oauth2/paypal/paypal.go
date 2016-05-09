// Copyright 2015 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package paypal provides constants for using OAuth2 to access PayPal.
package paypal

import (
	"golang.org/x/oauth2"
)

// Endpoint is PayPal's OAuth 2.0 endpoint in live (production) environment.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://www.paypal.com/webapps/auth/protocol/openidconnect/v1/authorize",
	TokenURL: "https://api.paypal.com/v1/identity/openidconnect/tokenservice",
}

// SandboxEndpoint is PayPal's OAuth 2.0 endpoint in sandbox (testing) environment.
var SandboxEndpoint = oauth2.Endpoint{
	AuthURL:  "https://www.sandbox.paypal.com/webapps/auth/protocol/openidconnect/v1/authorize",
	TokenURL: "https://api.sandbox.paypal.com/v1/identity/openidconnect/tokenservice",
}
