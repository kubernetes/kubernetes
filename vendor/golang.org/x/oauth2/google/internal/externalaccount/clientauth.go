// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package externalaccount

import (
	"encoding/base64"
	"net/http"
	"net/url"

	"golang.org/x/oauth2"
)

// clientAuthentication represents an OAuth client ID and secret and the mechanism for passing these credentials as stated in rfc6749#2.3.1.
type clientAuthentication struct {
	// AuthStyle can be either basic or request-body
	AuthStyle    oauth2.AuthStyle
	ClientID     string
	ClientSecret string
}

// InjectAuthentication is used to add authentication to a Secure Token Service exchange
// request.  It modifies either the passed url.Values or http.Header depending on the desired
// authentication format.
func (c *clientAuthentication) InjectAuthentication(values url.Values, headers http.Header) {
	if c.ClientID == "" || c.ClientSecret == "" || values == nil || headers == nil {
		return
	}

	switch c.AuthStyle {
	case oauth2.AuthStyleInHeader: // AuthStyleInHeader corresponds to basic authentication as defined in rfc7617#2
		plainHeader := c.ClientID + ":" + c.ClientSecret
		headers.Add("Authorization", "Basic "+base64.StdEncoding.EncodeToString([]byte(plainHeader)))
	case oauth2.AuthStyleInParams: // AuthStyleInParams corresponds to request-body authentication with ClientID and ClientSecret in the message body.
		values.Set("client_id", c.ClientID)
		values.Set("client_secret", c.ClientSecret)
	case oauth2.AuthStyleAutoDetect:
		values.Set("client_id", c.ClientID)
		values.Set("client_secret", c.ClientSecret)
	default:
		values.Set("client_id", c.ClientID)
		values.Set("client_secret", c.ClientSecret)
	}
}
