/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package websocket

import (
	"encoding/base64"
	"errors"
	"net/http"
	"net/textproto"
	"strings"
	"unicode/utf8"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/util/wsstream"
)

const bearerProtocolPrefix = "base64url.bearer.authorization.k8s.io."

var protocolHeader = textproto.CanonicalMIMEHeaderKey("Sec-WebSocket-Protocol")

var errInvalidToken = errors.New("invalid bearer token")

// ProtocolAuthenticator allows a websocket connection to provide a bearer token as a subprotocol
// in the format "base64url.bearer.authorization.<base64url-without-padding(bearer-token)>"
type ProtocolAuthenticator struct {
	// auth is the token authenticator to use to validate the token
	auth authenticator.Token
}

func NewProtocolAuthenticator(auth authenticator.Token) *ProtocolAuthenticator {
	return &ProtocolAuthenticator{auth}
}

func (a *ProtocolAuthenticator) AuthenticateRequest(req *http.Request) (*authenticator.Response, bool, error) {
	// Only accept websocket connections
	if !wsstream.IsWebSocketRequest(req) {
		return nil, false, nil
	}

	token := ""
	sawTokenProtocol := false
	filteredProtocols := []string{}
	for _, protocolHeader := range req.Header[protocolHeader] {
		for _, protocol := range strings.Split(protocolHeader, ",") {
			protocol = strings.TrimSpace(protocol)

			if !strings.HasPrefix(protocol, bearerProtocolPrefix) {
				filteredProtocols = append(filteredProtocols, protocol)
				continue
			}

			if sawTokenProtocol {
				return nil, false, errors.New("multiple base64.bearer.authorization tokens specified")
			}
			sawTokenProtocol = true

			encodedToken := strings.TrimPrefix(protocol, bearerProtocolPrefix)
			decodedToken, err := base64.RawURLEncoding.DecodeString(encodedToken)
			if err != nil {
				return nil, false, errors.New("invalid base64.bearer.authorization token encoding")
			}
			if !utf8.Valid(decodedToken) {
				return nil, false, errors.New("invalid base64.bearer.authorization token")
			}
			token = string(decodedToken)
		}
	}

	// Must pass at least one other subprotocol so that we can remove the one containing the bearer token,
	// and there is at least one to echo back to the client
	if len(token) > 0 && len(filteredProtocols) == 0 {
		return nil, false, errors.New("missing additional subprotocol")
	}

	if len(token) == 0 {
		return nil, false, nil
	}

	resp, ok, err := a.auth.AuthenticateToken(req.Context(), token)

	// on success, remove the protocol with the token
	if ok {
		// https://tools.ietf.org/html/rfc6455#section-11.3.4 indicates the Sec-WebSocket-Protocol header may appear multiple times
		// in a request, and is logically the same as a single Sec-WebSocket-Protocol header field that contains all values
		req.Header.Set(protocolHeader, strings.Join(filteredProtocols, ","))
	}

	// If the token authenticator didn't error, provide a default error
	if !ok && err == nil {
		err = errInvalidToken
	}

	return resp, ok, err
}
