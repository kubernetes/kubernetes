/*
Copyright 2014 Google Inc. All rights reserved.

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

package basicauth

import (
	"encoding/base64"
	"errors"
	"net/http"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
)

// Authenticator authenticates requests using basic auth
type Authenticator struct {
	auth authenticator.Password
}

// New returns a request authenticator that validates credentials using the provided password authenticator
func New(auth authenticator.Password) *Authenticator {
	return &Authenticator{auth}
}

// AuthenticateRequest authenticates the request using the "Authorization: Basic" header in the request
func (a *Authenticator) AuthenticateRequest(req *http.Request) (user.Info, http.Header, bool, error) {
	challenge := http.Header{"WWW-Authenticate": {"Basic realm=\"" + a.auth.GetRealm() + "\""}}
	auth := strings.TrimSpace(req.Header.Get("Authorization"))
	if auth == "" {
		return nil, challenge, false, nil
	}
	parts := strings.Split(auth, " ")
	if len(parts) < 2 || strings.ToLower(parts[0]) != "basic" {
		return nil, nil, false, nil
	}

	payload, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, challenge, false, err
	}

	pair := strings.SplitN(string(payload), ":", 2)
	if len(pair) != 2 {
		return nil, challenge, false, errors.New("malformed basic auth header")
	}

	username := pair[0]
	password := pair[1]
	return a.auth.AuthenticatePassword(username, password)
}
