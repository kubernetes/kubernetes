/*
Copyright 2015 Google Inc. All rights reserved.

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

package negotiate

import (
	"net/http"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
)

type Authenticator struct {
	auth authenticator.Token
}

func New(auth authenticator.Token) *Authenticator {
	return &Authenticator{auth}
}

func (a *Authenticator) AuthenticateRequest(req *http.Request) (user.Info, http.Header, bool, error) {
	challenge := http.Header{"WWW-Authenticate": {"Negotiate"}}
	auth := strings.TrimSpace(req.Header.Get("Authorization"))
	if auth == "" {
		return nil, challenge, false, nil
	}
	parts := strings.SplitN(auth, " ", 2)
	if len(parts) < 2 || strings.ToLower(parts[0]) != "negotiate" {
		return nil, nil, false, nil
	}

	b64token := strings.Replace(parts[1], " ", "", -1)
	return a.auth.AuthenticateToken(b64token)
}
