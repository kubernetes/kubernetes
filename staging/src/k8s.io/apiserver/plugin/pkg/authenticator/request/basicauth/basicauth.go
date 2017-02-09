/*
Copyright 2014 The Kubernetes Authors.

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
	"net/http"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
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
func (a *Authenticator) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	username, password, found := req.BasicAuth()
	if !found {
		return nil, false, nil
	}
	return a.auth.AuthenticatePassword(username, password)
}
