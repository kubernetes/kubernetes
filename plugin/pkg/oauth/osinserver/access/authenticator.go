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

package access

import (
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth"
	"github.com/RangelReale/osin"
	"github.com/golang/glog"
)

// Authenticator implements osinserver.AccessHandler to ensure non-token requests are authenticated
type Authenticator struct {
	password  authenticator.Password
	assertion authenticator.Assertion
	client    oauth.ClientAuthenticator
}

// NewAuthenticator returns a new Authenticator
func NewAuthenticator(password authenticator.Password, assertion authenticator.Assertion, client oauth.ClientAuthenticator) *Authenticator {
	return &Authenticator{password, assertion, client}
}

// HandleAccess implements osinserver.AccessHandler
func (h *Authenticator) HandleAccess(ar *osin.AccessRequest, w http.ResponseWriter) error {
	var info user.Info
	var ok bool
	var err error

	switch ar.Type {
	case osin.AUTHORIZATION_CODE, osin.REFRESH_TOKEN:
		// auth codes and refresh tokens are already enforced
		ok = true
	case osin.PASSWORD:
		info, ok, err = h.password.AuthenticatePassword(ar.Username, ar.Password)
	case osin.ASSERTION:
		info, ok, err = h.assertion.AuthenticateAssertion(ar.AssertionType, ar.Assertion)
	case osin.CLIENT_CREDENTIALS:
		info, ok, err = h.client.AuthenticateClient(ar.Client)
	default:
		glog.Warningf("unknown access type: %s", ar.Type)
	}

	if err != nil {
		return err
	}
	if ok {
		ar.Authorized = true
		if info != nil {
			ar.AccessData.UserData = info
		}
	}
	return nil
}

// NewDenyAuthenticator returns an Authenticator which rejects all non-token access requests
func NewDenyAuthenticator() *Authenticator {
	return &Authenticator{Deny, Deny, Deny}
}

// Deny implements Password, Assertion, and Client authentication to deny all requests
var Deny = &fixedAuthenticator{false}

// Allow implements Password, Assertion, and Client authentication to allow all requests
var Allow = &fixedAuthenticator{true}

// fixedAuthenticator implements Password, Assertion, and Client authentication to return a fixed response
type fixedAuthenticator struct {
	allow bool
}

// AuthenticatePassword implements authenticator.Password
func (f *fixedAuthenticator) AuthenticatePassword(user, password string) (user.Info, bool, error) {
	return nil, f.allow, nil
}

// AuthenticateAssertion implements authenticator.Assertion
func (f *fixedAuthenticator) AuthenticateAssertion(assertionType, data string) (user.Info, bool, error) {
	return nil, f.allow, nil
}

// AuthenticateClient implements authenticator.Client
func (f *fixedAuthenticator) AuthenticateClient(client oauth.Client) (user.Info, bool, error) {
	return nil, f.allow, nil
}
