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

package authorize

import (
	"net/http"
	"net/url"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/golang/glog"

	"github.com/RangelReale/osin"
)

// AuthenticationHandler reacts to unauthenticated requests
type AuthenticationHandler interface {
	// AuthenticationNeeded reacts to unauthenticated requests, and returns true if it handled writing the response
	AuthenticationNeeded(w http.ResponseWriter, req *http.Request) (handled bool, err error)
}

// Authenticator implements osinserver.AuthorizeHandler to ensure requests are authenticated
type Authenticator struct {
	request authenticator.Request
	handler AuthenticationHandler
}

// NewAuthenticator returns a new Authenticator
func NewAuthenticator(request authenticator.Request, handler AuthenticationHandler) *Authenticator {
	return &Authenticator{request, handler}
}

// HandleAuthorize implements osinserver.AuthorizeHandler to ensure the authorize request is authenticated.
// If the request is authenticated, UserData and Authorized are set and false is returned.
// If the request is not authenticated, the auth handler is called and the request is not authorized
func (h *Authenticator) HandleAuthorize(ar *osin.AuthorizeRequest, w http.ResponseWriter) (handled bool, err error) {
	info, ok, err := h.request.AuthenticateRequest(ar.HttpRequest)
	if err != nil {
		return false, err
	}
	if !ok {
		return h.handler.AuthenticationNeeded(w, ar.HttpRequest)
	}
	ar.UserData = info
	ar.Authorized = true
	return false, nil
}

// emptyAuth is a no-op auth handler
type emptyAuth struct{}

// NewEmptyAuth returns a no-op AuthenticationHandler
func NewEmptyAuth() AuthenticationHandler {
	return emptyAuth{}
}

// AuthenticationNeeded implements a no-op AuthenticationHandler
func (emptyAuth) AuthenticationNeeded(w http.ResponseWriter, req *http.Request) (handled bool, err error) {
	glog.Infof("AuthenticationNeeded")
	return false, nil
}

// RedirectAuthenticator captures the original request url as a "then" param and redirects
type RedirectAuthenticator struct {
	RedirectURL string
	ThenParam   string
}

// NewRedirectAuthenticator returns an AuthenticationHandler that redirects to the specified url
func NewRedirectAuthenticator(url, thenParam string) *RedirectAuthenticator {
	return &RedirectAuthenticator{url, thenParam}
}

// AuthenticationNeeded
func (auth *RedirectAuthenticator) AuthenticationNeeded(w http.ResponseWriter, req *http.Request) (handled bool, err error) {
	redirectURL, err := url.Parse(auth.RedirectURL)
	if err != nil {
		return false, err
	}
	if len(auth.ThenParam) != 0 {
		redirectURL.RawQuery = url.Values{
			auth.ThenParam: {req.URL.String()},
		}.Encode()
	}
	http.Redirect(w, req, redirectURL.String(), http.StatusFound)
	return true, nil
}
