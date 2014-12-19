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

package handlers

import (
	"fmt"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/golang/glog"
)

// AuthenticationSuccessHandler reacts to a user authenticating
type AuthenticationSuccessHandler interface {
	// AuthenticationSucceeded reacts to a user authenticating, returns true if the response was handled,
	// and returns false if the response was not handled
	AuthenticationSucceeded(user user.Info, state string, w http.ResponseWriter, req *http.Request) (bool, error)
}

// AuthenticationSuccessHandlers combines multiple AuthenticationSuccessHandler objects into a chain.
// On success, each handler is called. If any handler writes the response or returns an error, the chain is short-circuited
type AuthenticationSuccessHandlers []AuthenticationSuccessHandler

// AuthenticationSucceeded implements AuthenticationSuccessHandler to delegate to a chain of handlers
func (all AuthenticationSuccessHandlers) AuthenticationSucceeded(user user.Info, state string, w http.ResponseWriter, req *http.Request) (bool, error) {
	for _, h := range all {
		if handled, err := h.AuthenticationSucceeded(user, state, w, req); handled || err != nil {
			return handled, err
		}
	}
	return false, nil
}

// RedirectSuccessHandler redirects to the state parameter on successful authentication
type RedirectSuccessHandler struct{}

// AuthenticationSucceeded implements AuthenticationSuccessHandler to redirect to the state parameter
func (RedirectSuccessHandler) AuthenticationSucceeded(user user.Info, state string, w http.ResponseWriter, req *http.Request) (bool, error) {
	if len(state) == 0 {
		return false, fmt.Errorf("Auth succeeded, but no redirect existed - user=%#v", user)
	}

	http.Redirect(w, req, state, http.StatusFound)
	return true, nil
}

// EmptySuccessHandler is a no-op AuthenticationSuccessHandler
type EmptySuccessHandler struct{}

// AuthenticationSucceeded implements AuthenticationSuccessHandler as a no-op
func (EmptySuccessHandler) AuthenticationSucceeded(user user.Info, state string, w http.ResponseWriter, req *http.Request) (bool, error) {
	glog.V(4).Infof("AuthenticationSucceeded: %v (state=%s)", user, state)
	return false, nil
}
