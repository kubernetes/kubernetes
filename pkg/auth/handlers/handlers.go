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
	"net/http"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/golang/glog"
)

// RequestContext is the interface used to associate a user with an http Request.
type RequestContext interface {
	Set(*http.Request, user.Info)
	Get(req *http.Request) (user.Info, bool)
	Remove(*http.Request)
}

// NewRequestAuthenticator creates an http handler that tries to authenticate the given request as a user, and then
// stores any such user found onto the provided context for the request. If authentication fails or returns an error
// the failed handler is used. On success, handler is invoked to serve the request.
func NewRequestAuthenticator(context RequestContext, auth authenticator.Request, failed http.Handler, handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		user, ok, err := auth.AuthenticateRequest(req)
		if err != nil || !ok {
			if err != nil {
				glog.Errorf("Unable to authenticate the request due to an error: %v", err)
			}
			failed.ServeHTTP(w, req)
			return
		}

		context.Set(req, user)
		defer context.Remove(req)

		handler.ServeHTTP(w, req)
	})
}

var Unauthorized http.HandlerFunc = unauthorized

// unauthorized serves an unauthorized message to clients.
func unauthorized(w http.ResponseWriter, req *http.Request) {
	http.Error(w, "Unauthorized", http.StatusUnauthorized)
}

// UserRequestContext allows different levels of a call stack to store/retrieve info about the
// current user associated with an http.Request.
type UserRequestContext struct {
	requests map[*http.Request]user.Info
	lock     sync.Mutex
}

// NewUserRequestContext provides a map for storing and retrieving users associated with requests.
// Be sure to pair each `context.Set(req, user)` call with a `defer context.Remove(req)` call or
// you will leak requests. It implements the RequestContext interface.
func NewUserRequestContext() *UserRequestContext {
	return &UserRequestContext{
		requests: make(map[*http.Request]user.Info),
	}
}

func (c *UserRequestContext) Get(req *http.Request) (user.Info, bool) {
	c.lock.Lock()
	defer c.lock.Unlock()
	user, ok := c.requests[req]
	return user, ok
}

func (c *UserRequestContext) Set(req *http.Request, user user.Info) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.requests[req] = user
}

func (c *UserRequestContext) Remove(req *http.Request) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.requests, req)
}
