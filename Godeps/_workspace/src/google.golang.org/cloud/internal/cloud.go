// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package internal provides support for the cloud packages.
//
// Users should not import this package directly.
package internal

import (
	"fmt"
	"net/http"
	"sync"

	"golang.org/x/net/context"
)

type contextKey struct{}

func WithContext(parent context.Context, projID string, c *http.Client) context.Context {
	if c == nil {
		panic("nil *http.Client passed to WithContext")
	}
	if projID == "" {
		panic("empty project ID passed to WithContext")
	}
	return context.WithValue(parent, contextKey{}, &cloudContext{
		ProjectID:  projID,
		HTTPClient: c,
	})
}

const userAgent = "gcloud-golang/0.1"

type cloudContext struct {
	ProjectID  string
	HTTPClient *http.Client

	mu  sync.Mutex             // guards svc
	svc map[string]interface{} // e.g. "storage" => *rawStorage.Service
}

// Service returns the result of the fill function if it's never been
// called before for the given name (which is assumed to be an API
// service name, like "datastore"). If it has already been cached, the fill
// func is not run.
// It's safe for concurrent use by multiple goroutines.
func Service(ctx context.Context, name string, fill func(*http.Client) interface{}) interface{} {
	return cc(ctx).service(name, fill)
}

func (c *cloudContext) service(name string, fill func(*http.Client) interface{}) interface{} {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.svc == nil {
		c.svc = make(map[string]interface{})
	} else if v, ok := c.svc[name]; ok {
		return v
	}
	v := fill(c.HTTPClient)
	c.svc[name] = v
	return v
}

// Transport is an http.RoundTripper that appends
// Google Cloud client's user-agent to the original
// request's user-agent header.
type Transport struct {
	// Base represents the actual http.RoundTripper
	// the requests will be delegated to.
	Base http.RoundTripper
}

// RoundTrip appends a user-agent to the existing user-agent
// header and delegates the request to the base http.RoundTripper.
func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = cloneRequest(req)
	ua := req.Header.Get("User-Agent")
	if ua == "" {
		ua = userAgent
	} else {
		ua = fmt.Sprintf("%s;%s", ua, userAgent)
	}
	req.Header.Set("User-Agent", ua)
	return t.Base.RoundTrip(req)
}

// cloneRequest returns a clone of the provided *http.Request.
// The clone is a shallow copy of the struct and its Header map.
func cloneRequest(r *http.Request) *http.Request {
	// shallow copy of the struct
	r2 := new(http.Request)
	*r2 = *r
	// deep copy of the Header
	r2.Header = make(http.Header)
	for k, s := range r.Header {
		r2.Header[k] = s
	}
	return r2
}

func ProjID(ctx context.Context) string {
	return cc(ctx).ProjectID
}

func HTTPClient(ctx context.Context) *http.Client {
	return cc(ctx).HTTPClient
}

// cc returns the internal *cloudContext (cc) state for a context.Context.
// It panics if the user did it wrong.
func cc(ctx context.Context) *cloudContext {
	if c, ok := ctx.Value(contextKey{}).(*cloudContext); ok {
		return c
	}
	panic("invalid context.Context type; it should be created with cloud.NewContext")
}
