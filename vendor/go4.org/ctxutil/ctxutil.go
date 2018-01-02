/*
Copyright 2015 The Go4 Authors

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

// Package ctxutil contains golang.org/x/net/context related utilities.
package ctxutil // import "go4.org/ctxutil"

import (
	"net/http"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
)

// HTTPClient is the context key to use with golang.org/x/net/context's WithValue function
// to associate an *http.Client value with a context.
//
// We use the same value as the oauth2 package (which first introduced this key) rather
// than creating a new one and forcing users to possibly set two.
var HTTPClient = oauth2.HTTPClient

// Client returns the HTTP client to use for the provided context.
// If ctx is non-nil and has an associated HTTP client, that client is returned.
// Otherwise, http.DefaultClient is returned.
func Client(ctx context.Context) *http.Client {
	if ctx != nil {
		if hc, ok := ctx.Value(HTTPClient).(*http.Client); ok {
			return hc
		}
	}
	return http.DefaultClient
}
