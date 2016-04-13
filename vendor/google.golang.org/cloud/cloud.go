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

// Package cloud contains Google Cloud Platform APIs related types
// and common functions.
package cloud // import "google.golang.org/cloud"

import (
	"net/http"

	"golang.org/x/net/context"
	"google.golang.org/cloud/internal"
)

// NewContext returns a new context that uses the provided http.Client.
// Provided http.Client is responsible to authorize and authenticate
// the requests made to the Google Cloud APIs.
// It mutates the client's original Transport to append the cloud
// package's user-agent to the outgoing requests.
// You can obtain the project ID from the Google Developers Console,
// https://console.developers.google.com.
func NewContext(projID string, c *http.Client) context.Context {
	if c == nil {
		panic("invalid nil *http.Client passed to NewContext")
	}
	return WithContext(context.Background(), projID, c)
}

// WithContext returns a new context in a similar way NewContext does,
// but initiates the new context with the specified parent.
func WithContext(parent context.Context, projID string, c *http.Client) context.Context {
	// TODO(bradfitz): delete internal.Transport. It's too wrappy for what it does.
	// Do User-Agent some other way.
	if c == nil {
		panic("invalid nil *http.Client passed to WithContext")
	}
	if _, ok := c.Transport.(*internal.Transport); !ok {
		base := c.Transport
		if base == nil {
			base = http.DefaultTransport
		}
		c.Transport = &internal.Transport{Base: base}
	}
	return internal.WithContext(parent, projID, c)
}
