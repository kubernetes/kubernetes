// Copyright 2018 Google LLC
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

package metadata_test

import (
	"net/http"

	"cloud.google.com/go/compute/metadata"
)

// This example demonstrates how to use your own transport when using this package.
func ExampleNewClient() {
	c := metadata.NewClient(&http.Client{Transport: userAgentTransport{
		userAgent: "my-user-agent",
		base:      http.DefaultTransport,
	}})
	p, err := c.ProjectID()
	if err != nil {
		// TODO: Handle error.
	}
	_ = p // TODO: Use p.
}

// userAgentTransport sets the User-Agent header before calling base.
type userAgentTransport struct {
	userAgent string
	base      http.RoundTripper
}

// RoundTrip implements the http.RoundTripper interface.
func (t userAgentTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Set("User-Agent", t.userAgent)
	return t.base.RoundTrip(req)
}
