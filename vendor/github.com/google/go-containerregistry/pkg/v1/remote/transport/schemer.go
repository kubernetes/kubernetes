// Copyright 2019 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package transport

import (
	"net/http"

	"github.com/google/go-containerregistry/pkg/name"
)

type schemeTransport struct {
	// Scheme we should use, determined by ping response.
	scheme string

	// Registry we're talking to.
	registry name.Registry

	// Wrapped by schemeTransport.
	inner http.RoundTripper
}

// RoundTrip implements http.RoundTripper
func (st *schemeTransport) RoundTrip(in *http.Request) (*http.Response, error) {
	// When we ping() the registry, we determine whether to use http or https
	// based on which scheme was successful. That is only valid for the
	// registry server and not e.g. a separate token server or blob storage,
	// so we should only override the scheme if the host is the registry.
	if matchesHost(st.registry.String(), in, st.scheme) {
		in.URL.Scheme = st.scheme
	}
	return st.inner.RoundTrip(in)
}
