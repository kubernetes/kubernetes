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
	"fmt"
	"net/http"
	"runtime/debug"
)

var (
	// Version can be set via:
	// -ldflags="-X 'github.com/google/go-containerregistry/pkg/v1/remote/transport.Version=$TAG'"
	Version string

	ggcrVersion = defaultUserAgent
)

const (
	defaultUserAgent = "go-containerregistry"
	moduleName       = "github.com/google/go-containerregistry"
)

type userAgentTransport struct {
	inner http.RoundTripper
	ua    string
}

func init() {
	if v := version(); v != "" {
		ggcrVersion = fmt.Sprintf("%s/%s", defaultUserAgent, v)
	}
}

func version() string {
	if Version != "" {
		// Version was set via ldflags, just return it.
		return Version
	}

	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}

	// Happens for crane and gcrane.
	if info.Main.Path == moduleName {
		return info.Main.Version
	}

	// Anything else.
	for _, dep := range info.Deps {
		if dep.Path == moduleName {
			return dep.Version
		}
	}

	return ""
}

// NewUserAgent returns an http.Roundtripper that sets the user agent to
// The provided string plus additional go-containerregistry information,
// e.g. if provided "crane/v0.1.4" and this modules was built at v0.1.4:
//
// User-Agent: crane/v0.1.4 go-containerregistry/v0.1.4
func NewUserAgent(inner http.RoundTripper, ua string) http.RoundTripper {
	if ua == "" {
		ua = ggcrVersion
	} else {
		ua = fmt.Sprintf("%s %s", ua, ggcrVersion)
	}
	return &userAgentTransport{
		inner: inner,
		ua:    ua,
	}
}

// RoundTrip implements http.RoundTripper
func (ut *userAgentTransport) RoundTrip(in *http.Request) (*http.Response, error) {
	in.Header.Set("User-Agent", ut.ua)
	return ut.inner.RoundTrip(in)
}
