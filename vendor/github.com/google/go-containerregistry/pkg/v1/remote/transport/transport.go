// Copyright 2018 Google LLC All Rights Reserved.
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
	"context"
	"net/http"
	"strings"

	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
)

// New returns a new RoundTripper based on the provided RoundTripper that has been
// setup to authenticate with the remote registry "reg", in the capacity
// laid out by the specified scopes.
//
// Deprecated: Use NewWithContext.
func New(reg name.Registry, auth authn.Authenticator, t http.RoundTripper, scopes []string) (http.RoundTripper, error) {
	return NewWithContext(context.Background(), reg, auth, t, scopes)
}

// NewWithContext returns a new RoundTripper based on the provided RoundTripper that has been
// set up to authenticate with the remote registry "reg", in the capacity
// laid out by the specified scopes.
// In case the RoundTripper is already of the type Wrapper it assumes
// authentication was already done prior to this call, so it just returns
// the provided RoundTripper without further action
func NewWithContext(ctx context.Context, reg name.Registry, auth authn.Authenticator, t http.RoundTripper, scopes []string) (http.RoundTripper, error) {
	// When the transport provided is of the type Wrapper this function assumes that the caller already
	// executed the necessary login and check.
	switch t.(type) {
	case *Wrapper:
		return t, nil
	}
	// The handshake:
	//  1. Use "t" to ping() the registry for the authentication challenge.
	//
	//  2a. If we get back a 200, then simply use "t".
	//
	//  2b. If we get back a 401 with a Basic challenge, then use a transport
	//     that just attachs auth each roundtrip.
	//
	//  2c. If we get back a 401 with a Bearer challenge, then use a transport
	//     that attaches a bearer token to each request, and refreshes is on 401s.
	//     Perform an initial refresh to seed the bearer token.

	// First we ping the registry to determine the parameters of the authentication handshake
	// (if one is even necessary).
	pr, err := Ping(ctx, reg, t)
	if err != nil {
		return nil, err
	}

	// Wrap t with a useragent transport unless we already have one.
	if _, ok := t.(*userAgentTransport); !ok {
		t = NewUserAgent(t, "")
	}

	scheme := "https"
	if pr.Insecure {
		scheme = "http"
	}

	// Wrap t in a transport that selects the appropriate scheme based on the ping response.
	t = &schemeTransport{
		scheme:   scheme,
		registry: reg,
		inner:    t,
	}

	if strings.ToLower(pr.Scheme) != "bearer" {
		return &Wrapper{&basicTransport{inner: t, auth: auth, target: reg.RegistryStr()}}, nil
	}

	bt, err := fromChallenge(reg, auth, t, pr)
	if err != nil {
		return nil, err
	}
	bt.scopes = scopes

	if err := bt.refresh(ctx); err != nil {
		return nil, err
	}
	return &Wrapper{bt}, nil
}

// Wrapper results in *not* wrapping supplied transport with additional logic such as retries, useragent and debug logging
// Consumers are opt-ing into providing their own transport without any additional wrapping.
type Wrapper struct {
	inner http.RoundTripper
}

// RoundTrip delegates to the inner RoundTripper
func (w *Wrapper) RoundTrip(in *http.Request) (*http.Response, error) {
	return w.inner.RoundTrip(in)
}
