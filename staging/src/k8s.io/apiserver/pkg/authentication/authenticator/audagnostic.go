/*
Copyright 2018 The Kubernetes Authors.

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

package authenticator

import (
	"context"
	"fmt"
	"net/http"
)

func authenticate(ctx context.Context, implicitAuds Audiences, authenticate func() (*Response, bool, error)) (*Response, bool, error) {
	targetAuds, ok := AudiencesFrom(ctx)
	// We can remove this once api audiences is never empty. That will probably
	// be N releases after TokenRequest is GA.
	if !ok {
		return authenticate()
	}
	auds := implicitAuds.Intersect(targetAuds)
	if len(auds) == 0 {
		return nil, false, nil
	}
	resp, ok, err := authenticate()
	if err != nil || !ok {
		return nil, false, err
	}
	if len(resp.Audiences) > 0 {
		// maybe the authenticator was audience aware after all.
		return nil, false, fmt.Errorf("audience agnostic authenticator wrapped an authenticator that returned audiences: %q", resp.Audiences)
	}
	resp.Audiences = auds
	return resp, true, nil
}

type audAgnosticRequestAuthenticator struct {
	implicit Audiences
	delegate Request
}

var _ = Request(&audAgnosticRequestAuthenticator{})

func (a *audAgnosticRequestAuthenticator) AuthenticateRequest(req *http.Request) (*Response, bool, error) {
	return authenticate(req.Context(), a.implicit, func() (*Response, bool, error) {
		return a.delegate.AuthenticateRequest(req)
	})
}

// WrapAudienceAgnosticRequest wraps an audience agnostic request authenticator
// to restrict its accepted audiences to a set of implicit audiences.
func WrapAudienceAgnosticRequest(implicit Audiences, delegate Request) Request {
	return &audAgnosticRequestAuthenticator{
		implicit: implicit,
		delegate: delegate,
	}
}

type audAgnosticTokenAuthenticator struct {
	implicit Audiences
	delegate Token
}

var _ = Token(&audAgnosticTokenAuthenticator{})

func (a *audAgnosticTokenAuthenticator) AuthenticateToken(ctx context.Context, tok string) (*Response, bool, error) {
	return authenticate(ctx, a.implicit, func() (*Response, bool, error) {
		return a.delegate.AuthenticateToken(ctx, tok)
	})
}

// WrapAudienceAgnosticToken wraps an audience agnostic token authenticator to
// restrict its accepted audiences to a set of implicit audiences.
func WrapAudienceAgnosticToken(implicit Audiences, delegate Token) Token {
	return &audAgnosticTokenAuthenticator{
		implicit: implicit,
		delegate: delegate,
	}
}
