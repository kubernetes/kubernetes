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

package rest

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"k8s.io/klog"
)

// TokenSourceWrapTransport returns a WrapTransport that injects bearer tokens
// authentication from an oauth2.TokenSource.
func TokenSourceWrapTransport(ts oauth2.TokenSource) func(http.RoundTripper) http.RoundTripper {
	return func(rt http.RoundTripper) http.RoundTripper {
		return &tokenSourceTransport{
			base: rt,
			ort: &oauth2.Transport{
				Source: ts,
				Base:   rt,
			},
		}
	}
}

// NewCachedFileTokenSource returns a oauth2.TokenSource reads a token from a
// file at a specified path and periodically reloads it.
func NewCachedFileTokenSource(path string) oauth2.TokenSource {
	return &cachingTokenSource{
		now:    time.Now,
		leeway: 1 * time.Minute,
		base: &fileTokenSource{
			path: path,
			// This period was picked because it is half of the minimum validity
			// duration for a token provisioned by they TokenRequest API. This is
			// unsophisticated and should induce rotation at a frequency that should
			// work with the token volume source.
			period: 5 * time.Minute,
		},
	}
}

type tokenSourceTransport struct {
	base http.RoundTripper
	ort  http.RoundTripper
}

func (tst *tokenSourceTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// This is to allow --token to override other bearer token providers.
	if req.Header.Get("Authorization") != "" {
		return tst.base.RoundTrip(req)
	}
	return tst.ort.RoundTrip(req)
}

type fileTokenSource struct {
	path   string
	period time.Duration
}

var _ = oauth2.TokenSource(&fileTokenSource{})

func (ts *fileTokenSource) Token() (*oauth2.Token, error) {
	tokb, err := ioutil.ReadFile(ts.path)
	if err != nil {
		return nil, fmt.Errorf("failed to read token file %q: %v", ts.path, err)
	}
	tok := strings.TrimSpace(string(tokb))
	if len(tok) == 0 {
		return nil, fmt.Errorf("read empty token from file %q", ts.path)
	}

	return &oauth2.Token{
		AccessToken: tok,
		Expiry:      time.Now().Add(ts.period),
	}, nil
}

type cachingTokenSource struct {
	base   oauth2.TokenSource
	leeway time.Duration

	sync.RWMutex
	tok *oauth2.Token

	// for testing
	now func() time.Time
}

var _ = oauth2.TokenSource(&cachingTokenSource{})

func (ts *cachingTokenSource) Token() (*oauth2.Token, error) {
	now := ts.now()
	// fast path
	ts.RLock()
	tok := ts.tok
	ts.RUnlock()

	if tok != nil && tok.Expiry.Add(-1*ts.leeway).After(now) {
		return tok, nil
	}

	// slow path
	ts.Lock()
	defer ts.Unlock()
	if tok := ts.tok; tok != nil && tok.Expiry.Add(-1*ts.leeway).After(now) {
		return tok, nil
	}

	tok, err := ts.base.Token()
	if err != nil {
		if ts.tok == nil {
			return nil, err
		}
		klog.Errorf("Unable to rotate token: %v", err)
		return ts.tok, nil
	}

	ts.tok = tok
	return tok, nil
}
