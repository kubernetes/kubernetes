// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package internal contains support packages for oauth2 package.
package internal

import (
	"net/http"

	"golang.org/x/net/context"
)

// HTTPClient is the context key to use with golang.org/x/net/context's
// WithValue function to associate an *http.Client value with a context.
var HTTPClient ContextKey

// ContextKey is just an empty struct. It exists so HTTPClient can be
// an immutable public variable with a unique type. It's immutable
// because nobody else can create a ContextKey, being unexported.
type ContextKey struct{}

// ContextClientFunc is a func which tries to return an *http.Client
// given a Context value. If it returns an error, the search stops
// with that error.  If it returns (nil, nil), the search continues
// down the list of registered funcs.
type ContextClientFunc func(context.Context) (*http.Client, error)

var contextClientFuncs []ContextClientFunc

func RegisterContextClientFunc(fn ContextClientFunc) {
	contextClientFuncs = append(contextClientFuncs, fn)
}

func ContextClient(ctx context.Context) (*http.Client, error) {
	for _, fn := range contextClientFuncs {
		c, err := fn(ctx)
		if err != nil {
			return nil, err
		}
		if c != nil {
			return c, nil
		}
	}
	if hc, ok := ctx.Value(HTTPClient).(*http.Client); ok {
		return hc, nil
	}
	return http.DefaultClient, nil
}

func ContextTransport(ctx context.Context) http.RoundTripper {
	hc, err := ContextClient(ctx)
	// This is a rare error case (somebody using nil on App Engine).
	if err != nil {
		return ErrorTransport{err}
	}
	return hc.Transport
}

// ErrorTransport returns the specified error on RoundTrip.
// This RoundTripper should be used in rare error cases where
// error handling can be postponed to response handling time.
type ErrorTransport struct{ Err error }

func (t ErrorTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, t.Err
}
