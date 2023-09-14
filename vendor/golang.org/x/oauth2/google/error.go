// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"errors"

	"golang.org/x/oauth2"
)

// AuthenticationError indicates there was an error in the authentication flow.
//
// Use (*AuthenticationError).Temporary to check if the error can be retried.
type AuthenticationError struct {
	err *oauth2.RetrieveError
}

func newAuthenticationError(err error) error {
	re := &oauth2.RetrieveError{}
	if !errors.As(err, &re) {
		return err
	}
	return &AuthenticationError{
		err: re,
	}
}

// Temporary indicates that the network error has one of the following status codes and may be retried: 500, 503, 408, or 429.
func (e *AuthenticationError) Temporary() bool {
	if e.err.Response == nil {
		return false
	}
	sc := e.err.Response.StatusCode
	return sc == 500 || sc == 503 || sc == 408 || sc == 429
}

func (e *AuthenticationError) Error() string {
	return e.err.Error()
}

func (e *AuthenticationError) Unwrap() error {
	return e.err
}

type errWrappingTokenSource struct {
	src oauth2.TokenSource
}

func newErrWrappingTokenSource(ts oauth2.TokenSource) oauth2.TokenSource {
	return &errWrappingTokenSource{src: ts}
}

// Token returns the current token if it's still valid, else will
// refresh the current token (using r.Context for HTTP client
// information) and return the new one.
func (s *errWrappingTokenSource) Token() (*oauth2.Token, error) {
	t, err := s.src.Token()
	if err != nil {
		return nil, newAuthenticationError(err)
	}
	return t, nil
}
