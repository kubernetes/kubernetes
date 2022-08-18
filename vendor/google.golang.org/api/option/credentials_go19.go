// Copyright 2018 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.9
// +build go1.9

package option

import (
	"golang.org/x/oauth2/google"
	"google.golang.org/api/internal"
)

type withCreds google.Credentials

func (w *withCreds) Apply(o *internal.DialSettings) {
	o.Credentials = (*google.Credentials)(w)
}

// WithCredentials returns a ClientOption that authenticates API calls.
func WithCredentials(creds *google.Credentials) ClientOption {
	return (*withCreds)(creds)
}
