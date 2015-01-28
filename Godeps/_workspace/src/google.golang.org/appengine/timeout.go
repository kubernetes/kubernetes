// Copyright 2013 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package appengine

import (
	"time"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine/internal"
)

// IsTimeoutError reports whether err is a timeout error.
func IsTimeoutError(err error) bool {
	if t, ok := err.(interface {
		IsTimeout() bool
	}); ok {
		return t.IsTimeout()
	}
	return false
}

// Timeout returns a replacement context that uses d as the default API RPC timeout.
func Timeout(c Context, d time.Duration) Context {
	return &timeoutContext{
		Context: c,
		d:       d,
	}
}

type timeoutContext struct {
	Context
	d time.Duration
}

func (t *timeoutContext) Call(service, method string, in, out proto.Message, opts *internal.CallOptions) error {
	// Only affect calls that don't have a timeout.
	if opts == nil || opts.Timeout == 0 {
		newOpts := new(internal.CallOptions)
		if opts != nil {
			*newOpts = *opts
		}
		newOpts.Timeout = t.d
		opts = newOpts
	}
	return t.Context.Call(service, method, in, out, opts)
}
