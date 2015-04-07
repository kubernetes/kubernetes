// Copyright 2013 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package appengine

import (
	"testing"
	"time"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine/internal"
)

type timeoutRecorder struct {
	Context
	d time.Duration
}

func (tr *timeoutRecorder) Call(_, _ string, _, _ proto.Message, opts *internal.CallOptions) error {
	tr.d = 5 * time.Second // default
	if opts != nil {
		tr.d = opts.Timeout
	}
	return nil
}

func TestTimeout(t *testing.T) {
	tests := []struct {
		desc string
		opts *internal.CallOptions
		want time.Duration
	}{
		{
			"no opts",
			nil,
			6 * time.Second,
		},
		{
			"empty opts",
			&internal.CallOptions{},
			6 * time.Second,
		},
		{
			"set opts",
			&internal.CallOptions{Timeout: 7 * time.Second},
			7 * time.Second,
		},
	}
	for _, test := range tests {
		tr := new(timeoutRecorder)
		c := Timeout(tr, 6*time.Second)
		c.Call("service", "method", nil, nil, test.opts)
		if tr.d != test.want {
			t.Errorf("%s: timeout was %v, want %v", test.desc, tr.d, test.want)
		}
	}
}
