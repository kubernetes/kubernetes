// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package acme

import (
	"errors"
	"net/http"
	"testing"
	"time"
)

func TestRateLimit(t *testing.T) {
	now := time.Date(2017, 04, 27, 10, 0, 0, 0, time.UTC)
	f := timeNow
	defer func() { timeNow = f }()
	timeNow = func() time.Time { return now }

	h120, hTime := http.Header{}, http.Header{}
	h120.Set("Retry-After", "120")
	hTime.Set("Retry-After", "Tue Apr 27 11:00:00 2017")

	err1 := &Error{
		ProblemType: "urn:ietf:params:acme:error:nolimit",
		Header:      h120,
	}
	err2 := &Error{
		ProblemType: "urn:ietf:params:acme:error:rateLimited",
		Header:      h120,
	}
	err3 := &Error{
		ProblemType: "urn:ietf:params:acme:error:rateLimited",
		Header:      nil,
	}
	err4 := &Error{
		ProblemType: "urn:ietf:params:acme:error:rateLimited",
		Header:      hTime,
	}

	tt := []struct {
		err error
		res time.Duration
		ok  bool
	}{
		{nil, 0, false},
		{errors.New("dummy"), 0, false},
		{err1, 0, false},
		{err2, 2 * time.Minute, true},
		{err3, 0, true},
		{err4, time.Hour, true},
	}
	for i, test := range tt {
		res, ok := RateLimit(test.err)
		if ok != test.ok {
			t.Errorf("%d: RateLimit(%+v): ok = %v; want %v", i, test.err, ok, test.ok)
			continue
		}
		if res != test.res {
			t.Errorf("%d: RateLimit(%+v) = %v; want %v", i, test.err, res, test.res)
		}
	}
}
