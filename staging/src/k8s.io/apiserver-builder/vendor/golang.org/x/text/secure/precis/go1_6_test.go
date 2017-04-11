// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.7

package precis

import (
	"fmt"
	"testing"
)

// doTests runs all tests without using t.Run. As a result, context may be
// missing, but at least all tests are run.
func doTests(t *testing.T, fn func(t *testing.T, p *Profile, tc testCase)) {
	for _, g := range testCases {
		for i, tc := range g.cases {
			name := fmt.Sprintf("%s:%d:%+q", g.name, i, tc.input)
			t.Log("Testing ", name)
			fn(t, g.p, tc)
		}
	}
}
