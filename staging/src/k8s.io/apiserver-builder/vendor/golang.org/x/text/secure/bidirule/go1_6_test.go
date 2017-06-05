// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.7

package bidirule

import (
	"fmt"
	"testing"
)

// doTests runs all tests without using t.Run. As a result, context may be
// missing, but at least all tests are run.
func doTests(t *testing.T, fn func(t *testing.T, tc ruleTest)) {
	for rule, cases := range testCases {
		for i, tc := range cases {
			name := fmt.Sprintf("%d/%d:%+q:%s", rule, i, tc.in, tc.in)
			t.Log("Testing ", name)
			fn(t, tc)
		}
	}
}
