// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.7

package precis

import (
	"fmt"
	"testing"
)

func doTests(t *testing.T, fn func(t *testing.T, p *Profile, tc testCase)) {
	for _, g := range testCases {
		for i, tc := range g.cases {
			name := fmt.Sprintf("%s:%d:%+q", g.name, i, tc.input)
			t.Run(name, func(t *testing.T) {
				fn(t, g.p, tc)
			})
		}
	}
}
