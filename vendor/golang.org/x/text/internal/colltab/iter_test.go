// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colltab

import (
	"testing"
)

func TestDoNorm(t *testing.T) {
	const div = -1 // The insertion point of the next block.
	tests := []struct {
		in, out []int
	}{{
		in:  []int{4, div, 3},
		out: []int{3, 4},
	}, {
		in:  []int{4, div, 3, 3, 3},
		out: []int{3, 3, 3, 4},
	}, {
		in:  []int{0, 4, div, 3},
		out: []int{0, 3, 4},
	}, {
		in:  []int{0, 0, 4, 5, div, 3, 3},
		out: []int{0, 0, 3, 3, 4, 5},
	}, {
		in:  []int{0, 0, 1, 4, 5, div, 3, 3},
		out: []int{0, 0, 1, 3, 3, 4, 5},
	}, {
		in:  []int{0, 0, 1, 4, 5, div, 4, 4},
		out: []int{0, 0, 1, 4, 4, 4, 5},
	},
	}
	for j, tt := range tests {
		i := Iter{}
		var w, p int
		for k, cc := range tt.in {

			if cc == div {
				w = 100
				p = k
				continue
			}
			i.Elems = append(i.Elems, makeCE([]int{w, defaultSecondary, 2, cc}))
		}
		i.doNorm(p, i.Elems[p].CCC())
		if len(i.Elems) != len(tt.out) {
			t.Errorf("%d: length was %d; want %d", j, len(i.Elems), len(tt.out))
		}
		prevCCC := uint8(0)
		for k, ce := range i.Elems {
			if int(ce.CCC()) != tt.out[k] {
				t.Errorf("%d:%d: unexpected CCC. Was %d; want %d", j, k, ce.CCC(), tt.out[k])
			}
			if k > 0 && ce.CCC() == prevCCC && i.Elems[k-1].Primary() > ce.Primary() {
				t.Errorf("%d:%d: normalization crossed across CCC boundary.", j, k)
			}
		}
	}

	// Combining rune overflow is tested in search/pattern_test.go.
}
