// Copyright 2017 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"reflect"
	"testing"
)

func TestCompareSlices(t *testing.T) {
	for _, test := range []struct {
		a, b      []int
		wantEqual bool
	}{
		{nil, nil, true},
		{nil, []int{}, true},
		{[]int{1, 2}, []int{1, 2}, true},
		{[]int{1}, []int{1, 2}, false},
		{[]int{1, 2}, []int{1}, false},
		{[]int{1, 2}, []int{1, 3}, false},
	} {
		_, got := compareSlices(reflect.ValueOf(test.a), reflect.ValueOf(test.b))
		if got != test.wantEqual {
			t.Errorf("%v, %v: got %t, want %t", test.a, test.b, got, test.wantEqual)
		}
	}
}
