// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
