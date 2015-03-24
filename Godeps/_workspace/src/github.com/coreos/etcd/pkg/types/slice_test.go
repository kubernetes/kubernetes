// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"reflect"
	"sort"
	"testing"
)

func TestUint64Slice(t *testing.T) {
	g := Uint64Slice{10, 500, 5, 1, 100, 25}
	w := Uint64Slice{1, 5, 10, 25, 100, 500}
	sort.Sort(g)
	if !reflect.DeepEqual(g, w) {
		t.Errorf("slice after sort = %#v, want %#v", g, w)
	}
}
