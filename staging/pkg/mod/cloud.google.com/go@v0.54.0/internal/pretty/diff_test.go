// Copyright 2016 Google LLC
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

package pretty

import "testing"

func TestDiff(t *testing.T) {
	for _, test := range []struct {
		v1, v2 interface{}
		ok     bool
		want   string
	}{
		{5, 5, true, ""},
		{"foo", "foo", true, ""},
		{[]int{1, 2, 3}, []int{1, 0, 3}, false, `--- want
+++ got
@@ -1,5 +1,5 @@
 []int{
     1,
-    2,
+    0,
     3,
 }
`},
	} {
		got, ok, err := Diff(test.v1, test.v2)
		if err != nil {
			t.Errorf("%v vs. %v: %v", test.v1, test.v2, err)
			continue
		}
		if ok != test.ok {
			t.Errorf("%v vs. %v: got %t, want %t", test.v1, test.v2, ok, test.ok)
		}
		if got != test.want {
			t.Errorf("%v vs. %v: got:\n%q\nwant:\n%q", test.v1, test.v2, got, test.want)
		}
	}
}
