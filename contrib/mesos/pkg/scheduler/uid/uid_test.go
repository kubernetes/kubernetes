/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package uid

import (
	"testing"
)

func TestUID_Parse(t *testing.T) {
	valid := []string{"1234567890abcdef_foo", "123_bar", "face_time"}
	groups := []uint64{0x1234567890abcdef, 0x123, 0xface}

	for i, good := range valid {
		u := Parse(good)
		if u == nil {
			t.Errorf("expected parsed UID, not nil")
		}
		if groups[i] != u.Group() {
			t.Errorf("expected matching group instead of %x", u.Group())
		}
		if good != u.String() {
			t.Errorf("expected %q instead of %q", good, u.String())
		}
	}

	invalid := []string{"", "bad"}
	for _, bad := range invalid {
		u := Parse(bad)
		if u != nil {
			t.Errorf("expected nil UID instead of %v", u)
		}
	}
}
