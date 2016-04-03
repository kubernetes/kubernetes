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

package dbus

import (
	"testing"
)

func TestNeedsEscape(t *testing.T) {
	// Anything not 0-9a-zA-Z should always be escaped
	for want, vals := range map[bool][]byte{
		false: []byte{'a', 'b', 'z', 'A', 'Q', '1', '4', '9'},
		true:  []byte{'#', '%', '$', '!', '.', '_', '-', '%', '\\'},
	} {
		for i := 1; i < 10; i++ {
			for _, b := range vals {
				got := needsEscape(i, b)
				if got != want {
					t.Errorf("needsEscape(%d, %c) returned %t, want %t", i, b, got, want)
				}
			}
		}
	}

	// 0-9 in position 0 should be escaped
	for want, vals := range map[bool][]byte{
		false: []byte{'A', 'a', 'e', 'x', 'Q', 'Z'},
		true:  []byte{'0', '4', '5', '9'},
	} {
		for _, b := range vals {
			got := needsEscape(0, b)
			if got != want {
				t.Errorf("needsEscape(0, %c) returned %t, want %t", b, got, want)
			}
		}
	}

}

func TestPathBusEscape(t *testing.T) {
	for in, want := range map[string]string{
		"":                   "_",
		"foo.service":        "foo_2eservice",
		"foobar":             "foobar",
		"woof@woof.service":  "woof_40woof_2eservice",
		"0123456":            "_30123456",
		"account_db.service": "account_5fdb_2eservice",
		"got-dashes":         "got_2ddashes",
	} {
		got := PathBusEscape(in)
		if got != want {
			t.Errorf("bad result for PathBusEscape(%s): got %q, want %q", in, got, want)
		}
	}

}

// TestNew ensures that New() works without errors.
func TestNew(t *testing.T) {
	_, err := New()

	if err != nil {
		t.Fatal(err)
	}
}
