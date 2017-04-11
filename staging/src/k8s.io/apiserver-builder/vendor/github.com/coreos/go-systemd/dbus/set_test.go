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

// TestBasicSetActions asserts that Add & Remove behavior is correct
func TestBasicSetActions(t *testing.T) {
	s := newSet()

	if s.Contains("foo") {
		t.Fatal("set should not contain 'foo'")
	}

	s.Add("foo")

	if !s.Contains("foo") {
		t.Fatal("set should contain 'foo'")
	}

	v := s.Values()
	if len(v) != 1 {
		t.Fatal("set.Values did not report correct number of values")
	}
	if v[0] != "foo" {
		t.Fatal("set.Values did not report value")
	}

	s.Remove("foo")

	if s.Contains("foo") {
		t.Fatal("set should not contain 'foo'")
	}

	v = s.Values()
	if len(v) != 0 {
		t.Fatal("set.Values did not report correct number of values")
	}
}
