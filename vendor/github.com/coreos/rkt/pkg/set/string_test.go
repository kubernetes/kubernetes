// Copyright 2015 The rkt Authors
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

// For how the uidshift and uidcount are generated please check:
// http://cgit.freedesktop.org/systemd/systemd/commit/?id=03cfe0d51499e86b1573d1

package set

import (
	"strings"
	"testing"
)

func TestString(t *testing.T) {
	s := NewString("a", "b", "c")
	if len(s) != 3 {
		t.Errorf("Expected len=3: %d", len(s))
	}
	if !s.Has("a") || !s.Has("b") || !s.Has("c") {
		t.Errorf("Unexpected contents: %#v", s)
	}

	if !s.HasAll("a", "b", "c") {
		t.Errorf("Unexpected contents: %#v", s)
	}

	if s.HasAll("a", "b", "c", "d") {
		t.Errorf("Unexpected contents: %#v", s)
	}

	s.Delete("a", "b")
	if len(s) != 1 {
		t.Errorf("Expected len=1: %d", len(s))
	}
	if !s.Has("c") {
		t.Errorf("Unexpected contents: %#v", s)
	}

	s.Delete("a")
	if len(s) != 1 {
		t.Errorf("Expected len=1: %d", len(s))
	}
	if !s.Has("c") {
		t.Errorf("Unexpected contents: %#v", s)
	}

	s.Delete("")
	if len(s) != 1 {
		t.Errorf("Expected len=1: %d", len(s))
	}
	if !s.Has("c") {
		t.Errorf("Unexpected contents: %#v", s)
	}

	s.Delete("c")
	if len(s) != 0 {
		t.Errorf("Expected len=1: %d", len(s))
	}
	if s.Has("a") || s.Has("b") || s.Has("c") {
		t.Errorf("Unexpected contents: %#v", s)
	}
}

func TestContitionalHas(t *testing.T) {
	tests := []struct {
		keys          []string
		item          string
		conditionFunc func(string, string) bool
		expectResult  bool
	}{
		{
			[]string{"foo", "bar", "hello"},
			"bar",
			func(a, b string) bool { return a == b },
			true,
		},
		{
			[]string{"foo", "bar", "hello"},
			"baz",
			func(a, b string) bool { return a == b },
			false,
		},
		{
			[]string{"foo", "bar", "hello"},
			"he",
			strings.HasPrefix,
			true,
		},
	}

	for i, tt := range tests {
		s := NewString(tt.keys...)
		actualResult := s.ConditionalHas(tt.conditionFunc, tt.item)
		if tt.expectResult != actualResult {
			t.Errorf("test case %d: expected: %v, saw: %v", i, tt.expectResult, actualResult)
		}
	}
}
