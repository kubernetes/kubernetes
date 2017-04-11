// Copyright 2015 The etcd Authors
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

package store

import (
	"testing"
)

// TestIsHidden tests isHidden functions.
func TestIsHidden(t *testing.T) {
	// watch at "/"
	// key is "/_foo", hidden to "/"
	// expected: hidden = true
	watch := "/"
	key := "/_foo"
	hidden := isHidden(watch, key)
	if !hidden {
		t.Fatalf("%v should be hidden to %v\n", key, watch)
	}

	// watch at "/_foo"
	// key is "/_foo", not hidden to "/_foo"
	// expected: hidden = false
	watch = "/_foo"
	hidden = isHidden(watch, key)
	if hidden {
		t.Fatalf("%v should not be hidden to %v\n", key, watch)
	}

	// watch at "/_foo/"
	// key is "/_foo/foo", not hidden to "/_foo"
	key = "/_foo/foo"
	hidden = isHidden(watch, key)
	if hidden {
		t.Fatalf("%v should not be hidden to %v\n", key, watch)
	}

	// watch at "/_foo/"
	// key is "/_foo/_foo", hidden to "/_foo"
	key = "/_foo/_foo"
	hidden = isHidden(watch, key)
	if !hidden {
		t.Fatalf("%v should be hidden to %v\n", key, watch)
	}

	// watch at "/_foo/foo"
	// key is "/_foo"
	watch = "_foo/foo"
	key = "/_foo/"
	hidden = isHidden(watch, key)
	if hidden {
		t.Fatalf("%v should not be hidden to %v\n", key, watch)
	}
}
