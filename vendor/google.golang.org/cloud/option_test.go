// Copyright 2015 Google Inc. All Rights Reserved.
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

package cloud

import (
	"testing"

	"google.golang.org/cloud/internal/opts"
)

// Check that the slice passed into WithScopes is copied.
func TestCopyScopes(t *testing.T) {
	o := &opts.DialOpt{}

	scopes := []string{"a", "b"}
	WithScopes(scopes...).Resolve(o)

	// Modify after using.
	scopes[1] = "c"

	if o.Scopes[0] != "a" || o.Scopes[1] != "b" {
		t.Errorf("want ['a', 'b'], got %+v", o.Scopes)
	}
}

// Check that resolution of WithScopes uses the most recent.
// That is, it doesn't append scopes or ignore subsequent calls.
func TestScopesOverride(t *testing.T) {
	o := &opts.DialOpt{}

	WithScopes("a").Resolve(o)
	WithScopes("b").Resolve(o)

	if want, got := 1, len(o.Scopes); want != got {
		t.Errorf("want %d scope, got %d", want, got)
	}

	if want, got := "b", o.Scopes[0]; want != got {
		t.Errorf("want %s scope, got %s", want, got)
	}
}
