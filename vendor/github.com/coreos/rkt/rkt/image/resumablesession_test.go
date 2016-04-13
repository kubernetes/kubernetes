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

package image

import (
	"testing"
)

func TestGetMaxAge(t *testing.T) {
	// members are not important
	s := &resumableSession{}
	tests := []struct {
		value  string
		maxAge int
	}{
		{
			value:  "max-age=10",
			maxAge: 10,
		},
		{
			value:  "no-cache",
			maxAge: 0,
		},
		{
			value:  "no-store",
			maxAge: 0,
		},
	}
	for _, tt := range tests {
		got := s.getMaxAge(tt.value)
		if tt.maxAge != got {
			t.Errorf("expected max-age of %d from %q, got %d", tt.maxAge, tt.value, got)
		}
	}
}
