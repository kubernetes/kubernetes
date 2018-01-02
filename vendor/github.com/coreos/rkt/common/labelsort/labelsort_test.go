// Copyright 2016 The rkt Authors
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

package labelsort

import (
	"testing"

	"github.com/appc/spec/schema/types"
)

func TestRankedName(t *testing.T) {
	tests := []struct {
		li string
		lj string

		expected bool
	}{
		{
			"version", "os",
			true,
		},
		{
			"os", "version",
			false,
		},
		{
			"version", "a",
			true,
		},
		{
			"a", "os",
			false,
		},
		{
			"os", "a",
			true,
		},
		{
			"", "os",
			false,
		},
		{
			"os", "",
			true,
		},
		{
			"a", "version",
			false,
		},
		{
			"a", "b",
			true,
		},
		{
			"b", "a",
			false,
		},
		{
			"a", "a",
			false,
		},
		{
			"version", "version",
			false,
		},
	}

	for i, tt := range tests {
		li := types.Label{
			Name: types.ACIdentifier(tt.li),
		}

		lj := types.Label{
			Name: types.ACIdentifier(tt.lj),
		}

		if result := RankedName(li, lj); result != tt.expected {
			t.Errorf("test %d expected %q < %q = %t but got %t", i, tt.li, tt.lj, tt.expected, result)
		}
	}
}
