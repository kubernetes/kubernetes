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

package common

import (
	"fmt"
	"testing"

	"github.com/appc/spec/schema/types"
)

func TestVolumesToKvmDiskArgs(t *testing.T) {
	tests := []struct {
		volumes  []types.Volume
		expected []string
	}{
		{ // one host volume - one argument
			volumes:  []types.Volume{{Name: types.ACName("foo"), Kind: "host", Source: "src1"}},
			expected: []string{fmt.Sprintf("--9p=src1,%s", makeHashFromVolumeName("foo"))},
		},
		{ // on empty volume - no arguments
			volumes:  []types.Volume{{Name: types.ACName("foo"), Kind: "empty", Source: "src1"}},
			expected: []string{},
		},
		{ // two host volumes
			volumes: []types.Volume{
				{Name: types.ACName("foo"), Kind: "host", Source: "src1"},
				{Name: types.ACName("bar"), Kind: "host", Source: "src2"},
			},
			expected: []string{fmt.Sprintf("--9p=src1,%s", makeHashFromVolumeName("foo")),
				fmt.Sprintf("--9p=src2,%s", makeHashFromVolumeName("bar"))},
		},
		{ // mix host and empty
			volumes: []types.Volume{
				{Name: types.ACName("foo"), Kind: "host", Source: "src1"},
				{Name: types.ACName("baz"), Kind: "empty", Source: "src1"},
				{Name: types.ACName("bar"), Kind: "host", Source: "src2"},
			},
			expected: []string{fmt.Sprintf("--9p=src1,%s", makeHashFromVolumeName("foo")),
				fmt.Sprintf("--9p=src2,%s", makeHashFromVolumeName("bar"))},
		},
	}

	for i, tt := range tests {
		got := VolumesToKvmDiskArgs(tt.volumes)
		if len(got) != len(tt.expected) {
			t.Errorf("#%d: expected %v elements got %v", i, len(tt.expected), len(got))
		} else {
			for iarg, argExpected := range tt.expected {
				if got[iarg] != argExpected {
					t.Errorf("#%d: arg %d expected `%v` got `%v`", i, iarg, argExpected, got[iarg])
				}
			}
		}
	}
}
