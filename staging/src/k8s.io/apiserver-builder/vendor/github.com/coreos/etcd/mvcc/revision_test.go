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

package mvcc

import (
	"bytes"
	"math"
	"reflect"
	"testing"
)

// TestRevision tests that revision could be encoded to and decoded from
// bytes slice. Moreover, the lexicographical order of its byte slice representation
// follows the order of (main, sub).
func TestRevision(t *testing.T) {
	tests := []revision{
		// order in (main, sub)
		{},
		{main: 1, sub: 0},
		{main: 1, sub: 1},
		{main: 2, sub: 0},
		{main: math.MaxInt64, sub: math.MaxInt64},
	}

	bs := make([][]byte, len(tests))
	for i, tt := range tests {
		b := newRevBytes()
		revToBytes(tt, b)
		bs[i] = b

		if grev := bytesToRev(b); !reflect.DeepEqual(grev, tt) {
			t.Errorf("#%d: revision = %+v, want %+v", i, grev, tt)
		}
	}

	for i := 0; i < len(tests)-1; i++ {
		if bytes.Compare(bs[i], bs[i+1]) >= 0 {
			t.Errorf("#%d: %v (%+v) should be smaller than %v (%+v)", i, bs[i], tests[i], bs[i+1], tests[i+1])
		}
	}
}
