// Copyright 2016 Google Inc. All Rights Reserved.
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
package devicemapper

import (
	"reflect"
	"testing"
)

func TestParseThinLsOutput(t *testing.T) {
	cases := []struct {
		name           string
		input          string
		expectedResult map[string]uint64
	}{
		{
			name: "ok",
			input: `
  1         2293760
  2         2097152
  3          131072
  4         2031616`,
			expectedResult: map[string]uint64{
				"1": 2293760,
				"2": 2097152,
				"3": 131072,
				"4": 2031616,
			},
		},
		{
			name: "skip bad rows",
			input: `
  1         2293760
  2         2097152
  3          131072ads
  4d dsrv         2031616`,
			expectedResult: map[string]uint64{
				"1": 2293760,
				"2": 2097152,
			},
		},
	}

	for _, tc := range cases {
		actualResult := parseThinLsOutput([]byte(tc.input))
		if e, a := tc.expectedResult, actualResult; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected result: expected %+v got %+v", tc.name, e, a)
		}
	}
}
