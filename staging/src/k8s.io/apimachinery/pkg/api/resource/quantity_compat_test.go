/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package resource

import (
	"encoding/json"
	"strconv"
	"testing"
)

func TestQuantityOutOfInt32ExponentCompatibility(t *testing.T) {
	testCases := []struct {
		in          string
		wantValue   int64
		wantFitsI64 bool
		wantString  string
	}{
		{in: "1e4294967296", wantValue: 1, wantFitsI64: true, wantString: "1e4294967296"},
		{in: "1e4294967297", wantValue: 10, wantFitsI64: true, wantString: "10"},
		{in: "-1e4294967296", wantValue: -1, wantFitsI64: true, wantString: "-1e4294967296"},
		{in: "1e-4294967296", wantValue: 1, wantFitsI64: true, wantString: "1e-4294967296"},
		{in: "1e8589934592", wantValue: 1, wantFitsI64: true, wantString: "1e8589934592"},

		// Fraction digits move the scale off the exponent, so these are the
		// spellings at 2147483648 that a <=1.37 apiserver could write. None
		// of them fits an int64, so only the spelling has to survive.
		{in: "1.25e2147483648", wantString: "1.25e2147483648"},
		{in: "1.5e2147483648", wantString: "150e2147483646"},
		{in: "1.25e-2147483648", wantString: "1.25e-2147483648"},
		{in: "0.5e2147483648", wantString: "50e2147483646"},
	}

	for _, tc := range testCases {
		t.Run(tc.in, func(t *testing.T) {
			q, err := ParseQuantity(tc.in)
			if err != nil {
				t.Fatalf("ParseQuantity(%q) failed: %v", tc.in, err)
			}
			// Only the flag is portable; the value returned alongside a
			// false has changed between releases.
			if got, ok := q.AsInt64(); ok != tc.wantFitsI64 || (ok && got != tc.wantValue) {
				t.Errorf("AsInt64() = (%d, %v), want (%d, %v)", got, ok, tc.wantValue, tc.wantFitsI64)
			}
			if got := q.String(); got != tc.wantString {
				t.Errorf("String() = %q, want %q", got, tc.wantString)
			}

			// Decoding is the path that matters: the spelling below is what a
			// 1.37 apiserver wrote into etcd.
			var fromJSON Quantity
			if err := json.Unmarshal([]byte(strconv.Quote(tc.in)), &fromJSON); err != nil {
				t.Fatalf("json.Unmarshal(%q) failed: %v", tc.in, err)
			}
			if got, ok := fromJSON.AsInt64(); ok != tc.wantFitsI64 || (ok && got != tc.wantValue) {
				t.Errorf("after json decode: AsInt64() = (%d, %v), want (%d, %v)", got, ok, tc.wantValue, tc.wantFitsI64)
			}

			// Re-encoding must not rewrite what is already stored.
			data, err := json.Marshal(&fromJSON)
			if err != nil {
				t.Fatalf("json.Marshal failed: %v", err)
			}
			if got, want := string(data), strconv.Quote(tc.wantString); got != want {
				t.Errorf("json.Marshal = %s, want %s", got, want)
			}

			// 1.37 stored the same spelling through protobuf.
			buf, err := q.Marshal()
			if err != nil {
				t.Fatalf("proto Marshal failed: %v", err)
			}
			var fromProto Quantity
			if err := fromProto.Unmarshal(buf); err != nil {
				t.Fatalf("proto Unmarshal failed: %v", err)
			}
			if got, ok := fromProto.AsInt64(); ok != tc.wantFitsI64 || (ok && got != tc.wantValue) {
				t.Errorf("after proto decode: AsInt64() = (%d, %v), want (%d, %v)", got, ok, tc.wantValue, tc.wantFitsI64)
			}
		})
	}
}
