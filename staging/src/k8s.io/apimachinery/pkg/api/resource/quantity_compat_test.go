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
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"testing"
	"time"
)

func TestQuantityOutOfInt32ExponentCompatibility(t *testing.T) {
	testCases := []struct {
		in          string
		wantValue   int64
		wantFitsI64 bool
		wantString  string
		want        *Quantity
	}{
		{in: "1e4294967296", wantValue: 1, wantFitsI64: true, wantString: "1e4294967296"},
		{in: "1e4294967297", wantValue: 10, wantFitsI64: true, wantString: "10"},
		{in: "-1e4294967296", wantValue: -1, wantFitsI64: true, wantString: "-1e4294967296"},
		{in: "1e-4294967296", wantValue: 1, wantFitsI64: true, wantString: "1e-4294967296"},
		{in: "1e8589934592", wantValue: 1, wantFitsI64: true, wantString: "1e8589934592"},

		// Fraction digits move the scale off the exponent, so these are the
		// spellings at 2147483648 that a <=1.37 apiserver could write. None
		// of them fits an int64, so their value is compared with what 1.37 parsed.
		{in: "1.25e2147483648", wantString: "1.25e2147483648", want: NewScaledQuantity(125, 2147483646)},
		{in: "1.5e2147483648", wantString: "150e2147483646", want: NewScaledQuantity(15, 2147483647)},
		{in: "1.25e-2147483648", wantString: "1.25e-2147483648", want: NewScaledQuantity(125, 2147483646)},
		{in: "0.5e2147483648", wantString: "50e2147483646", want: NewScaledQuantity(5, 2147483647)},
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
			if tc.want != nil && q.Cmp(*tc.want) != 0 {
				t.Errorf("value = %s, want %s", rawValue(&q), rawValue(tc.want))
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
			if tc.want != nil && fromJSON.Cmp(*tc.want) != 0 {
				t.Errorf("after json decode: value = %s, want %s", rawValue(&fromJSON), rawValue(tc.want))
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
			if tc.want != nil && fromProto.Cmp(*tc.want) != 0 {
				t.Errorf("after proto decode: value = %s, want %s", rawValue(&fromProto), rawValue(tc.want))
			}
		})
	}
}

// rawValue formats the value by hand: String() can return the parsed input unchanged.
func rawValue(q *Quantity) string {
	if q.d.Dec != nil {
		return fmt.Sprintf("%ve%d", q.d.Dec.UnscaledBig(), -int64(q.d.Dec.Scale()))
	}
	return fmt.Sprintf("%de%d", q.i.value, q.i.scale)
}

// zeroMinInt32Spellings are zeros whose exponent is MinInt32, directly or once narrowed.
// 1.37 parsed every one of them as 0 and wrote it as "0".
var zeroMinInt32Spellings = []string{
	"0e2147483648", "+0e2147483648", "-0e2147483648",
	"0e-2147483648", "+0e-2147483648", "-0e-2147483648",
	"0.000000000000000000000e2147483648",
}

// TestQuantityOutOfInt32ExponentZero pins the 1.37 behavior: the zeros parse, read as 0 and print as "0".
func TestQuantityOutOfInt32ExponentZero(t *testing.T) {
	for _, in := range zeroMinInt32Spellings {
		t.Run(in, func(t *testing.T) {
			q, err := ParseQuantity(in)
			if err != nil {
				t.Fatalf("ParseQuantity(%q) failed: %v", in, err)
			}
			if !q.IsZero() || q.String() != "0" || q.Format != DecimalExponent {
				t.Errorf("ParseQuantity(%q) = %s (%s), want 0 (%s)", in, q.String(), q.Format, DecimalExponent)
			}
			var fromJSON Quantity
			if err := json.Unmarshal([]byte(strconv.Quote(in)), &fromJSON); err != nil {
				t.Fatalf("json.Unmarshal(%q) failed: %v", in, err)
			}
			if !fromJSON.IsZero() || fromJSON.String() != "0" {
				t.Errorf("after json decode: %s, want 0", fromJSON.String())
			}
		})
	}
}

// zeroAddHelperEnv marks the child process that TestQuantityOutOfInt32ExponentZeroAdd starts.
const zeroAddHelperEnv = "KUBE_QUANTITY_ZERO_ADD_HELPER"

// TestQuantityOutOfInt32ExponentZeroAdd pins the exponent reset, which 1.37 did not have:
// with the exponent left at MinInt32, Add panics in inf.Dec.rescale or does not return.
// The cases run in a child process with its own timeout, so a regression is stopped instead of left running.
func TestQuantityOutOfInt32ExponentZeroAdd(t *testing.T) {
	if os.Getenv(zeroAddHelperEnv) == "1" {
		for _, in := range zeroMinInt32Spellings {
			q, err := ParseQuantity(in)
			if err != nil {
				t.Fatalf("ParseQuantity(%q) failed: %v", in, err)
			}
			q.Add(MustParse("1"))
			if got := q.String(); got != "1" {
				t.Errorf("%q: Add(1) = %s, want 1", in, got)
			}
		}
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	// The child's own timeout is shorter than ctx, so a hang prints its stack, and it still applies if this process times out first.
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestQuantityOutOfInt32ExponentZeroAdd$", "-test.count=1", "-test.timeout=25s")
	cmd.Env = append(os.Environ(), zeroAddHelperEnv+"=1")
	cmd.WaitDelay = time.Second
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("helper did not finish within 30s:\n%s", out)
	}
	if err != nil {
		t.Fatalf("helper failed: %v\n%s", err, out)
	}
}
