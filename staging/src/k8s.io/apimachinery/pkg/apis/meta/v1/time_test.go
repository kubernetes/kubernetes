/*
Copyright 2014 The Kubernetes Authors.

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

package v1

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/internal/encodingtest"
	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/randfill"
)

func TestTimeMarshalYAML(t *testing.T) {
	//nolint:govet // Unkeyed fields are fine for test cases.
	cases := []encodingtest.MarshalCase[Time]{
		{Time{}, "null\n"},
		{Date(1998, time.May, 5, 1, 5, 5, 50, time.FixedZone("test", -4*60*60)), "\"1998-05-05T05:05:05Z\"\n"},
		{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC), "\"1998-05-05T05:05:05Z\"\n"},
	}

	encodingtest.MarshalYAML(t, cases)
}

func TestTimeUnmarshalYAML(t *testing.T) {
	//nolint:govet // Unkeyed fields are fine for test cases.
	cases := []encodingtest.UnmarshalCase[Time]{
		{"null\n", Time{}},
		{"1998-05-05T05:05:05Z\n", Time{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC).Local()}},
	}

	encodingtest.UnmarshalYAML(t, cases)
}

func timeMarshalJSONCases() []encodingtest.MarshalCase[Time] {
	//nolint:govet // Unkeyed fields are fine for test cases.
	return []encodingtest.MarshalCase[Time]{
		{Time{}, "null"},
		{Date(1998, time.May, 5, 5, 5, 5, 50, time.UTC), "\"1998-05-05T05:05:05Z\""},
		{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC), "\"1998-05-05T05:05:05Z\""},
		{Date(1998, time.May, 5, 5, 5, 5, 0, time.FixedZone("x", 2*3600)), `"1998-05-05T03:05:05Z"`},
		{Date(999, time.May, 5, 5, 5, 5, 0, time.UTC), `"0999-05-05T05:05:05Z"`},
	}
}

func timeUnmarshalJSONCases() []encodingtest.UnmarshalCase[Time] {
	//nolint:govet // Unkeyed fields are fine for test cases.
	return []encodingtest.UnmarshalCase[Time]{
		{"null", Time{}},
		{"\"1998-05-05T05:05:05Z\"", Time{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC).Local()}},
		{"\"1998-05-05T05:05:05.123456789Z\"", Time{Date(1998, time.May, 5, 5, 5, 5, 123456789, time.UTC).Local()}},
		{`"1998-05-05T05:05:05+02:00"`, Time{Date(1998, time.May, 5, 3, 5, 5, 0, time.UTC).Local()}},
		{`"1998-05-05T05:05:05\u005a"`, Time{Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC).Local()}},
	}
}

func TestTimeMarshalJSON(t *testing.T) {
	encodingtest.MarshalJSON(t, timeMarshalJSONCases())
}

func TestTimeUnmarshalJSON(t *testing.T) {
	encodingtest.UnmarshalJSON(t, timeUnmarshalJSONCases())
	encodingtest.UnmarshalJSONErrors[Time](t, []string{
		`5`,
		`true`,
		`{}`,
		`[]`,
		`""`,
		`"garbage"`,
		`"1998-05-05"`,
	})
}

func BenchmarkTimeMarshalJSON(b *testing.B) {
	encodingtest.BenchmarkMarshalJSON(b, timeMarshalJSONCases())
}

func BenchmarkTimeUnmarshalJSON(b *testing.B) {
	encodingtest.BenchmarkUnmarshalJSON(b, timeUnmarshalJSONCases())
}

func TestTimeMarshalJSONUnmarshalYAML(t *testing.T) {
	inputs := []Time{
		{},
		Date(1998, time.May, 5, 5, 5, 5, 50, time.Local).Rfc3339Copy(),
		Date(1998, time.May, 5, 5, 5, 5, 0, time.Local).Rfc3339Copy(),
	}

	encodingtest.JSONToYAML(t, inputs, func(a, b Time) bool {
		aN, aO := a.Zone()
		bN, bO := b.Zone()
		return aN == bN && aO == bO && a.UnixNano() == b.UnixNano()
	})
}

func TestTimeMarshalCBOR(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   Time
		out  []byte
	}{
		{name: "zero value", in: Time{}, out: []byte{0xf6}},                                                                                        // null
		{name: "no fractional seconds", in: Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC), out: []byte("\x541998-05-05T05:05:05Z")},                // '1998-05-05T05:05:05Z'
		{name: "fractional seconds truncated", in: Date(1998, time.May, 5, 5, 5, 5, 123456789, time.UTC), out: []byte("\x541998-05-05T05:05:05Z")}, // '1998-05-05T05:05:05Z'
		{name: "epoch", in: Time{Time: time.Unix(0, 0)}, out: []byte("\x541970-01-01T00:00:00Z")},                                                  // '1970-01-01T00:00:00Z'
		{name: "pre-epoch", in: Date(1960, time.January, 1, 0, 0, 0, 0, time.UTC), out: []byte("\x541960-01-01T00:00:00Z")},                        // '1960-01-01T00:00:00Z'
	} {
		t.Run(fmt.Sprintf("%+v", tc.in), func(t *testing.T) {
			got, err := tc.in.MarshalCBOR()
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tc.out, got); diff != "" {
				t.Errorf("unexpected output:\n%s", diff)
			}
		})
	}
}

func TestTimeUnmarshalCBOR(t *testing.T) {
	for _, tc := range []struct {
		name       string
		in         []byte
		out        Time
		errMessage string
	}{
		{name: "null", in: []byte{0xf6}, out: Time{}}, // null
		{name: "no fractional seconds", in: []byte("\x58\x141998-05-05T05:05:05Z"), out: Time{Time: Date(1998, time.May, 5, 5, 5, 5, 0, time.UTC).Local()}},                    // '1998-05-05T05:05:05Z'
		{name: "fractional seconds", in: []byte("\x58\x1e1998-05-05T05:05:05.123456789Z"), out: Time{Time: Date(1998, time.May, 5, 5, 5, 5, 123456789, time.UTC).Local()}},     // '1998-05-05T05:05:05.123456789Z'
		{name: "invalid cbor type", in: []byte{0x07}, out: Time{}, errMessage: "cbor: cannot unmarshal positive integer into Go value of type string"},                         // 7
		{name: "malformed timestamp", in: []byte("\x45hello"), out: Time{}, errMessage: `parsing time "hello" as "2006-01-02T15:04:05Z07:00": cannot parse "hello" as "2006"`}, // 'hello'
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got Time
			err := got.UnmarshalCBOR(tc.in)
			if err != nil {
				if tc.errMessage == "" {
					t.Fatalf("want nil error, got: %v", err)
				} else if gotMessage := err.Error(); tc.errMessage != gotMessage {
					t.Fatalf("want error: %q, got: %q", tc.errMessage, gotMessage)
				}
			} else if tc.errMessage != "" {
				t.Fatalf("got nil error, want: %s", tc.errMessage)
			}
			if diff := cmp.Diff(tc.out, got); diff != "" {
				t.Errorf("unexpected output:\n%s", diff)
			}
		})
	}
}

func TestTimeProto(t *testing.T) {
	cases := []struct {
		input Time
	}{
		{Time{}},
		{Date(1998, time.May, 5, 1, 5, 5, 0, time.Local)},
		{Date(1998, time.May, 5, 5, 5, 5, 0, time.Local)},
	}

	for _, c := range cases {
		input := c.input
		data, err := input.Marshal()
		if err != nil {
			t.Fatalf("Failed to marshal input: '%v': %v", input, err)
		}
		time := Time{}
		if err := time.Unmarshal(data); err != nil {
			t.Fatalf("Failed to unmarshal output: '%v': %v", input, err)
		}
		if !reflect.DeepEqual(input, time) {
			t.Errorf("Marshal->Unmarshal is not idempotent: '%v' vs '%v'", input, time)
		}
	}
}

func TestTimeEqual(t *testing.T) {
	t1 := NewTime(time.Now())
	cases := []struct {
		name   string
		x      *Time
		y      *Time
		result bool
	}{
		{"nil =? nil", nil, nil, true},
		{"!nil =? !nil", &t1, &t1, true},
		{"nil =? !nil", nil, &t1, false},
		{"!nil =? nil", &t1, nil, false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result := c.x.Equal(c.y)
			if result != c.result {
				t.Errorf("Failed equality test for '%v', '%v': expected %+v, got %+v", c.x, c.y, c.result, result)
			}
		})
	}
}

func TestTimeBefore(t *testing.T) {
	t1 := NewTime(time.Now())
	cases := []struct {
		name string
		x    *Time
		y    *Time
	}{
		{"nil <? nil", nil, nil},
		{"!nil <? !nil", &t1, &t1},
		{"nil <? !nil", nil, &t1},
		{"!nil <? nil", &t1, nil},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result := c.x.Before(c.y)
			if result {
				t.Errorf("Failed equality test for '%v', '%v': expected false, got %+v", c.x, c.y, result)
			}
		})
	}
}

func TestTimeIsZero(t *testing.T) {
	t1 := NewTime(time.Now())
	cases := []struct {
		name   string
		x      *Time
		result bool
	}{
		{"nil =? 0", nil, true},
		{"!nil =? 0", &t1, false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result := c.x.IsZero()
			if result != c.result {
				t.Errorf("Failed equality test for '%v': expected %+v, got %+v", c.x, c.result, result)
			}
		})
	}
}

func TestTimeRoundtripCBOR(t *testing.T) {
	fuzzer := randfill.New()
	for i := 0; i < 500; i++ {
		var initial, final Time
		fuzzer.Fill(&initial)
		b, err := cbor.Marshal(initial)
		if err != nil {
			t.Errorf("error encoding %v: %v", initial, err)
			continue
		}
		err = cbor.Unmarshal(b, &final)
		if err != nil {
			t.Errorf("%v: error decoding %v: %v", initial, string(b), err)
		}
		if !final.Equal(&initial) {
			diag, err := cbor.Diagnose(b)
			if err != nil {
				t.Logf("failed to produce diagnostic encoding of 0x%x: %v", b, err)
			}
			t.Errorf("expected equal: %v, %v (cbor was '%s')", initial, final, diag)
		}
	}
}
