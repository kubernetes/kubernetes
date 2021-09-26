// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package util_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/prometheus/procfs/internal/util"
)

func TestValueParser(t *testing.T) {
	tests := []struct {
		name string
		v    string
		ok   bool
		fn   func(t *testing.T, vp *util.ValueParser)
	}{
		{
			name: "ok Int",
			v:    "10",
			ok:   true,
			fn: func(t *testing.T, vp *util.ValueParser) {
				want := 10
				got := vp.Int()

				if diff := cmp.Diff(want, got); diff != "" {
					t.Fatalf("unexpected integer (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "bad PInt64",
			v:    "hello",
			fn: func(_ *testing.T, vp *util.ValueParser) {
				_ = vp.PInt64()
			},
		},
		{
			name: "bad hex PInt64",
			v:    "0xhello",
			fn: func(_ *testing.T, vp *util.ValueParser) {
				_ = vp.PInt64()
			},
		},
		{
			name: "ok PInt64",
			v:    "1",
			ok:   true,
			fn: func(t *testing.T, vp *util.ValueParser) {
				want := int64(1)
				got := vp.PInt64()

				if diff := cmp.Diff(&want, got); diff != "" {
					t.Fatalf("unexpected integer (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "ok hex PInt64",
			v:    "0xff",
			ok:   true,
			fn: func(t *testing.T, vp *util.ValueParser) {
				want := int64(255)
				got := vp.PInt64()

				if diff := cmp.Diff(&want, got); diff != "" {
					t.Fatalf("unexpected integer (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "bad PUInt64",
			v:    "-42",
			fn: func(_ *testing.T, vp *util.ValueParser) {
				_ = vp.PUInt64()
			},
		},
		{
			name: "bad hex PUInt64",
			v:    "0xhello",
			fn: func(_ *testing.T, vp *util.ValueParser) {
				_ = vp.PUInt64()
			},
		},
		{
			name: "ok PUInt64",
			v:    "1",
			ok:   true,
			fn: func(t *testing.T, vp *util.ValueParser) {
				want := uint64(1)
				got := vp.PUInt64()

				if diff := cmp.Diff(&want, got); diff != "" {
					t.Fatalf("unexpected integer (-want +got):\n%s", diff)
				}
			},
		},
		{
			name: "ok hex PUInt64",
			v:    "0xff",
			ok:   true,
			fn: func(t *testing.T, vp *util.ValueParser) {
				want := uint64(255)
				got := vp.PUInt64()

				if diff := cmp.Diff(&want, got); diff != "" {
					t.Fatalf("unexpected integer (-want +got):\n%s", diff)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vp := util.NewValueParser(tt.v)
			tt.fn(t, vp)

			err := vp.Err()
			if err != nil {
				if tt.ok {
					t.Fatalf("unexpected error: %v", err)
				}

				t.Logf("OK err: %v", err)
				return
			}

			if !tt.ok {
				t.Fatal("expected an error, but none occurred")
			}
		})
	}
}
