/*
Copyright 2019 The Kubernetes Authors.

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

package json

import (
	gojson "encoding/json"
	"strings"
	"testing"

	utiljson "k8s.io/apimachinery/pkg/util/json"
)

type testcase struct {
	name     string
	data     []byte
	checkErr func(t testing.TB, err error)

	benchmark bool
}

func testcases() []testcase {
	// verify we got an error of some kind
	nonNilError := func(t testing.TB, err error) {
		if err == nil {
			t.Errorf("expected error, got none")
		}
	}
	// verify the parse completed, either with success or a max depth error
	successOrMaxDepthError := func(t testing.TB, err error) {
		if err != nil && !strings.Contains(err.Error(), "max depth") {
			t.Errorf("expected success or error containing 'max depth', got: %v", err)
		}
	}

	return []testcase{
		{
			name:     "3MB of deeply nested slices",
			checkErr: successOrMaxDepthError,
			data:     []byte(`{"a":` + strings.Repeat(`[`, 3*1024*1024/2) + strings.Repeat(`]`, 3*1024*1024/2) + "}"),
		},
		{
			name:     "3MB of unbalanced nested slices",
			checkErr: nonNilError,
			data:     []byte(`{"a":` + strings.Repeat(`[`, 3*1024*1024)),
		},
		{
			name:     "3MB of deeply nested maps",
			checkErr: successOrMaxDepthError,
			data:     []byte(strings.Repeat(`{"":`, 3*1024*1024/5/2) + "{}" + strings.Repeat(`}`, 3*1024*1024/5/2)),
		},
		{
			name:     "3MB of unbalanced nested maps",
			checkErr: nonNilError,
			data:     []byte(strings.Repeat(`{"":`, 3*1024*1024/5)),
		},
		{
			name:      "3MB of empty slices",
			data:      []byte(`{"a":[` + strings.Repeat(`[],`, 3*1024*1024/3-2) + `[]]}`),
			benchmark: true,
		},
		{
			name:      "3MB of slices",
			data:      []byte(`{"a":[` + strings.Repeat(`[0],`, 3*1024*1024/4-2) + `[0]]}`),
			benchmark: true,
		},
		{
			name:      "3MB of empty maps",
			data:      []byte(`{"a":[` + strings.Repeat(`{},`, 3*1024*1024/3-2) + `{}]}`),
			benchmark: true,
		},
		{
			name:      "3MB of maps",
			data:      []byte(`{"a":[` + strings.Repeat(`{"a":0},`, 3*1024*1024/8-2) + `{"a":0}]}`),
			benchmark: true,
		},
		{
			name:      "3MB of ints",
			data:      []byte(`{"a":[` + strings.Repeat(`0,`, 3*1024*1024/2-2) + `0]}`),
			benchmark: true,
		},
		{
			name:      "3MB of floats",
			data:      []byte(`{"a":[` + strings.Repeat(`0.0,`, 3*1024*1024/4-2) + `0.0]}`),
			benchmark: true,
		},
		{
			name:      "3MB of bools",
			data:      []byte(`{"a":[` + strings.Repeat(`true,`, 3*1024*1024/5-2) + `true]}`),
			benchmark: true,
		},
		{
			name:      "3MB of empty strings",
			data:      []byte(`{"a":[` + strings.Repeat(`"",`, 3*1024*1024/3-2) + `""]}`),
			benchmark: true,
		},
		{
			name:      "3MB of strings",
			data:      []byte(`{"a":[` + strings.Repeat(`"abcdefghijklmnopqrstuvwxyz012",`, 3*1024*1024/30-2) + `"abcdefghijklmnopqrstuvwxyz012"]}`),
			benchmark: true,
		},
		{
			name:      "3MB of nulls",
			data:      []byte(`{"a":[` + strings.Repeat(`null,`, 3*1024*1024/5-2) + `null]}`),
			benchmark: true,
		},
	}
}

var decoders = map[string]func([]byte, interface{}) error{
	"gojson":   gojson.Unmarshal,
	"utiljson": utiljson.Unmarshal,
}

func TestJSONLimits(t *testing.T) {
	for _, tc := range testcases() {
		if tc.benchmark {
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			for decoderName, decoder := range decoders {
				t.Run(decoderName, func(t *testing.T) {
					v := map[string]interface{}{}
					err := decoder(tc.data, &v)

					if tc.checkErr != nil {
						tc.checkErr(t, err)
					} else if err != nil {
						t.Errorf("unexpected error: %v", err)
					}
				})
			}
		})
	}
}

func BenchmarkJSONLimits(b *testing.B) {
	for _, tc := range testcases() {
		b.Run(tc.name, func(b *testing.B) {
			for decoderName, decoder := range decoders {
				b.Run(decoderName, func(b *testing.B) {
					for i := 0; i < b.N; i++ {
						v := map[string]interface{}{}
						err := decoder(tc.data, &v)

						if tc.checkErr != nil {
							tc.checkErr(b, err)
						} else if err != nil {
							b.Errorf("unexpected error: %v", err)
						}
					}
				})
			}
		})
	}
}
