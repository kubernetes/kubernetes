// Copyright 2017, OpenCensus Authors
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

package view

import (
	"context"
	"testing"

	"go.opencensus.io/tag"
)

func TestEncodeDecodeTags(t *testing.T) {
	ctx := context.Background()
	type testData struct {
		m    *tag.Map
		keys []tag.Key
		want map[tag.Key][]byte
	}

	k1 = tag.MustNewKey("/encodedecodetest/k1")
	k2 = tag.MustNewKey("/encodedecodetest/k2")
	k3 = tag.MustNewKey("/encodedecodetest/k3")

	ctx1, _ := tag.New(ctx)
	ctx2, _ := tag.New(ctx, tag.Insert(k2, "v2"))
	ctx3, _ := tag.New(ctx, tag.Insert(k1, "v1"), tag.Insert(k2, "v2"))
	ctx4, _ := tag.New(ctx, tag.Insert(k1, "v1"), tag.Insert(k2, "v2"), tag.Insert(k3, "v3"))

	m1 := tag.FromContext(ctx1)
	m2 := tag.FromContext(ctx2)
	m3 := tag.FromContext(ctx3)
	m4 := tag.FromContext(ctx4)

	tests := []testData{
		{
			m1,
			[]tag.Key{k1},
			nil,
		},
		{
			m2,
			[]tag.Key{},
			nil,
		},
		{
			m2,
			[]tag.Key{k1},
			nil,
		},
		{
			m2,
			[]tag.Key{k2},
			map[tag.Key][]byte{
				k2: []byte("v2"),
			},
		},
		{
			m3,
			[]tag.Key{k1},
			map[tag.Key][]byte{
				k1: []byte("v1"),
			},
		},
		{
			m3,
			[]tag.Key{k1, k2},
			map[tag.Key][]byte{
				k1: []byte("v1"),
				k2: []byte("v2"),
			},
		},
		{
			m4,
			[]tag.Key{k3, k1},
			map[tag.Key][]byte{
				k1: []byte("v1"),
				k3: []byte("v3"),
			},
		},
	}

	for label, tt := range tests {
		tags := decodeTags(encodeWithKeys(tt.m, tt.keys), tt.keys)
		if got, want := len(tags), len(tt.want); got != want {
			t.Fatalf("%d: len(decoded) = %v; not %v", label, got, want)
		}

		for _, tag := range tags {
			if _, ok := tt.want[tag.Key]; !ok {
				t.Errorf("%d: missing key %v", label, tag.Key)
			}
			if got, want := tag.Value, string(tt.want[tag.Key]); got != want {
				t.Errorf("%d: got value %q; want %q", label, got, want)
			}
		}
	}
}
