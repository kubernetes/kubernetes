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
//

package tag

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

var (
	ttlUnlimitedPropMd = createMetadatas(WithTTL(TTLUnlimitedPropagation))
	ttlNoPropMd        = createMetadatas(WithTTL(TTLNoPropagation))
)

func TestContext(t *testing.T) {
	k1, _ := NewKey("k1")
	k2, _ := NewKey("k2")

	ctx := context.Background()
	ctx, _ = New(ctx,
		Insert(k1, "v1"),
		Insert(k2, "v2"),
	)
	got := FromContext(ctx)
	want := newMap()
	want.insert(k1, "v1", ttlUnlimitedPropMd)
	want.insert(k2, "v2", ttlUnlimitedPropMd)

	if !reflect.DeepEqual(got, want) {
		t.Errorf("Map = %#v; want %#v", got, want)
	}
}

func TestDo(t *testing.T) {
	k1, _ := NewKey("k1")
	k2, _ := NewKey("k2")
	ctx := context.Background()
	ctx, _ = New(ctx,
		Insert(k1, "v1"),
		Insert(k2, "v2"),
	)
	got := FromContext(ctx)
	want := newMap()
	want.insert(k1, "v1", ttlUnlimitedPropMd)
	want.insert(k2, "v2", ttlUnlimitedPropMd)
	Do(ctx, func(ctx context.Context) {
		got = FromContext(ctx)
	})
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Map = %#v; want %#v", got, want)
	}
}

func TestNewMap(t *testing.T) {
	k1, _ := NewKey("k1")
	k2, _ := NewKey("k2")
	k3, _ := NewKey("k3")
	k4, _ := NewKey("k4")
	k5, _ := NewKey("k5")

	initial := makeTestTagMap(5)

	tests := []struct {
		name    string
		initial *Map
		mods    []Mutator
		want    *Map
	}{
		{
			name:    "from empty; insert",
			initial: nil,
			mods: []Mutator{
				Insert(k5, "v5"),
			},
			want: makeTestTagMap(2, 4, 5),
		},
		{
			name:    "from empty; insert existing",
			initial: nil,
			mods: []Mutator{
				Insert(k1, "v1"),
			},
			want: makeTestTagMap(1, 2, 4),
		},
		{
			name:    "from empty; update",
			initial: nil,
			mods: []Mutator{
				Update(k1, "v1"),
			},
			want: makeTestTagMap(2, 4),
		},
		{
			name:    "from empty; update unexisting",
			initial: nil,
			mods: []Mutator{
				Update(k5, "v5"),
			},
			want: makeTestTagMap(2, 4),
		},
		{
			name:    "from existing; upsert",
			initial: initial,
			mods: []Mutator{
				Upsert(k5, "v5"),
			},
			want: makeTestTagMap(2, 4, 5),
		},
		{
			name:    "from existing; delete",
			initial: initial,
			mods: []Mutator{
				Delete(k2),
			},
			want: makeTestTagMap(4, 5),
		},
		{
			name:    "from empty; invalid",
			initial: nil,
			mods: []Mutator{
				Insert(k5, "v\x19"),
				Upsert(k5, "v\x19"),
				Update(k5, "v\x19"),
			},
			want: nil,
		},
		{
			name:    "from empty; no partial",
			initial: nil,
			mods: []Mutator{
				Insert(k5, "v1"),
				Update(k5, "v\x19"),
			},
			want: nil,
		},
	}

	for _, tt := range tests {
		mods := []Mutator{
			Insert(k1, "v1"),
			Insert(k2, "v2"),
			Update(k3, "v3"),
			Upsert(k4, "v4"),
			Insert(k2, "v2"),
			Delete(k1),
		}
		mods = append(mods, tt.mods...)
		ctx := NewContext(context.Background(), tt.initial)
		ctx, err := New(ctx, mods...)
		if tt.want != nil && err != nil {
			t.Errorf("%v: New = %v", tt.name, err)
		}

		if got, want := FromContext(ctx), tt.want; !reflect.DeepEqual(got, want) {
			t.Errorf("%v: got %v; want %v", tt.name, got, want)
		}
	}
}

func TestNewMapWithMetadata(t *testing.T) {
	k3, _ := NewKey("k3")
	k4, _ := NewKey("k4")
	k5, _ := NewKey("k5")

	tests := []struct {
		name    string
		initial *Map
		mods    []Mutator
		want    *Map
	}{
		{
			name:    "from empty; insert",
			initial: nil,
			mods: []Mutator{
				Insert(k5, "5", WithTTL(TTLNoPropagation)),
				Insert(k4, "4"),
			},
			want: makeTestTagMapWithMetadata(
				tagContent{"5", ttlNoPropMd},
				tagContent{"4", ttlUnlimitedPropMd}),
		},
		{
			name:    "from existing; insert existing",
			initial: makeTestTagMapWithMetadata(tagContent{"5", ttlNoPropMd}),
			mods: []Mutator{
				Insert(k5, "5", WithTTL(TTLUnlimitedPropagation)),
			},
			want: makeTestTagMapWithMetadata(tagContent{"5", ttlNoPropMd}),
		},
		{
			name:    "from existing; update non-existing",
			initial: makeTestTagMapWithMetadata(tagContent{"5", ttlNoPropMd}),
			mods: []Mutator{
				Update(k4, "4", WithTTL(TTLUnlimitedPropagation)),
			},
			want: makeTestTagMapWithMetadata(tagContent{"5", ttlNoPropMd}),
		},
		{
			name: "from existing; update existing",
			initial: makeTestTagMapWithMetadata(
				tagContent{"5", ttlUnlimitedPropMd},
				tagContent{"4", ttlNoPropMd}),
			mods: []Mutator{
				Update(k5, "5"),
				Update(k4, "4", WithTTL(TTLUnlimitedPropagation)),
			},
			want: makeTestTagMapWithMetadata(
				tagContent{"5", ttlUnlimitedPropMd},
				tagContent{"4", ttlUnlimitedPropMd}),
		},
		{
			name: "from existing; upsert existing",
			initial: makeTestTagMapWithMetadata(
				tagContent{"5", ttlNoPropMd},
				tagContent{"4", ttlNoPropMd}),
			mods: []Mutator{
				Upsert(k4, "4", WithTTL(TTLUnlimitedPropagation)),
			},
			want: makeTestTagMapWithMetadata(
				tagContent{"5", ttlNoPropMd},
				tagContent{"4", ttlUnlimitedPropMd}),
		},
		{
			name: "from existing; upsert non-existing",
			initial: makeTestTagMapWithMetadata(
				tagContent{"5", ttlNoPropMd}),
			mods: []Mutator{
				Upsert(k4, "4", WithTTL(TTLUnlimitedPropagation)),
				Upsert(k3, "3"),
			},
			want: makeTestTagMapWithMetadata(
				tagContent{"5", ttlNoPropMd},
				tagContent{"4", ttlUnlimitedPropMd},
				tagContent{"3", ttlUnlimitedPropMd}),
		},
		{
			name: "from existing; delete",
			initial: makeTestTagMapWithMetadata(
				tagContent{"5", ttlNoPropMd},
				tagContent{"4", ttlNoPropMd}),
			mods: []Mutator{
				Delete(k5),
			},
			want: makeTestTagMapWithMetadata(
				tagContent{"4", ttlNoPropMd}),
		},
		{
			name:    "from non-existing; upsert with multiple-metadata",
			initial: nil,
			mods: []Mutator{
				Upsert(k4, "4", WithTTL(TTLUnlimitedPropagation), WithTTL(TTLNoPropagation)),
				Upsert(k5, "5", WithTTL(TTLNoPropagation), WithTTL(TTLUnlimitedPropagation)),
			},
			want: makeTestTagMapWithMetadata(
				tagContent{"4", ttlNoPropMd},
				tagContent{"5", ttlUnlimitedPropMd}),
		},
		{
			name:    "from non-existing; insert with multiple-metadata",
			initial: nil,
			mods: []Mutator{
				Insert(k5, "5", WithTTL(TTLNoPropagation), WithTTL(TTLUnlimitedPropagation)),
			},
			want: makeTestTagMapWithMetadata(
				tagContent{"5", ttlUnlimitedPropMd}),
		},
		{
			name: "from existing; update with multiple-metadata",
			initial: makeTestTagMapWithMetadata(
				tagContent{"5", ttlNoPropMd}),
			mods: []Mutator{
				Update(k5, "5", WithTTL(TTLNoPropagation), WithTTL(TTLUnlimitedPropagation)),
			},
			want: makeTestTagMapWithMetadata(
				tagContent{"5", ttlUnlimitedPropMd}),
		},
		{
			name:    "from empty; update invalid",
			initial: nil,
			mods: []Mutator{
				Insert(k4, "4\x19", WithTTL(TTLUnlimitedPropagation)),
				Upsert(k4, "4\x19", WithTTL(TTLUnlimitedPropagation)),
				Update(k4, "4\x19", WithTTL(TTLUnlimitedPropagation)),
			},
			want: nil,
		},
		{
			name:    "from empty; insert partial",
			initial: nil,
			mods: []Mutator{
				Upsert(k3, "3", WithTTL(TTLUnlimitedPropagation)),
				Upsert(k4, "4\x19", WithTTL(TTLUnlimitedPropagation)),
			},
			want: nil,
		},
	}

	// Test api for insert, update, and upsert using metadata.
	for _, tt := range tests {
		ctx := NewContext(context.Background(), tt.initial)
		ctx, err := New(ctx, tt.mods...)
		if tt.want != nil && err != nil {
			t.Errorf("%v: New = %v", tt.name, err)
		}

		if got, want := FromContext(ctx), tt.want; !reflect.DeepEqual(got, want) {
			t.Errorf("%v: got %v; want %v", tt.name, got, want)
		}
	}
}

func TestNewValidation(t *testing.T) {
	tests := []struct {
		err  string
		seed *Map
	}{
		// Key name validation in seed
		{err: "invalid key", seed: &Map{m: map[Key]tagContent{{name: ""}: {"foo", ttlNoPropMd}}}},
		{err: "", seed: &Map{m: map[Key]tagContent{{name: "key"}: {"foo", ttlNoPropMd}}}},
		{err: "", seed: &Map{m: map[Key]tagContent{{name: strings.Repeat("a", 255)}: {"census", ttlNoPropMd}}}},
		{err: "invalid key", seed: &Map{m: map[Key]tagContent{{name: strings.Repeat("a", 256)}: {"census", ttlNoPropMd}}}},
		{err: "invalid key", seed: &Map{m: map[Key]tagContent{{name: "Приве́т"}: {"census", ttlNoPropMd}}}},

		// Value validation
		{err: "", seed: &Map{m: map[Key]tagContent{{name: "key"}: {"", ttlNoPropMd}}}},
		{err: "", seed: &Map{m: map[Key]tagContent{{name: "key"}: {strings.Repeat("a", 255), ttlNoPropMd}}}},
		{err: "invalid value", seed: &Map{m: map[Key]tagContent{{name: "key"}: {"Приве́т", ttlNoPropMd}}}},
		{err: "invalid value", seed: &Map{m: map[Key]tagContent{{name: "key"}: {strings.Repeat("a", 256), ttlNoPropMd}}}},
	}

	for i, tt := range tests {
		ctx := NewContext(context.Background(), tt.seed)
		ctx, err := New(ctx)

		if tt.err != "" {
			if err == nil {
				t.Errorf("#%d: got nil error; want %q", i, tt.err)
				continue
			} else if s, substr := err.Error(), tt.err; !strings.Contains(s, substr) {
				t.Errorf("#%d:\ngot %q\nwant %q", i, s, substr)
			}
			continue
		}
		if err != nil {
			t.Errorf("#%d: got %q want nil", i, err)
			continue
		}
		m := FromContext(ctx)
		if m == nil {
			t.Errorf("#%d: got nil map", i)
			continue
		}
	}
}

func makeTestTagMap(ids ...int) *Map {
	m := newMap()
	for _, v := range ids {
		k, _ := NewKey(fmt.Sprintf("k%d", v))
		m.m[k] = tagContent{fmt.Sprintf("v%d", v), ttlUnlimitedPropMd}
	}
	return m
}

func makeTestTagMapWithMetadata(tcs ...tagContent) *Map {
	m := newMap()
	for _, tc := range tcs {
		k, _ := NewKey(fmt.Sprintf("k%s", tc.value))
		m.m[k] = tc
	}
	return m
}
