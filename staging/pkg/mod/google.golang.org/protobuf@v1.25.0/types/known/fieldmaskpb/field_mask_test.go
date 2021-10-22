// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fieldmaskpb_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/protobuf/proto"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
	fmpb "google.golang.org/protobuf/types/known/fieldmaskpb"
)

func TestAppend(t *testing.T) {
	tests := []struct {
		inMessage proto.Message
		inPaths   []string
		wantPaths []string
		wantError error
	}{{
		inMessage: (*fmpb.FieldMask)(nil),
		inPaths:   []string{},
		wantPaths: []string{},
	}, {
		inMessage: (*fmpb.FieldMask)(nil),
		inPaths:   []string{"paths", "paths"},
		wantPaths: []string{"paths", "paths"},
	}, {
		inMessage: (*fmpb.FieldMask)(nil),
		inPaths:   []string{"paths", "<INVALID>", "paths"},
		wantPaths: []string{"paths"},
		wantError: cmpopts.AnyError,
	}, {
		inMessage: (*testpb.TestAllTypes)(nil),
		inPaths:   []string{"optional_int32", "OptionalGroup.optional_nested_message", "map_uint32_uint32", "map_string_nested_message.corecursive", "oneof_bool"},
		wantPaths: []string{"optional_int32", "OptionalGroup.optional_nested_message", "map_uint32_uint32", "map_string_nested_message.corecursive", "oneof_bool"},
	}, {
		inMessage: (*testpb.TestAllTypes)(nil),
		inPaths:   []string{"optional_nested_message", "optional_nested_message.corecursive", "optional_nested_message.corecursive.optional_nested_message", "optional_nested_message.corecursive.optional_nested_message.corecursive"},
		wantPaths: []string{"optional_nested_message", "optional_nested_message.corecursive", "optional_nested_message.corecursive.optional_nested_message", "optional_nested_message.corecursive.optional_nested_message.corecursive"},
	}, {
		inMessage: (*testpb.TestAllTypes)(nil),
		inPaths:   []string{"optional_int32", "optional_nested_message.corecursive.optional_int64", "optional_nested_message.corecursive.<INVALID>", "optional_int64"},
		wantPaths: []string{"optional_int32", "optional_nested_message.corecursive.optional_int64"},
		wantError: cmpopts.AnyError,
	}, {
		inMessage: (*testpb.TestAllTypes)(nil),
		inPaths:   []string{"optional_int32", "optional_nested_message.corecursive.oneof_uint32", "optional_nested_message.oneof_field", "optional_int64"},
		wantPaths: []string{"optional_int32", "optional_nested_message.corecursive.oneof_uint32"},
		wantError: cmpopts.AnyError,
	}}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var mask fmpb.FieldMask
			gotError := mask.Append(tt.inMessage, tt.inPaths...)
			gotPaths := mask.GetPaths()
			if diff := cmp.Diff(tt.wantPaths, gotPaths, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Append() paths mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantError, gotError, cmpopts.EquateErrors()); diff != "" {
				t.Errorf("Append() error mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestCombine(t *testing.T) {
	tests := []struct {
		in            [][]string
		wantUnion     []string
		wantIntersect []string
	}{{
		in: [][]string{
			{},
			{},
		},
		wantUnion:     []string{},
		wantIntersect: []string{},
	}, {
		in: [][]string{
			{"a"},
			{},
		},
		wantUnion:     []string{"a"},
		wantIntersect: []string{},
	}, {
		in: [][]string{
			{"a"},
			{"a"},
		},
		wantUnion:     []string{"a"},
		wantIntersect: []string{"a"},
	}, {
		in: [][]string{
			{"a"},
			{"b"},
			{"c"},
		},
		wantUnion:     []string{"a", "b", "c"},
		wantIntersect: []string{},
	}, {
		in: [][]string{
			{"a", "b"},
			{"b.b"},
			{"b"},
			{"b", "a.A"},
			{"b", "c", "c.a", "c.b"},
		},
		wantUnion:     []string{"a", "b", "c"},
		wantIntersect: []string{"b.b"},
	}, {
		in: [][]string{
			{"a.b", "a.c.d"},
			{"a"},
		},
		wantUnion:     []string{"a"},
		wantIntersect: []string{"a.b", "a.c.d"},
	}, {
		in: [][]string{
			{},
			{"a.b", "a.c", "d"},
		},
		wantUnion:     []string{"a.b", "a.c", "d"},
		wantIntersect: []string{},
	}}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var masks []*fmpb.FieldMask
			for _, paths := range tt.in {
				masks = append(masks, &fmpb.FieldMask{Paths: paths})
			}

			union := fmpb.Union(masks[0], masks[1], masks[2:]...)
			gotUnion := union.GetPaths()
			if diff := cmp.Diff(tt.wantUnion, gotUnion, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Union() mismatch (-want +got):\n%s", diff)
			}

			intersect := fmpb.Intersect(masks[0], masks[1], masks[2:]...)
			gotIntersect := intersect.GetPaths()
			if diff := cmp.Diff(tt.wantIntersect, gotIntersect, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Intersect() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNormalize(t *testing.T) {
	tests := []struct {
		in   []string
		want []string
	}{{
		in:   []string{},
		want: []string{},
	}, {
		in:   []string{"a"},
		want: []string{"a"},
	}, {
		in:   []string{"foo", "foo.bar", "foo.baz"},
		want: []string{"foo"},
	}, {
		in:   []string{"foo.bar", "foo.baz"},
		want: []string{"foo.bar", "foo.baz"},
	}, {
		in:   []string{"", "a.", ".b", "a.b", ".", "", "a.", ".b", "a.b", "."},
		want: []string{"", "a.", "a.b"},
	}, {
		in:   []string{"e.a", "e.b", "e.c", "e.d", "e.f", "e.g", "e.b.a", "e$c", "e.b.c"},
		want: []string{"e.a", "e.b", "e.c", "e.d", "e.f", "e.g", "e$c"},
	}, {
		in:   []string{"a", "aa", "aaa", "a$", "AAA", "aA.a", "a.a", "a", "aa", "aaa", "a$", "AAA", "aA.a"},
		want: []string{"AAA", "a", "aA.a", "aa", "aaa", "a$"},
	}, {
		in:   []string{"a.b", "aa.bb.cc", ".", "a$b", "aa", "a.", "a", "b.c.d", ".a", "", "a$", "a$", "a.b", "a", "a.bb", ""},
		want: []string{"", "a", "aa", "a$", "a$b", "b.c.d"},
	}}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			mask := &fmpb.FieldMask{
				Paths: append([]string(nil), tt.in...),
			}
			mask.Normalize()
			got := mask.GetPaths()
			if diff := cmp.Diff(tt.want, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("Normalize() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
