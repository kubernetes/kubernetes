/*
Copyright 2025 The Kubernetes Authors.

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

package validate

import (
	"context"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

type multiKeyItem struct {
	K1 string `json:"k1"`
	K2 string `json:"k2"`
	V  int    `json:"v"`
}

func TestSliceItem(t *testing.T) {
	testCases := []struct {
		name      string
		new       []multiKeyItem
		old       []multiKeyItem
		match     MatchItemFn[multiKeyItem]
		validator func(context.Context, operation.Operation, *field.Path, *multiKeyItem, *multiKeyItem) field.ErrorList
		expected  field.ErrorList
	}{
		{
			name: "no match",
			new: []multiKeyItem{
				{K1: "a", K2: "1", V: 1},
			},
			match: func(i *multiKeyItem) bool {
				return i.K1 == "target"
			},
			validator: func(_ context.Context, _ operation.Operation, fp *field.Path, _, _ *multiKeyItem) field.ErrorList {
				return field.ErrorList{field.Invalid(fp, nil, "err")}
			},
			expected: nil,
		},
		{
			name: "new item with matching keys",
			new: []multiKeyItem{
				{K1: "a", K2: "1", V: 1},
				{K1: "target", K2: "target2", V: 2},
			},
			match: func(i *multiKeyItem) bool {
				return i.K1 == "target" && i.K2 == "target2"
			},
			validator: func(_ context.Context, _ operation.Operation, fp *field.Path, n, o *multiKeyItem) field.ErrorList {
				if n != nil && o == nil {
					return field.ErrorList{field.Invalid(fp, n.K1, "added")}
				}
				return nil
			},
			expected: field.ErrorList{field.Invalid(field.NewPath("").Index(1), "target", "added")},
		},
		{
			name: "updated item - same keys different values",
			new: []multiKeyItem{
				{K1: "a", K2: "1", V: 1},
				{K1: "update", K2: "target2", V: 20},
			},
			old: []multiKeyItem{
				{K1: "a", K2: "1", V: 1},
				{K1: "update", K2: "target2", V: 2},
			},
			match: func(i *multiKeyItem) bool {
				return i.K1 == "update" && i.K2 == "target2"
			},
			validator: func(_ context.Context, _ operation.Operation, fp *field.Path, n, o *multiKeyItem) field.ErrorList {
				if n != nil && o != nil && n.V != o.V {
					return field.ErrorList{field.Invalid(fp.Child("v"), n.V, "changed")}
				}
				return nil
			},
			expected: field.ErrorList{field.Invalid(field.NewPath("").Index(1).Child("v"), 20, "changed")},
		},
		{
			// For completeness as listType=map && listKey=... required tags prevents dupes.
			name: "first match only - multiple items with same keys",
			new: []multiKeyItem{
				{K1: "dup", K2: "target2", V: 1},
				{K1: "dup", K2: "target2", V: 2},
			},
			old: []multiKeyItem{
				{K1: "dup", K2: "target2", V: 10},
			},
			match: func(i *multiKeyItem) bool {
				return i.K1 == "dup" && i.K2 == "target2"
			},
			validator: func(_ context.Context, _ operation.Operation, fp *field.Path, n, o *multiKeyItem) field.ErrorList {
				if n != nil && o != nil {
					return field.ErrorList{field.Invalid(fp, n.V, "value")}
				}
				return nil
			},
			expected: field.ErrorList{field.Invalid(field.NewPath("").Index(0), 1, "value")},
		},
		{
			name: "nil new list",
			new:  nil,
			old: []multiKeyItem{
				{K1: "exists", K2: "target2", V: 1},
			},
			match: func(i *multiKeyItem) bool {
				return i.K1 == "exists" && i.K2 == "target2"
			},
			validator: func(_ context.Context, _ operation.Operation, fp *field.Path, n, o *multiKeyItem) field.ErrorList {
				if n == nil && o != nil {
					return field.ErrorList{field.Invalid(fp, nil, "deleted")}
				}
				return nil
			},
			expected: nil,
		},
		{
			name:  "empty lists",
			new:   []multiKeyItem{},
			old:   []multiKeyItem{},
			match: func(i *multiKeyItem) bool { return true },
			validator: func(_ context.Context, _ operation.Operation, fp *field.Path, _, _ *multiKeyItem) field.ErrorList {
				return field.ErrorList{field.Invalid(fp, nil, "err")}
			},
			expected: nil,
		},
		{
			name:  "nil lists",
			new:   nil,
			old:   nil,
			match: func(i *multiKeyItem) bool { return true },
			validator: func(_ context.Context, _ operation.Operation, fp *field.Path, _, _ *multiKeyItem) field.ErrorList {
				return field.ErrorList{field.Invalid(fp, nil, "err")}
			},
			expected: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			op := operation.Operation{Type: operation.Update}
			fp := field.NewPath("")

			got := SliceItem(ctx, op, fp, tc.new, tc.old, tc.match, SemanticDeepEqual, tc.validator)

			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("got %v want %v", got, tc.expected)
			}
		})
	}
}
