// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"reflect"
	"testing"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

func TestBoolDecisionValue_And(t *testing.T) {
	tests := []struct {
		name   string
		value  types.Bool
		ands   []ref.Val
		result ref.Val
	}{
		{
			name:   "init_false_end_false",
			value:  types.False,
			ands:   []ref.Val{types.NewErr("err"), types.True},
			result: types.False,
		},
		{
			name:   "init_true_end_false",
			value:  types.True,
			ands:   []ref.Val{types.NewErr("err"), types.False},
			result: types.False,
		},
		{
			name:   "init_true_end_err",
			value:  types.True,
			ands:   []ref.Val{types.True, types.NewErr("err")},
			result: types.NewErr("err"),
		},
		{
			name:   "init_true_end_unk",
			value:  types.True,
			ands:   []ref.Val{types.True, types.Unknown{1}, types.NewErr("err"), types.Unknown{2}},
			result: types.Unknown{1, 2},
		},
	}
	for _, tst := range tests {
		tc := tst
		t.Run(tc.name, func(tt *testing.T) {
			v := NewBoolDecisionValue(tc.name, tc.value)
			for _, av := range tc.ands {
				v = v.And(av)
			}
			v.Finalize(nil, nil)
			if !reflect.DeepEqual(v.Value(), tc.result) {
				tt.Errorf("decision AND failed. got %v, wanted %v", v.Value(), tc.result)
			}
		})
	}
}

func TestBoolDecisionValue_Or(t *testing.T) {
	tests := []struct {
		name   string
		value  types.Bool
		ors    []ref.Val
		result ref.Val
	}{
		{
			name:   "init_false_end_true",
			value:  types.False,
			ors:    []ref.Val{types.NewErr("err"), types.Unknown{1}, types.True},
			result: types.True,
		},
		{
			name:   "init_true_end_true",
			value:  types.True,
			ors:    []ref.Val{types.NewErr("err"), types.False},
			result: types.True,
		},
		{
			name:   "init_false_end_err",
			value:  types.False,
			ors:    []ref.Val{types.False, types.NewErr("err1"), types.NewErr("err2")},
			result: types.NewErr("err1"),
		},
		{
			name:   "init_false_end_unk",
			value:  types.False,
			ors:    []ref.Val{types.False, types.Unknown{1}, types.NewErr("err"), types.Unknown{2}},
			result: types.Unknown{1, 2},
		},
	}
	for _, tst := range tests {
		tc := tst
		t.Run(tc.name, func(tt *testing.T) {
			v := NewBoolDecisionValue(tc.name, tc.value)
			for _, av := range tc.ors {
				v = v.Or(av)
			}
			// Test finalization
			v.Finalize(nil, nil)
			// Ensure that calling string on the value doesn't error.
			_ = v.String()
			// Compare the output result
			if !reflect.DeepEqual(v.Value(), tc.result) {
				tt.Errorf("decision OR failed. got %v, wanted %v", v.Value(), tc.result)
			}
		})
	}
}
