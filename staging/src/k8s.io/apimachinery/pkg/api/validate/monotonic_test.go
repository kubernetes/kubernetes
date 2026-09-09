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
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate/constraints"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestMonotonic(t *testing.T) {
	testMonotonicPositive[int](t)
	testMonotonicNegative[int](t)
	testMonotonicPositive[int64](t)
	testMonotonicPositive[uint64](t)
}

type monotonicTestCase[T constraints.Integer] struct {
	name     string
	op       operation.Operation
	value    *T
	oldValue *T
	wantErrs field.ErrorList
}

func testMonotonicPositive[T constraints.Integer](t *testing.T) {
	t.Helper()
	v0 := T(0)
	v1 := T(1)
	v2 := T(2)

	cases := []monotonicTestCase[T]{{
		name:     "create (ignored)",
		op:       operation.Operation{Type: operation.Create},
		value:    &v0,
		oldValue: nil,
	}, {
		name:     "update same value",
		op:       operation.Operation{Type: operation.Update},
		value:    &v1,
		oldValue: &v1,
	}, {
		name:     "update increase",
		op:       operation.Operation{Type: operation.Update},
		value:    &v2,
		oldValue: &v1,
	}, {
		name:     "update decrease",
		op:       operation.Operation{Type: operation.Update},
		value:    &v1,
		oldValue: &v2,
		wantErrs: field.ErrorList{
			field.Invalid(field.NewPath("fldpath"), v1, "").WithOrigin("monotonic"),
		},
	}, {
		name:     "update nil value",
		op:       operation.Operation{Type: operation.Update},
		value:    nil,
		oldValue: &v1,
	}, {
		name:     "update nil old value",
		op:       operation.Operation{Type: operation.Update},
		value:    &v1,
		oldValue: nil,
	}}

	doTestMonotonic[T](t, cases)
}

func testMonotonicNegative[T constraints.Signed](t *testing.T) {
	t.Helper()
	vM1 := T(-1)
	vM2 := T(-2)

	cases := []monotonicTestCase[T]{{
		name:     "update negative increase",
		op:       operation.Operation{Type: operation.Update},
		value:    &vM1,
		oldValue: &vM2,
	}, {
		name:     "update negative decrease",
		op:       operation.Operation{Type: operation.Update},
		value:    &vM2,
		oldValue: &vM1,
		wantErrs: field.ErrorList{
			field.Invalid(field.NewPath("fldpath"), vM2, "").WithOrigin("monotonic"),
		},
	}}

	doTestMonotonic[T](t, cases)
}

func doTestMonotonic[T constraints.Integer](t *testing.T, cases []monotonicTestCase[T]) {
	t.Helper()
	matcher := field.ErrorMatcher{}.ByOrigin().ByDetailSubstring().ByField().ByType()
	for _, tc := range cases {
		name := fmt.Sprintf("%T %s", *new(T), tc.name)
		t.Run(name, func(t *testing.T) {
			gotErrs := Monotonic(context.Background(), tc.op, field.NewPath("fldpath"), tc.value, tc.oldValue)
			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}
