/*
Copyright 2024 The Kubernetes Authors.

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
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestIfOption(t *testing.T) {
	errs := field.ErrorList{field.Invalid(field.NewPath("field"), "value", "detail")}
	opWithOption := operation.Operation{Options: []string{"the-option"}}
	opWithoutOption := operation.Operation{}

	testCases := []struct {
		name        string
		op          operation.Operation
		optionName  string
		enabled     bool
		shouldBeNil bool
		expectCall  bool
	}{
		{
			name:        "enabled: true, option present: true -> should call wrapped",
			op:          opWithOption,
			optionName:  "the-option",
			enabled:     true,
			expectCall:  true,
			shouldBeNil: false,
		},
		{
			name:        "enabled: true, option present: false -> should not call wrapped",
			op:          opWithoutOption,
			optionName:  "the-option",
			enabled:     true,
			expectCall:  false,
			shouldBeNil: true,
		},
		{
			name:        "enabled: false, option present: true -> should not call wrapped",
			op:          opWithOption,
			optionName:  "the-option",
			enabled:     false,
			expectCall:  false,
			shouldBeNil: true,
		},
		{
			name:        "enabled: false, option present: false -> should call wrapped",
			op:          opWithoutOption,
			optionName:  "the-option",
			enabled:     false,
			expectCall:  true,
			shouldBeNil: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			called := false
			wrapped := func(ctx context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *string) field.ErrorList {
				called = true
				return errs
			}

			result := IfOption(context.TODO(), tc.op, nil, nil, nil, tc.optionName, tc.enabled, wrapped)

			if tc.shouldBeNil {
				if result != nil {
					t.Errorf("expected nil but got %v", result)
				}
			} else {
				if result == nil {
					t.Errorf("expected non-nil but got nil")
				}
			}

			if tc.expectCall != called {
				t.Errorf("wrapped function call expectation failed: expected %v, got %v", tc.expectCall, called)
			}
		})
	}
}
