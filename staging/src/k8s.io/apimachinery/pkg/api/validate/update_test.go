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
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestUpdateValue(t *testing.T) {
	tests := []struct {
		name        string
		op          operation.Type
		value       string
		oldValue    string
		constraints []UpdateConstraint
		wantErrs    int
		wantMsgs    []string
	}{
		{
			name:        "create operation - no validation",
			op:          operation.Create,
			value:       "value",
			oldValue:    "",
			constraints: []UpdateConstraint{NoSet, NoUnset, NoModify},
			wantErrs:    0,
		},
		{
			name:        "NoSet - unset to set transition (forbidden)",
			op:          operation.Update,
			value:       "value",
			oldValue:    "",
			constraints: []UpdateConstraint{NoSet},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be set once created"},
		},
		{
			name:        "NoSet - set to set transition (allowed)",
			op:          operation.Update,
			value:       "value2",
			oldValue:    "value1",
			constraints: []UpdateConstraint{NoSet},
			wantErrs:    0,
		},
		{
			name:        "NoUnset - set to unset transition (forbidden)",
			op:          operation.Update,
			value:       "",
			oldValue:    "value",
			constraints: []UpdateConstraint{NoUnset},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be cleared once set"},
		},
		{
			name:        "NoUnset - unset to set transition (allowed)",
			op:          operation.Update,
			value:       "value",
			oldValue:    "",
			constraints: []UpdateConstraint{NoUnset},
			wantErrs:    0,
		},
		{
			name:        "NoModify - set to different value (forbidden)",
			op:          operation.Update,
			value:       "value2",
			oldValue:    "value1",
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be modified once set"},
		},
		{
			name:        "NoModify - unset to set transition (allowed)",
			op:          operation.Update,
			value:       "value",
			oldValue:    "",
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    0,
		},
		{
			name:        "NoModify - set to unset transition (allowed)",
			op:          operation.Update,
			value:       "",
			oldValue:    "value",
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    0,
		},
		{
			name:        "Multiple constraints - NoSet and NoUnset",
			op:          operation.Update,
			value:       "value",
			oldValue:    "",
			constraints: []UpdateConstraint{NoSet, NoUnset},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be set once created"},
		},
		{
			name:        "Multiple constraints - NoUnset and NoModify",
			op:          operation.Update,
			value:       "",
			oldValue:    "value",
			constraints: []UpdateConstraint{NoUnset, NoModify},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be cleared once set"},
		},
		{
			name:        "Multiple constraints - NoSet, NoUnset, NoModify - modify attempt",
			op:          operation.Update,
			value:       "value2",
			oldValue:    "value1",
			constraints: []UpdateConstraint{NoSet, NoUnset, NoModify},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be modified once set"},
		},
		{
			name:        "No constraints",
			op:          operation.Update,
			value:       "value2",
			oldValue:    "value1",
			constraints: []UpdateConstraint{},
			wantErrs:    0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := UpdateValueByCompare(context.TODO(), op, field.NewPath("test"), &tt.value, &tt.oldValue, tt.constraints...)
			if len(errs) != tt.wantErrs {
				t.Errorf("UpdateValue() returned %d errors, want %d: %v", len(errs), tt.wantErrs, errs)
			}
			for i, msg := range tt.wantMsgs {
				if i >= len(errs) {
					t.Errorf("Expected error message %q not found", msg)
					continue
				}
				if errs[i].Detail != msg {
					t.Errorf("UpdateValue() error message = %q, want %q", errs[i].Detail, msg)
				}
			}
		})
	}
}

func TestUpdatePointer(t *testing.T) {
	stringPtr := func(s string) *string { return &s }

	tests := []struct {
		name        string
		op          operation.Type
		value       *string
		oldValue    *string
		constraints []UpdateConstraint
		wantErrs    int
		wantMsgs    []string
	}{
		{
			name:        "create operation - no validation",
			op:          operation.Create,
			value:       stringPtr("value"),
			oldValue:    nil,
			constraints: []UpdateConstraint{NoSet, NoUnset, NoModify},
			wantErrs:    0,
		},
		{
			name:        "NoSet - nil to non-nil transition (forbidden)",
			op:          operation.Update,
			value:       stringPtr("value"),
			oldValue:    nil,
			constraints: []UpdateConstraint{NoSet},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be set once created"},
		},
		{
			name:        "NoSet - non-nil to non-nil transition (allowed)",
			op:          operation.Update,
			value:       stringPtr("value2"),
			oldValue:    stringPtr("value1"),
			constraints: []UpdateConstraint{NoSet},
			wantErrs:    0,
		},
		{
			name:        "NoUnset - non-nil to nil transition (forbidden)",
			op:          operation.Update,
			value:       nil,
			oldValue:    stringPtr("value"),
			constraints: []UpdateConstraint{NoUnset},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be cleared once set"},
		},
		{
			name:        "NoUnset - nil to non-nil transition (allowed)",
			op:          operation.Update,
			value:       stringPtr("value"),
			oldValue:    nil,
			constraints: []UpdateConstraint{NoUnset},
			wantErrs:    0,
		},
		{
			name:        "NoModify - different values (forbidden)",
			op:          operation.Update,
			value:       stringPtr("value2"),
			oldValue:    stringPtr("value1"),
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be modified once set"},
		},
		{
			name:        "NoModify - nil to non-nil transition (allowed)",
			op:          operation.Update,
			value:       stringPtr("value"),
			oldValue:    nil,
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    0,
		},
		{
			name:        "NoModify - non-nil to nil transition (allowed)",
			op:          operation.Update,
			value:       nil,
			oldValue:    stringPtr("value"),
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    0,
		},
		{
			name:        "Multiple constraints - all three",
			op:          operation.Update,
			value:       stringPtr("value2"),
			oldValue:    stringPtr("value1"),
			constraints: []UpdateConstraint{NoSet, NoUnset, NoModify},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be modified once set"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := UpdatePointer(context.TODO(), op, field.NewPath("test"), tt.value, tt.oldValue, tt.constraints...)
			if len(errs) != tt.wantErrs {
				t.Errorf("UpdatePointer() returned %d errors, want %d: %v", len(errs), tt.wantErrs, errs)
			}
			for i, msg := range tt.wantMsgs {
				if i >= len(errs) {
					t.Errorf("Expected error message %q not found", msg)
					continue
				}
				if errs[i].Detail != msg {
					t.Errorf("UpdatePointer() error message = %q, want %q", errs[i].Detail, msg)
				}
			}
		})
	}
}

func TestUpdateValueByReflect(t *testing.T) {
	type CustomStruct struct {
		Field1 string
		Field2 int
	}

	tests := []struct {
		name        string
		op          operation.Type
		value       CustomStruct
		oldValue    CustomStruct
		constraints []UpdateConstraint
		wantErrs    int
		wantMsgs    []string
	}{
		{
			name:        "NoModify - zero to non-zero transition (allowed)",
			op:          operation.Update,
			value:       CustomStruct{Field1: "test", Field2: 42},
			oldValue:    CustomStruct{},
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    0,
		},
		{
			name:        "NoModify - non-zero to zero transition (allowed)",
			op:          operation.Update,
			value:       CustomStruct{},
			oldValue:    CustomStruct{Field1: "test", Field2: 42},
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    0,
		},
		{
			name:        "NoModify - different values (forbidden)",
			op:          operation.Update,
			value:       CustomStruct{Field1: "test2", Field2: 100},
			oldValue:    CustomStruct{Field1: "test1", Field2: 42},
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be modified once set"},
		},
		{
			name:        "NoSet - zero to non-zero (forbidden)",
			op:          operation.Update,
			value:       CustomStruct{Field1: "test", Field2: 42},
			oldValue:    CustomStruct{},
			constraints: []UpdateConstraint{NoSet},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be set once created"},
		},
		{
			name:        "NoUnset - non-zero to zero (forbidden)",
			op:          operation.Update,
			value:       CustomStruct{},
			oldValue:    CustomStruct{Field1: "test", Field2: 42},
			constraints: []UpdateConstraint{NoUnset},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be cleared once set"},
		},
		{
			name:        "Multiple constraints",
			op:          operation.Update,
			value:       CustomStruct{Field1: "test", Field2: 42},
			oldValue:    CustomStruct{},
			constraints: []UpdateConstraint{NoSet, NoModify},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be set once created"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := UpdateValueByReflect(context.TODO(), op, field.NewPath("test"), &tt.value, &tt.oldValue, tt.constraints...)
			if len(errs) != tt.wantErrs {
				t.Errorf("UpdateValueByReflect() returned %d errors, want %d: %v", len(errs), tt.wantErrs, errs)
			}
			for i, msg := range tt.wantMsgs {
				if i >= len(errs) {
					t.Errorf("Expected error message %q not found", msg)
					continue
				}
				if errs[i].Detail != msg {
					t.Errorf("UpdateValueByReflect() error message = %q, want %q", errs[i].Detail, msg)
				}
			}
		})
	}
}

func TestUpdateStruct(t *testing.T) {
	type TestStruct struct {
		Field1 string
		Field2 int
	}

	tests := []struct {
		name        string
		op          operation.Type
		value       TestStruct
		oldValue    TestStruct
		constraints []UpdateConstraint
		wantErrs    int
		wantMsgs    []string
	}{
		{
			name:        "create operation - no validation",
			op:          operation.Create,
			value:       TestStruct{Field1: "test", Field2: 42},
			oldValue:    TestStruct{},
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    0,
		},
		{
			name:        "NoModify - different values (forbidden)",
			op:          operation.Update,
			value:       TestStruct{Field1: "test2", Field2: 100},
			oldValue:    TestStruct{Field1: "test1", Field2: 42},
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be modified once set"},
		},
		{
			name:        "NoModify - zero value to non-zero (forbidden)",
			op:          operation.Update,
			value:       TestStruct{Field1: "test", Field2: 42},
			oldValue:    TestStruct{},
			constraints: []UpdateConstraint{NoModify},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be modified once set"},
		},
		{
			name:        "NoSet and NoUnset with modification - only NoModify triggers",
			op:          operation.Update,
			value:       TestStruct{Field1: "test2", Field2: 100},
			oldValue:    TestStruct{Field1: "test1", Field2: 42},
			constraints: []UpdateConstraint{NoSet, NoUnset, NoModify},
			wantErrs:    1,
			wantMsgs:    []string{"field cannot be modified once set"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := UpdateStruct(context.TODO(), op, field.NewPath("test"), &tt.value, &tt.oldValue, tt.constraints...)
			if len(errs) != tt.wantErrs {
				t.Errorf("UpdateStruct() returned %d errors, want %d: %v", len(errs), tt.wantErrs, errs)
			}
			for i, msg := range tt.wantMsgs {
				if i >= len(errs) {
					t.Errorf("Expected error message %q not found", msg)
					continue
				}
				if errs[i].Detail != msg {
					t.Errorf("UpdateStruct() error message = %q, want %q", errs[i].Detail, msg)
				}
			}
		})
	}
}
