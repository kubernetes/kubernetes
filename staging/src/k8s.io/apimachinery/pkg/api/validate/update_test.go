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

func TestNoSetValue(t *testing.T) {
	tests := []struct {
		name     string
		op       operation.Type
		value    string
		oldValue string
		wantErr  bool
	}{
		{
			name:     "create operation - no validation",
			op:       operation.Create,
			value:    "value",
			oldValue: "",
			wantErr:  false,
		},
		{
			name:     "update - unset to set transition (forbidden)",
			op:       operation.Update,
			value:    "value",
			oldValue: "",
			wantErr:  true,
		},
		{
			name:     "update - set to set transition (allowed)",
			op:       operation.Update,
			value:    "value2",
			oldValue: "value1",
			wantErr:  false,
		},
		{
			name:     "update - unset to unset (allowed)",
			op:       operation.Update,
			value:    "",
			oldValue: "",
			wantErr:  false,
		},
		{
			name:     "update - set to unset transition (allowed)",
			op:       operation.Update,
			value:    "",
			oldValue: "value",
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := NoSetValue(context.TODO(), op, field.NewPath("test"), &tt.value, &tt.oldValue)
			if (len(errs) > 0) != tt.wantErr {
				t.Errorf("NoSetValue() error = %v, wantErr %v", errs, tt.wantErr)
			}
			if tt.wantErr && len(errs) > 0 {
				if errs[0].Detail != "field cannot be set once created" {
					t.Errorf("NoSetValue() wrong error message: %v", errs[0].Detail)
				}
			}
		})
	}
}

func TestNoSetPointer(t *testing.T) {
	stringPtr := func(s string) *string { return &s }

	tests := []struct {
		name     string
		op       operation.Type
		value    *string
		oldValue *string
		wantErr  bool
	}{
		{
			name:     "create operation - no validation",
			op:       operation.Create,
			value:    stringPtr("value"),
			oldValue: nil,
			wantErr:  false,
		},
		{
			name:     "update - nil to non-nil transition (forbidden)",
			op:       operation.Update,
			value:    stringPtr("value"),
			oldValue: nil,
			wantErr:  true,
		},
		{
			name:     "update - non-nil to non-nil transition (allowed)",
			op:       operation.Update,
			value:    stringPtr("value2"),
			oldValue: stringPtr("value1"),
			wantErr:  false,
		},
		{
			name:     "update - nil to nil (allowed)",
			op:       operation.Update,
			value:    nil,
			oldValue: nil,
			wantErr:  false,
		},
		{
			name:     "update - non-nil to nil transition (allowed)",
			op:       operation.Update,
			value:    nil,
			oldValue: stringPtr("value"),
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := NoSetPointer(context.TODO(), op, field.NewPath("test"), tt.value, tt.oldValue)
			if (len(errs) > 0) != tt.wantErr {
				t.Errorf("NoSetPointer() error = %v, wantErr %v", errs, tt.wantErr)
			}
			if tt.wantErr && len(errs) > 0 {
				if errs[0].Detail != "field cannot be set once created" {
					t.Errorf("NoSetPointer() wrong error message: %v", errs[0].Detail)
				}
			}
		})
	}
}

func TestNoUnsetValue(t *testing.T) {
	tests := []struct {
		name     string
		op       operation.Type
		value    string
		oldValue string
		wantErr  bool
	}{
		{
			name:     "create operation - no validation",
			op:       operation.Create,
			value:    "",
			oldValue: "value",
			wantErr:  false,
		},
		{
			name:     "update - set to unset transition (forbidden)",
			op:       operation.Update,
			value:    "",
			oldValue: "value",
			wantErr:  true,
		},
		{
			name:     "update - unset to set transition (allowed)",
			op:       operation.Update,
			value:    "value",
			oldValue: "",
			wantErr:  false,
		},
		{
			name:     "update - set to set transition (allowed)",
			op:       operation.Update,
			value:    "value2",
			oldValue: "value1",
			wantErr:  false,
		},
		{
			name:     "update - unset to unset (allowed)",
			op:       operation.Update,
			value:    "",
			oldValue: "",
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := NoUnsetValue(context.TODO(), op, field.NewPath("test"), &tt.value, &tt.oldValue)
			if (len(errs) > 0) != tt.wantErr {
				t.Errorf("NoUnsetValue() error = %v, wantErr %v", errs, tt.wantErr)
			}
			if tt.wantErr && len(errs) > 0 {
				if errs[0].Detail != "field cannot be cleared once set" {
					t.Errorf("NoUnsetValue() wrong error message: %v", errs[0].Detail)
				}
			}
		})
	}
}

func TestNoUnsetPointer(t *testing.T) {
	stringPtr := func(s string) *string { return &s }

	tests := []struct {
		name     string
		op       operation.Type
		value    *string
		oldValue *string
		wantErr  bool
	}{
		{
			name:     "create operation - no validation",
			op:       operation.Create,
			value:    nil,
			oldValue: stringPtr("value"),
			wantErr:  false,
		},
		{
			name:     "update - non-nil to nil transition (forbidden)",
			op:       operation.Update,
			value:    nil,
			oldValue: stringPtr("value"),
			wantErr:  true,
		},
		{
			name:     "update - nil to non-nil transition (allowed)",
			op:       operation.Update,
			value:    stringPtr("value"),
			oldValue: nil,
			wantErr:  false,
		},
		{
			name:     "update - non-nil to non-nil transition (allowed)",
			op:       operation.Update,
			value:    stringPtr("value2"),
			oldValue: stringPtr("value1"),
			wantErr:  false,
		},
		{
			name:     "update - nil to nil (allowed)",
			op:       operation.Update,
			value:    nil,
			oldValue: nil,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := NoUnsetPointer(context.TODO(), op, field.NewPath("test"), tt.value, tt.oldValue)
			if (len(errs) > 0) != tt.wantErr {
				t.Errorf("NoUnsetPointer() error = %v, wantErr %v", errs, tt.wantErr)
			}
			if tt.wantErr && len(errs) > 0 {
				if errs[0].Detail != "field cannot be cleared once set" {
					t.Errorf("NoUnsetPointer() wrong error message: %v", errs[0].Detail)
				}
			}
		})
	}
}

func TestNoModifyValue(t *testing.T) {
	tests := []struct {
		name     string
		op       operation.Type
		value    string
		oldValue string
		wantErr  bool
	}{
		{
			name:     "create operation - no validation",
			op:       operation.Create,
			value:    "value",
			oldValue: "",
			wantErr:  false,
		},
		{
			name:     "update - unset to set transition (allowed)",
			op:       operation.Update,
			value:    "value",
			oldValue: "",
			wantErr:  false,
		},
		{
			name:     "update - set to unset transition (allowed)",
			op:       operation.Update,
			value:    "",
			oldValue: "value",
			wantErr:  false,
		},
		{
			name:     "update - set to different value (forbidden)",
			op:       operation.Update,
			value:    "value2",
			oldValue: "value1",
			wantErr:  true,
		},
		{
			name:     "update - same value (allowed)",
			op:       operation.Update,
			value:    "value",
			oldValue: "value",
			wantErr:  false,
		},
		{
			name:     "update - both unset (allowed)",
			op:       operation.Update,
			value:    "",
			oldValue: "",
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := NoModifyValue(context.TODO(), op, field.NewPath("test"), &tt.value, &tt.oldValue)
			if (len(errs) > 0) != tt.wantErr {
				t.Errorf("NoModifyValue() error = %v, wantErr %v", errs, tt.wantErr)
			}
			if tt.wantErr && len(errs) > 0 {
				if errs[0].Detail != "field cannot be modified once set" {
					t.Errorf("NoModifyValue() wrong error message: %v", errs[0].Detail)
				}
			}
		})
	}
}

func TestNoModifyValueWithInts(t *testing.T) {
	tests := []struct {
		name     string
		op       operation.Type
		value    int
		oldValue int
		wantErr  bool
	}{
		{
			name:     "update - zero to non-zero transition (allowed)",
			op:       operation.Update,
			value:    42,
			oldValue: 0,
			wantErr:  false,
		},
		{
			name:     "update - non-zero to zero transition (allowed)",
			op:       operation.Update,
			value:    0,
			oldValue: 42,
			wantErr:  false,
		},
		{
			name:     "update - non-zero to different non-zero (forbidden)",
			op:       operation.Update,
			value:    100,
			oldValue: 42,
			wantErr:  true,
		},
		{
			name:     "update - same non-zero value (allowed)",
			op:       operation.Update,
			value:    42,
			oldValue: 42,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := NoModifyValue(context.TODO(), op, field.NewPath("test"), &tt.value, &tt.oldValue)
			if (len(errs) > 0) != tt.wantErr {
				t.Errorf("NoModifyValue() error = %v, wantErr %v", errs, tt.wantErr)
			}
		})
	}
}

func TestNoModifyValueByReflect(t *testing.T) {
	type CustomStruct struct {
		Field1 string
		Field2 int
	}

	tests := []struct {
		name     string
		op       operation.Type
		value    CustomStruct
		oldValue CustomStruct
		wantErr  bool
	}{
		{
			name:     "update - zero to non-zero transition (allowed)",
			op:       operation.Update,
			value:    CustomStruct{Field1: "test", Field2: 42},
			oldValue: CustomStruct{},
			wantErr:  false,
		},
		{
			name:     "update - non-zero to zero transition (allowed)",
			op:       operation.Update,
			value:    CustomStruct{},
			oldValue: CustomStruct{Field1: "test", Field2: 42},
			wantErr:  false,
		},
		{
			name:     "update - different values (forbidden)",
			op:       operation.Update,
			value:    CustomStruct{Field1: "test2", Field2: 100},
			oldValue: CustomStruct{Field1: "test1", Field2: 42},
			wantErr:  true,
		},
		{
			name:     "update - same values (allowed)",
			op:       operation.Update,
			value:    CustomStruct{Field1: "test", Field2: 42},
			oldValue: CustomStruct{Field1: "test", Field2: 42},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := NoModifyValueByReflect(context.TODO(), op, field.NewPath("test"), &tt.value, &tt.oldValue)
			if (len(errs) > 0) != tt.wantErr {
				t.Errorf("NoModifyValueByReflect() error = %v, wantErr %v", errs, tt.wantErr)
			}
			if tt.wantErr && len(errs) > 0 {
				if errs[0].Detail != "field cannot be modified once set" {
					t.Errorf("NoModifyValueByReflect() wrong error message: %v", errs[0].Detail)
				}
			}
		})
	}
}

func TestNoModifyPointer(t *testing.T) {
	stringPtr := func(s string) *string { return &s }

	tests := []struct {
		name     string
		op       operation.Type
		value    *string
		oldValue *string
		wantErr  bool
	}{
		{
			name:     "update - nil to non-nil transition (allowed)",
			op:       operation.Update,
			value:    stringPtr("value"),
			oldValue: nil,
			wantErr:  false,
		},
		{
			name:     "update - non-nil to nil transition (allowed)",
			op:       operation.Update,
			value:    nil,
			oldValue: stringPtr("value"),
			wantErr:  false,
		},
		{
			name:     "update - different values (forbidden)",
			op:       operation.Update,
			value:    stringPtr("value2"),
			oldValue: stringPtr("value1"),
			wantErr:  true,
		},
		{
			name:     "update - same values (allowed)",
			op:       operation.Update,
			value:    stringPtr("value"),
			oldValue: stringPtr("value"),
			wantErr:  false,
		},
		{
			name:     "update - both nil (allowed)",
			op:       operation.Update,
			value:    nil,
			oldValue: nil,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := NoModifyPointer(context.TODO(), op, field.NewPath("test"), tt.value, tt.oldValue)
			if (len(errs) > 0) != tt.wantErr {
				t.Errorf("NoModifyPointer() error = %v, wantErr %v", errs, tt.wantErr)
			}
			if tt.wantErr && len(errs) > 0 {
				if errs[0].Detail != "field cannot be modified once set" {
					t.Errorf("NoModifyPointer() wrong error message: %v", errs[0].Detail)
				}
			}
		})
	}
}

func TestNoModifyStruct(t *testing.T) {
	type TestStruct struct {
		Field1 string
		Field2 int
	}

	tests := []struct {
		name     string
		op       operation.Type
		value    TestStruct
		oldValue TestStruct
		wantErr  bool
	}{
		{
			name:     "create operation - no validation",
			op:       operation.Create,
			value:    TestStruct{Field1: "test", Field2: 42},
			oldValue: TestStruct{},
			wantErr:  false,
		},
		{
			name:     "update - different values (forbidden)",
			op:       operation.Update,
			value:    TestStruct{Field1: "test2", Field2: 100},
			oldValue: TestStruct{Field1: "test1", Field2: 42},
			wantErr:  true,
		},
		{
			name:     "update - same values (allowed)",
			op:       operation.Update,
			value:    TestStruct{Field1: "test", Field2: 42},
			oldValue: TestStruct{Field1: "test", Field2: 42},
			wantErr:  false,
		},
		{
			name:     "update - zero value to non-zero (forbidden)",
			op:       operation.Update,
			value:    TestStruct{Field1: "test", Field2: 42},
			oldValue: TestStruct{},
			wantErr:  true,
		},
		{
			name:     "update - non-zero to zero value (forbidden)",
			op:       operation.Update,
			value:    TestStruct{},
			oldValue: TestStruct{Field1: "test", Field2: 42},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := operation.Operation{Type: tt.op}
			errs := NoModifyStruct(context.TODO(), op, field.NewPath("test"), &tt.value, &tt.oldValue)
			if (len(errs) > 0) != tt.wantErr {
				t.Errorf("NoModifyStruct() error = %v, wantErr %v", errs, tt.wantErr)
			}
			if tt.wantErr && len(errs) > 0 {
				if errs[0].Detail != "field cannot be modified once set" {
					t.Errorf("NoModifyStruct() wrong error message: %v", errs[0].Detail)
				}
			}
		})
	}
}
