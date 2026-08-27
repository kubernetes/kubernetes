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

package validators

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/gengo/v2/types"
)

var testValidateFn = types.Name{Package: libValidationPkg, Name: "FakeValidator"}

func TestWrapGroup_EmptyEnvelopePassthrough(t *testing.T) {
	fn := Function("test", DefaultFlags, testValidateFn)
	got, err := WrapGroup(Context{Type: types.String}, ValidationGroup{
		Validations: Validations{Functions: []FunctionGen{fn}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Functions) != 1 {
		t.Fatalf("expected 1 function, got %d", len(got.Functions))
	}
	if got.Functions[0].TagName != "test" || got.Functions[0].StabilityLevel != "" {
		t.Errorf("expected unwrapped function, got %+v", got.Functions[0])
	}
}

func TestWrapGroup_ConditionsThenStability(t *testing.T) {
	fn := Function("test", DefaultFlags, testValidateFn)
	got, err := WrapGroup(Context{Type: types.String}, ValidationGroup{
		Validations:    Validations{Functions: []FunctionGen{fn}},
		Conditions:     Conditions{}.WithOptionEnabled("Foo"),
		StabilityLevel: ValidationStabilityLevelAlpha,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Functions) != 1 {
		t.Fatalf("expected 1 function, got %d", len(got.Functions))
	}
	wrapper := got.Functions[0]
	if wrapper.TagName != ifEnabledTag {
		t.Errorf("expected conditions wrapper %q, got %q", ifEnabledTag, wrapper.TagName)
	}
	if wrapper.StabilityLevel != ValidationStabilityLevelAlpha {
		t.Errorf("expected stability level applied to the conditions wrapper, got %q", wrapper.StabilityLevel)
	}
	inner, ok := wrapper.Args[2].(WrapperFunction)
	if !ok {
		t.Fatalf("expected WrapperFunction arg, got %T", wrapper.Args[2])
	}
	if inner.Function.TagName != "test" {
		t.Errorf("expected original function inside wrapper, got %q", inner.Function.TagName)
	}
}

func TestWrapGroup_FieldTypeSelectsWrapperObjType(t *testing.T) {
	targetType := &types.Type{Name: types.Name{Package: "example.com/pkg", Name: "Target"}, Kind: types.Struct}
	fn := Function("test", DefaultFlags, testValidateFn)
	got, err := WrapGroup(Context{Type: types.String}, ValidationGroup{
		Validations: Validations{Functions: []FunctionGen{fn}},
		Conditions:  Conditions{}.WithOptionEnabled("Foo"),
		FieldType:  targetType,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	inner := got.Functions[0].Args[2].(WrapperFunction)
	if inner.ObjType != targetType {
		t.Errorf("expected wrapper ObjType %v, got %v", targetType, inner.ObjType)
	}
}

func TestWrapGroup_TypeFunctionsWrapped(t *testing.T) {
	fn := Function("test", DefaultFlags, testValidateFn)
	got, err := WrapGroup(Context{Type: types.String}, ValidationGroup{
		Validations:    Validations{TypeFunctions: []FunctionGen{fn}},
		Conditions:     Conditions{}.WithOptionEnabled("Foo"),
		StabilityLevel: ValidationStabilityLevelBeta,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.TypeFunctions) != 1 {
		t.Fatalf("expected 1 type function, got %d", len(got.TypeFunctions))
	}
	wrapper := got.TypeFunctions[0]
	if wrapper.TagName != ifEnabledTag {
		t.Errorf("expected conditions wrapper %q, got %q", ifEnabledTag, wrapper.TagName)
	}
	if wrapper.StabilityLevel != ValidationStabilityLevelBeta {
		t.Errorf("expected stability level on type function wrapper, got %q", wrapper.StabilityLevel)
	}
}

func TestWrapGroup_SelfManagedStabilitySkipped(t *testing.T) {
	fn := Function("test", DefaultFlags, testValidateFn)
	fn.StabilityLevelSelfManaged = true
	got, err := WrapGroup(Context{Type: types.String}, ValidationGroup{
		Validations:    Validations{Functions: []FunctionGen{fn}},
		StabilityLevel: ValidationStabilityLevelAlpha,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Functions[0].StabilityLevel != "" {
		t.Errorf("expected self-managed function to keep empty stability level, got %q", got.Functions[0].StabilityLevel)
	}
}

func TestWrapGroup_SubfieldFieldPath(t *testing.T) {
	subType := types.String
	structType := &types.Type{
		Name: types.Name{Package: "example.com/pkg", Name: "Struct"},
		Kind: types.Struct,
		Members: []types.Member{{
			Name: "Sub",
			Type: subType,
			Tags: `json:"sub"`,
		}},
	}
	fn := Function("test", DefaultFlags, testValidateFn)
	got, err := WrapGroup(
		Context{Type: structType, Path: field.NewPath("spec")},
		ValidationGroup{
			Validations: Validations{Functions: []FunctionGen{fn}},
			FieldPath:  field.NewPath("spec", "sub"),
		})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Functions) != 1 {
		t.Fatalf("expected 1 function, got %d", len(got.Functions))
	}
	wrapper := got.Functions[0]
	if wrapper.TagName != subfieldTagName {
		t.Errorf("expected subfield wrapper %q, got %q", subfieldTagName, wrapper.TagName)
	}
	if wrapper.Cohort != "sub" {
		t.Errorf("expected cohort %q, got %q", "sub", wrapper.Cohort)
	}
}

func TestWrapGroup_NonDescendantFieldPathErrors(t *testing.T) {
	fn := Function("test", DefaultFlags, testValidateFn)
	_, err := WrapGroup(
		Context{Type: types.String, Path: field.NewPath("spec")},
		ValidationGroup{
			Validations: Validations{Functions: []FunctionGen{fn}},
			FieldPath:  field.NewPath("status", "sub"),
		})
	if err == nil {
		t.Fatal("expected error for non-descendant target path, got nil")
	}
}

func TestWrapGroup_UnresolvableFieldPathErrors(t *testing.T) {
	fn := Function("test", DefaultFlags, testValidateFn)
	_, err := WrapGroup(
		Context{Type: types.String, Path: field.NewPath("spec")},
		ValidationGroup{
			Validations: Validations{Functions: []FunctionGen{fn}},
			FieldPath:  field.NewPath("spec", "nosuch"),
		})
	if err == nil {
		t.Fatal("expected error for unresolvable target path, got nil")
	}
}

func TestWrapGroup_EmitOnTypeMovesFunctionsToTypeFunctions(t *testing.T) {
	fn := Function("test", DefaultFlags, testValidateFn)
	got, err := WrapGroup(Context{Type: types.String}, ValidationGroup{
		Validations: Validations{Functions: []FunctionGen{fn}},
		Conditions:  Conditions{}.WithOptionEnabled("Foo"),
		EmitOnType:       true,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got.Functions) != 0 {
		t.Errorf("expected no plain functions after hoisting, got %d", len(got.Functions))
	}
	if len(got.TypeFunctions) != 1 {
		t.Fatalf("expected 1 type function after hoisting, got %d", len(got.TypeFunctions))
	}
	if got.TypeFunctions[0].TagName != ifEnabledTag {
		t.Errorf("expected hoisted function to be condition-wrapped, got %q", got.TypeFunctions[0].TagName)
	}
}

func TestRegisterTypeValidator_ChainOrder(t *testing.T) {
	wantOrder := []string{"unions", "modes"}
	if len(typeValidators) != len(wantOrder) {
		t.Fatalf("expected %d registered aggregate emitters, got %d", len(wantOrder), len(typeValidators))
	}
	for i, want := range wantOrder {
		if got := typeValidators[i].emitter.Name(); got != want {
			t.Errorf("aggregate emitter %d: expected %q, got %q", i, want, got)
		}
	}
}

func TestRegisterValidationWrapper_ChainOrder(t *testing.T) {
	wantOrder := []string{"conditions", "stability", "subfield"}
	if len(validationWrappers) != len(wantOrder) {
		t.Fatalf("expected %d registered finalizers, got %d", len(wantOrder), len(validationWrappers))
	}
	for i, want := range wantOrder {
		if got := validationWrappers[i].finalizer.Name(); got != want {
			t.Errorf("finalizer %d: expected %q, got %q", i, want, got)
		}
	}
}
