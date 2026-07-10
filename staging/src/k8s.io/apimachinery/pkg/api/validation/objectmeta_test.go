/*
Copyright 2017 The Kubernetes Authors.

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

package validation

import (
	"context"
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const (
	maxLengthErrMsg = "must be no more than"
	namePartErrMsg  = "name part must consist of"
	nameErrMsg      = "a valid label key must consist of"
)

// Ensure custom name functions are allowed
func TestValidateObjectMetaCustomName(t *testing.T) {
	testCases := []struct {
		name   string
		input  metav1.ObjectMeta
		nErrs  int
		errStr string
	}{{
		name:  "valid name, empty generateName",
		input: metav1.ObjectMeta{Name: "test", GenerateName: ""},
	}, {
		name:  "valid name and generateName",
		input: metav1.ObjectMeta{Name: "test", GenerateName: "test"},
	}, {
		name:   "invalid name, empty generateName",
		input:  metav1.ObjectMeta{Name: "invalid", GenerateName: ""},
		nErrs:  1,
		errStr: "wrong value",
	}, {
		name:   "invalid name, valid generateName",
		input:  metav1.ObjectMeta{Name: "invalid", GenerateName: "test"},
		nErrs:  1,
		errStr: "wrong value",
	}, {
		name:   "invalid name, invalid generateName",
		input:  metav1.ObjectMeta{Name: "invalid", GenerateName: "invalid"},
		nErrs:  2,
		errStr: "wrong value",
	}}

	fn := func(s string, prefix bool) []string {
		// Note: this is called on both name and generateName
		if s == "test" {
			return nil
		}
		return []string{"wrong value"}
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := ValidateObjectMeta(&tc.input, false, fn, field.NewPath("field"))

			if len(errs) == 0 {
				if len(tc.errStr) != 0 {
					t.Fatalf("expected 1 error, got none")
				}
			} else {
				if len(tc.errStr) == 0 {
					t.Fatalf("expected no errors, got: %v", errs)
				}
				if len(errs) != tc.nErrs {
					t.Fatalf("expected %d errors, got %d: %q", tc.nErrs, len(errs), errs)
				}
				if !strings.Contains(errs[0].Error(), "wrong value") {
					t.Errorf("unexpected error message: %v", errs[0].Error())
				}
			}
		})
	}
}

// Ensure custom name functions work
func TestValidateObjectMetaWithOptsName(t *testing.T) {
	testCases := []struct {
		name   string
		input  metav1.ObjectMeta
		errStr string
	}{{
		name:  "valid name, empty generateName",
		input: metav1.ObjectMeta{Name: "test", GenerateName: ""},
	}, {
		name:  "valid name and generateName",
		input: metav1.ObjectMeta{Name: "test", GenerateName: "test"},
	}, {
		name:   "invalid name, empty generateName",
		input:  metav1.ObjectMeta{Name: "invalid", GenerateName: ""},
		errStr: "wrong value",
	}, {
		name:   "invalid name, valid generateName",
		input:  metav1.ObjectMeta{Name: "invalid", GenerateName: "test"},
		errStr: "wrong value",
	}, {
		name:   "invalid name, invalid generateName",
		input:  metav1.ObjectMeta{Name: "invalid", GenerateName: "invalid"},
		errStr: "wrong value",
	}}

	fn := func(fldPath *field.Path, s string) field.ErrorList {
		if s == "test" {
			return nil
		}
		return field.ErrorList{field.Invalid(fldPath, s, "wrong value")}
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := ValidateObjectMetaWithOpts(&tc.input, false, fn, field.NewPath("field"))

			if len(errs) == 0 {
				if len(tc.errStr) != 0 {
					t.Fatalf("expected 1 error, got none")
				}
			} else {
				if len(tc.errStr) == 0 {
					t.Fatalf("expected no errors, got: %v", errs)
				}
				if len(errs) != 1 {
					t.Fatalf("expected 1 error, got %d: %q", len(errs), errs)
				}
				if !strings.Contains(errs[0].Error(), "wrong value") {
					t.Errorf("unexpected error message: %v", errs[0].Error())
				}
			}
		})
	}
}

// Ensure namespace names follow dns label format
func TestValidateObjectMetaNamespaces(t *testing.T) {
	errs := validateObjectMetaAccessorWithOptsCommon(
		&metav1.ObjectMeta{Name: "test", Namespace: "foo.bar"},
		true, field.NewPath("field"), nil)
	if len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !strings.Contains(errs[0].Error(), `Invalid value: "foo.bar"`) {
		t.Errorf("unexpected error message: %v", errs)
	}
	maxLength := 63
	letters := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
	b := make([]rune, maxLength+1)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	errs = validateObjectMetaAccessorWithOptsCommon(
		&metav1.ObjectMeta{Name: "test", Namespace: string(b)},
		true, field.NewPath("field"), nil)
	if len(errs) != 2 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !strings.Contains(errs[0].Error(), "Invalid value") || !strings.Contains(errs[1].Error(), "Invalid value") {
		t.Errorf("unexpected error message: %v", errs)
	}
}

func TestValidateObjectMetaOwnerReferences(t *testing.T) {
	trueVar := true
	falseVar := false
	testCases := []struct {
		description          string
		ownerReferences      []metav1.OwnerReference
		expectError          bool
		expectedErrorMessage string
	}{
		{
			description: "simple success - third party extension.",
			ownerReferences: []metav1.OwnerReference{
				{
					APIVersion: "customresourceVersion",
					Kind:       "customresourceKind",
					Name:       "name",
					UID:        "1",
				},
			},
			expectError:          false,
			expectedErrorMessage: "",
		},
		{
			description: "simple failures - event shouldn't be set as an owner",
			ownerReferences: []metav1.OwnerReference{
				{
					APIVersion: "v1",
					Kind:       "Event",
					Name:       "name",
					UID:        "1",
				},
			},
			expectError:          true,
			expectedErrorMessage: "is disallowed from being an owner",
		},
		{
			description: "simple controller ref success - one reference with Controller set",
			ownerReferences: []metav1.OwnerReference{
				{
					APIVersion: "customresourceVersion",
					Kind:       "customresourceKind",
					Name:       "name",
					UID:        "1",
					Controller: &falseVar,
				},
				{
					APIVersion: "customresourceVersion",
					Kind:       "customresourceKind",
					Name:       "name",
					UID:        "2",
					Controller: &trueVar,
				},
				{
					APIVersion: "customresourceVersion",
					Kind:       "customresourceKind",
					Name:       "name",
					UID:        "3",
					Controller: &falseVar,
				},
				{
					APIVersion: "customresourceVersion",
					Kind:       "customresourceKind",
					Name:       "name",
					UID:        "4",
				},
			},
			expectError:          false,
			expectedErrorMessage: "",
		},
		{
			description: "simple controller ref failure - two references with Controller set",
			ownerReferences: []metav1.OwnerReference{
				{
					APIVersion: "customresourceVersion",
					Kind:       "customresourceKind1",
					Name:       "name",
					UID:        "1",
					Controller: &falseVar,
				},
				{
					APIVersion: "customresourceVersion",
					Kind:       "customresourceKind2",
					Name:       "name",
					UID:        "2",
					Controller: &trueVar,
				},
				{
					APIVersion: "customresourceVersion",
					Kind:       "customresourceKind3",
					Name:       "name",
					UID:        "3",
					Controller: &trueVar,
				},
				{
					APIVersion: "customresourceVersion",
					Kind:       "customresourceKind4",
					Name:       "name",
					UID:        "4",
				},
			},
			expectError:          true,
			expectedErrorMessage: "Only one reference can have Controller set to true. Found \"true\" in references for customresourceKind2/name and customresourceKind3/name",
		},
	}

	for _, tc := range testCases {
		errs := validateObjectMetaAccessorWithOptsCommon(
			&metav1.ObjectMeta{Name: "test", Namespace: "test", OwnerReferences: tc.ownerReferences},
			true, field.NewPath("field"), nil)
		if len(errs) != 0 && !tc.expectError {
			t.Errorf("unexpected error: %v in test case %v", errs, tc.description)
		}
		if len(errs) == 0 && tc.expectError {
			t.Errorf("expect error in test case %v", tc.description)
		}
		if len(errs) != 0 && !strings.Contains(errs[0].Error(), tc.expectedErrorMessage) {
			t.Errorf("unexpected error message: %v in test case %v", errs, tc.description)
		}
	}
}

func TestValidateObjectMetaUpdateIgnoresCreationTimestamp(t *testing.T) {
	if errs := ValidateObjectMetaUpdate(
		&metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
		&metav1.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: metav1.NewTime(time.Unix(10, 0))},
		field.NewPath("field"),
	); len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if errs := ValidateObjectMetaUpdate(
		&metav1.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: metav1.NewTime(time.Unix(10, 0))},
		&metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
		field.NewPath("field"),
	); len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if errs := ValidateObjectMetaUpdate(
		&metav1.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: metav1.NewTime(time.Unix(10, 0))},
		&metav1.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: metav1.NewTime(time.Unix(11, 0))},
		field.NewPath("field"),
	); len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
}

func TestValidateFinalizersUpdate(t *testing.T) {
	testcases := map[string]struct {
		Old         metav1.ObjectMeta
		New         metav1.ObjectMeta
		ExpectedErr string
	}{
		"invalid adding finalizers": {
			Old:         metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &metav1.Time{}, Finalizers: []string{"x/a"}},
			New:         metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &metav1.Time{}, Finalizers: []string{"x/a", "y/b"}},
			ExpectedErr: "y/b",
		},
		"invalid changing finalizers": {
			Old:         metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &metav1.Time{}, Finalizers: []string{"x/a"}},
			New:         metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &metav1.Time{}, Finalizers: []string{"x/b"}},
			ExpectedErr: "x/b",
		},
		"valid removing finalizers": {
			Old:         metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &metav1.Time{}, Finalizers: []string{"x/a", "y/b"}},
			New:         metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &metav1.Time{}, Finalizers: []string{"x/a"}},
			ExpectedErr: "",
		},
		"valid adding finalizers for objects not being deleted": {
			Old:         metav1.ObjectMeta{Name: "test", ResourceVersion: "1", Finalizers: []string{"x/a"}},
			New:         metav1.ObjectMeta{Name: "test", ResourceVersion: "1", Finalizers: []string{"x/a", "y/b"}},
			ExpectedErr: "",
		},
	}
	for name, tc := range testcases {
		errs := ValidateObjectMetaUpdate(&tc.New, &tc.Old, field.NewPath("field"))
		if len(errs) == 0 {
			if len(tc.ExpectedErr) != 0 {
				t.Errorf("case: %q, expected error to contain %q", name, tc.ExpectedErr)
			}
		} else if e, a := tc.ExpectedErr, errs.ToAggregate().Error(); !strings.Contains(a, e) {
			t.Errorf("case: %q, expected error to contain %q, got error %q", name, e, a)
		}
	}
}

func TestValidateFinalizersPreventConflictingFinalizers(t *testing.T) {
	testcases := map[string]struct {
		ObjectMeta  metav1.ObjectMeta
		ExpectedErr string
	}{
		"conflicting finalizers": {
			ObjectMeta:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", Finalizers: []string{metav1.FinalizerOrphanDependents, metav1.FinalizerDeleteDependents}},
			ExpectedErr: "cannot be both set",
		},
	}
	for name, tc := range testcases {
		errs := validateObjectMetaAccessorWithOptsCommon(&tc.ObjectMeta, false, field.NewPath("field"), nil)
		if len(errs) == 0 {
			if len(tc.ExpectedErr) != 0 {
				t.Errorf("case: %q, expected error to contain %q", name, tc.ExpectedErr)
			}
		} else if e, a := tc.ExpectedErr, errs.ToAggregate().Error(); !strings.Contains(a, e) {
			t.Errorf("case: %q, expected error to contain %q, got error %q", name, e, a)
		}
	}
}

func TestValidateObjectMetaUpdatePreventsDeletionFieldMutation(t *testing.T) {
	now := metav1.NewTime(time.Unix(1000, 0).UTC())
	later := metav1.NewTime(time.Unix(2000, 0).UTC())
	gracePeriodShort := int64(30)
	gracePeriodLong := int64(40)

	testcases := map[string]struct {
		Old          metav1.ObjectMeta
		New          metav1.ObjectMeta
		ExpectedNew  metav1.ObjectMeta
		ExpectedErrs []string
	}{
		"valid without deletion fields": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedErrs: []string{},
		},
		"valid with deletion fields": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now, DeletionGracePeriodSeconds: &gracePeriodShort},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now, DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now, DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedErrs: []string{},
		},

		"invalid set deletionTimestamp": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			ExpectedErrs: []string{"field.deletionTimestamp: Invalid value: \"1970-01-01T00:16:40Z\": field is immutable"},
		},
		"invalid clear deletionTimestamp": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedErrs: []string{"field.deletionTimestamp: Invalid value: null: field is immutable"},
		},
		"invalid change deletionTimestamp": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &later},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &later},
			ExpectedErrs: []string{"field.deletionTimestamp: Invalid value: \"1970-01-01T00:33:20Z\": field is immutable"},
		},

		"invalid set deletionGracePeriodSeconds": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedErrs: []string{"field.deletionGracePeriodSeconds: Invalid value: 30: field is immutable"},
		},
		"invalid clear deletionGracePeriodSeconds": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedErrs: []string{"field.deletionGracePeriodSeconds: Invalid value: null: field is immutable"},
		},
		"invalid change deletionGracePeriodSeconds": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodLong},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodLong},
			ExpectedErrs: []string{"field.deletionGracePeriodSeconds: Invalid value: 40: field is immutable"},
		},
	}

	for k, tc := range testcases {
		errs := ValidateObjectMetaUpdate(&tc.New, &tc.Old, field.NewPath("field"))
		if len(errs) != len(tc.ExpectedErrs) {
			t.Logf("%s: Expected: %#v", k, tc.ExpectedErrs)
			t.Logf("%s: Got: %#v", k, errs)
			t.Errorf("%s: expected %d errors, got %d", k, len(tc.ExpectedErrs), len(errs))
			continue
		}
		for i := range errs {
			if errs[i].Error() != tc.ExpectedErrs[i] {
				t.Errorf("%s: error #%d:\n  expected: %q\n       got: %q", k, i, tc.ExpectedErrs[i], errs[i].Error())
			}
		}
		if !reflect.DeepEqual(tc.New, tc.ExpectedNew) {
			t.Errorf("%s: Expected after validation:\n%#v\ngot\n%#v", k, tc.ExpectedNew, tc.New)
		}
	}
}

func TestObjectMetaGenerationUpdate(t *testing.T) {
	testcases := map[string]struct {
		Old          metav1.ObjectMeta
		New          metav1.ObjectMeta
		ExpectedErrs []string
	}{
		"invalid generation change - decremented": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", Generation: 5},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", Generation: 4},
			ExpectedErrs: []string{"field.generation: Invalid value: 4: must not be decremented"},
		},
		"valid generation change - incremented by one": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", Generation: 1},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", Generation: 2},
			ExpectedErrs: []string{},
		},
		"valid generation field - not updated": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", Generation: 5},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", Generation: 5},
			ExpectedErrs: []string{},
		},
	}

	for k, tc := range testcases {
		errList := []string{}
		errs := ValidateObjectMetaUpdate(&tc.New, &tc.Old, field.NewPath("field"))
		if len(errs) != len(tc.ExpectedErrs) {
			t.Logf("%s: Expected: %#v", k, tc.ExpectedErrs)
			for _, err := range errs {
				errList = append(errList, err.Error())
			}
			t.Logf("%s: Got: %#v", k, errList)
			t.Errorf("%s: expected %d errors, got %d", k, len(tc.ExpectedErrs), len(errs))
			continue
		}
		for i := range errList {
			if errList[i] != tc.ExpectedErrs[i] {
				t.Errorf("%s: error #%d:\n  expected: %q\n       got: %q", k, i, tc.ExpectedErrs[i], errs[i].Error())
			}
		}
	}
}

// Ensure trailing dash is allowed in generate name
func TestValidateObjectMetaTrimsTrailingDash(t *testing.T) {
	errs := ValidateObjectMeta(
		&metav1.ObjectMeta{Name: "test", GenerateName: "foo-"},
		false,
		NameIsDNSSubdomain,
		field.NewPath("field"))
	if len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
}

func TestValidateAnnotations(t *testing.T) {
	successCases := []map[string]string{
		{"simple": "bar"},
		{"now-with-dashes": "bar"},
		{"1-starts-with-num": "bar"},
		{"1234": "bar"},
		{"simple/simple": "bar"},
		{"now-with-dashes/simple": "bar"},
		{"now-with-dashes/now-with-dashes": "bar"},
		{"now.with.dots/simple": "bar"},
		{"now-with.dashes-and.dots/simple": "bar"},
		{"1-num.2-num/3-num": "bar"},
		{"1234/5678": "bar"},
		{"1.2.3.4/5678": "bar"},
		{"UpperCase123": "bar"},
		{"a": strings.Repeat("b", TotalAnnotationSizeLimitB-1)},
		{
			"a": strings.Repeat("b", TotalAnnotationSizeLimitB/2-1),
			"c": strings.Repeat("d", TotalAnnotationSizeLimitB/2-1),
		},
	}
	for i := range successCases {
		errs := ValidateAnnotations(successCases[i], field.NewPath("field"))
		if len(errs) != 0 {
			t.Errorf("case[%d] expected success, got %#v", i, errs)
		}
	}

	nameErrorCases := []struct {
		annotations map[string]string
		expect      string
	}{
		{map[string]string{"nospecialchars^=@": "bar"}, namePartErrMsg},
		{map[string]string{"cantendwithadash-": "bar"}, namePartErrMsg},
		{map[string]string{"only/one/slash": "bar"}, nameErrMsg},
		{map[string]string{strings.Repeat("a", 254): "bar"}, maxLengthErrMsg},
	}
	for i := range nameErrorCases {
		errs := ValidateAnnotations(nameErrorCases[i].annotations, field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d]: expected failure", i)
		} else {
			if !strings.Contains(errs[0].Detail, nameErrorCases[i].expect) {
				t.Errorf("case[%d]: error details do not include %q: %q", i, nameErrorCases[i].expect, errs[0].Detail)
			}
		}
	}
	totalSizeErrorCases := []map[string]string{
		{"a": strings.Repeat("b", TotalAnnotationSizeLimitB)},
		{
			"a": strings.Repeat("b", TotalAnnotationSizeLimitB/2),
			"c": strings.Repeat("d", TotalAnnotationSizeLimitB/2),
		},
	}
	for i := range totalSizeErrorCases {
		errs := ValidateAnnotations(totalSizeErrorCases[i], field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		}
	}
}

func TestValidateObjectMetaDeclaratively(t *testing.T) {
	ctx := context.Background()
	fldPath := field.NewPath("metadata")
	now := metav1.NewTime(time.Unix(1000, 0).UTC())
	later := metav1.NewTime(time.Unix(2000, 0).UTC())
	gracePeriod30 := int64(30)
	gracePeriod40 := int64(40)

	createCases := []struct {
		name              string
		obj               *metav1.ObjectMeta
		requiresNamespace bool
		expectedErrs      field.ErrorList
	}{
		{
			name:              "valid metadata",
			obj:               mkMeta(),
			requiresNamespace: true,
			expectedErrs:      nil,
		},
		{
			name:              "invalid name format",
			obj:               mkMeta(tweakName("invalid_name")),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("name"), "invalid_name", "").MarkFromImperative(),
			},
		},
		{
			name:              "missing required namespace",
			obj:               mkMeta(tweakNamespace("")),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Required(fldPath.Child("namespace"), "").MarkFromImperative(),
			},
		},
		{
			name:              "negative generation",
			obj:               mkMeta(tweakGeneration(-1)),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("generation"), int64(-1), "").WithOrigin("minimum").MarkAlpha(),
			},
		},
		{
			name:              "managedFields empty operation",
			obj:               mkMeta(tweakManagedFields(metav1.ManagedFieldsEntry{FieldsType: "FieldsV1"})),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Required(fldPath.Child("managedFields").Index(0).Child("operation"), "").MarkAlpha(),
			},
		},
		{
			name:              "invalid annotation key",
			obj:               mkMeta(tweakAnnotations(map[string]string{"-invalid": "val"})),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("annotations"), "-invalid", "").WithOrigin("format=k8s-label-key").MarkFromImperative(),
			},
		},
		{
			name:              "annotations size limit exceeded",
			obj:               mkMeta(tweakAnnotations(map[string]string{"a": strings.Repeat("b", TotalAnnotationSizeLimitB)})),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.TooLong(fldPath.Child("annotations"), "", TotalAnnotationSizeLimitB).MarkFromImperative(),
			},
		},
	}

	matcher := field.ErrorMatcher{}.ByField().ByType().BySource().ByOrigin()

	toExpectedErrs := func(allDeclarativeEnforced bool, errs field.ErrorList) field.ErrorList {
		expected := make(field.ErrorList, 0, len(errs))
		for _, err := range errs {
			e := *err
			if !allDeclarativeEnforced && (e.IsAlpha() || e.IsBeta()) {
				_ = e.MarkFromImperative()
				e.ValidationStabilityLevel = 0
			}
			expected = append(expected, &e)
		}
		return expected
	}

	for _, tc := range createCases {
		for _, betaEnabled := range []bool{true, false} {
			for _, allDeclarativeEnforced := range []bool{true, false} {
				t.Run(fmt.Sprintf("Create: %s (betaEnabled=%v, allDeclarativeEnforced=%v)", tc.name, betaEnabled, allDeclarativeEnforced), func(t *testing.T) {
					testCtx := ctx
					if allDeclarativeEnforced {
						testCtx = validate.WithAllDeclarativeEnforcedForTest(ctx)
					}
					errs := ValidateObjectMetaDeclaratively(testCtx, operation.Create, tc.obj, nil, tc.requiresNamespace, NameIsDNSSubdomain, fldPath, betaEnabled)
					matcher.Test(t, toExpectedErrs(allDeclarativeEnforced, tc.expectedErrs), errs)
				})
			}
		}
	}

	updateCases := []struct {
		name              string
		obj               *metav1.ObjectMeta
		oldObj            *metav1.ObjectMeta
		requiresNamespace bool
		expectedErrs      field.ErrorList
	}{
		{
			name:              "valid update",
			obj:               mkMeta(tweakResourceVersion("2")),
			oldObj:            mkMeta(tweakResourceVersion("1")),
			requiresNamespace: true,
			expectedErrs:      nil,
		},
		{
			name:              "immutable namespace",
			obj:               mkMeta(tweakNamespace("new-ns"), tweakResourceVersion("2")),
			oldObj:            mkMeta(tweakNamespace("old-ns"), tweakResourceVersion("1")),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("namespace"), "new-ns", "").MarkFromImperative(),
			},
		},
		{
			name:              "invalid annotation key on update",
			obj:               mkMeta(tweakResourceVersion("2"), tweakAnnotations(map[string]string{"-invalid": "val"})),
			oldObj:            mkMeta(tweakResourceVersion("1")),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("annotations"), "-invalid", "").WithOrigin("format=k8s-label-key").MarkFromImperative(),
			},
		},
		{
			name:              "immutable uid on update",
			obj:               mkMeta(tweakResourceVersion("2"), tweakUID("uid-new")),
			oldObj:            mkMeta(tweakResourceVersion("1"), tweakUID("uid-old")),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("uid"), types.UID("uid-new"), "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		{
			name:              "immutable creationTimestamp on update",
			obj:               mkMeta(tweakResourceVersion("2"), tweakCreationTimestamp(later)),
			oldObj:            mkMeta(tweakResourceVersion("1"), tweakCreationTimestamp(now)),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("creationTimestamp"), later, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		{
			name:              "immutable deletionTimestamp on update",
			obj:               mkMeta(tweakResourceVersion("2"), tweakDeletionTimestamp(&later)),
			oldObj:            mkMeta(tweakResourceVersion("1"), tweakDeletionTimestamp(&now)),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("deletionTimestamp"), &later, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		{
			name:              "immutable deletionGracePeriodSeconds on update",
			obj:               mkMeta(tweakResourceVersion("2"), tweakDeletionGracePeriodSeconds(&gracePeriod40)),
			oldObj:            mkMeta(tweakResourceVersion("1"), tweakDeletionGracePeriodSeconds(&gracePeriod30)),
			requiresNamespace: true,
			expectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("deletionGracePeriodSeconds"), &gracePeriod40, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}

	for _, tc := range updateCases {
		for _, betaEnabled := range []bool{true, false} {
			for _, allDeclarativeEnforced := range []bool{true, false} {
				t.Run(fmt.Sprintf("Update: %s (betaEnabled=%v, allDeclarativeEnforced=%v)", tc.name, betaEnabled, allDeclarativeEnforced), func(t *testing.T) {
					testCtx := ctx
					if allDeclarativeEnforced {
						testCtx = validate.WithAllDeclarativeEnforcedForTest(ctx)
					}
					errs := ValidateObjectMetaDeclaratively(testCtx, operation.Update, tc.obj, tc.oldObj, tc.requiresNamespace, NameIsDNSSubdomain, fldPath, betaEnabled)
					matcher.Test(t, toExpectedErrs(allDeclarativeEnforced, tc.expectedErrs), errs)
				})
			}
		}
	}
}

func mkMeta(tweaks ...func(*metav1.ObjectMeta)) *metav1.ObjectMeta {
	obj := &metav1.ObjectMeta{
		Name:      "valid-name",
		Namespace: "valid-ns",
	}
	for _, tweak := range tweaks {
		tweak(obj)
	}
	return obj
}

func tweakName(n string) func(*metav1.ObjectMeta) {
	return func(o *metav1.ObjectMeta) { o.Name = n }
}

func tweakNamespace(ns string) func(*metav1.ObjectMeta) {
	return func(o *metav1.ObjectMeta) { o.Namespace = ns }
}

func tweakResourceVersion(rv string) func(*metav1.ObjectMeta) {
	return func(o *metav1.ObjectMeta) { o.ResourceVersion = rv }
}

func tweakGeneration(g int64) func(*metav1.ObjectMeta) {
	return func(o *metav1.ObjectMeta) { o.Generation = g }
}

func tweakAnnotations(ann map[string]string) func(*metav1.ObjectMeta) {
	return func(o *metav1.ObjectMeta) { o.Annotations = ann }
}

func tweakManagedFields(entries ...metav1.ManagedFieldsEntry) func(*metav1.ObjectMeta) {
	return func(o *metav1.ObjectMeta) { o.ManagedFields = entries }
}

func tweakUID(u string) func(*metav1.ObjectMeta) {
	return func(o *metav1.ObjectMeta) { o.UID = types.UID(u) }
}

func tweakCreationTimestamp(t metav1.Time) func(*metav1.ObjectMeta) {
	return func(o *metav1.ObjectMeta) { o.CreationTimestamp = t }
}

func tweakDeletionTimestamp(t *metav1.Time) func(*metav1.ObjectMeta) {
	return func(o *metav1.ObjectMeta) { o.DeletionTimestamp = t }
}

func tweakDeletionGracePeriodSeconds(gps *int64) func(*metav1.ObjectMeta) {
	return func(o *metav1.ObjectMeta) { o.DeletionGracePeriodSeconds = gps }
}
