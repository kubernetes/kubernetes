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
	"math/rand"
	"reflect"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const (
	maxLengthErrMsg = "must be no more than"
	namePartErrMsg  = "name part must consist of"
	nameErrMsg      = "a qualified name must consist of"
)

// Ensure custom name functions are allowed
func TestValidateObjectMetaCustomName(t *testing.T) {
	errs := ValidateObjectMeta(
		&metav1.ObjectMeta{Name: "test", GenerateName: "foo"},
		false,
		func(s string, prefix bool) []string {
			if s == "test" {
				return nil
			}
			return []string{"name-gen"}
		},
		field.NewPath("field"))
	if len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !strings.Contains(errs[0].Error(), "name-gen") {
		t.Errorf("unexpected error message: %v", errs)
	}
}

// Ensure namespace names follow dns label format
func TestValidateObjectMetaNamespaces(t *testing.T) {
	errs := ValidateObjectMeta(
		&metav1.ObjectMeta{Name: "test", Namespace: "foo.bar"},
		true,
		func(s string, prefix bool) []string {
			return nil
		},
		field.NewPath("field"))
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
	errs = ValidateObjectMeta(
		&metav1.ObjectMeta{Name: "test", Namespace: string(b)},
		true,
		func(s string, prefix bool) []string {
			return nil
		},
		field.NewPath("field"))
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
					APIVersion: "thirdpartyVersion",
					Kind:       "thirdpartyKind",
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
					APIVersion: "thirdpartyVersion",
					Kind:       "thirdpartyKind",
					Name:       "name",
					UID:        "1",
					Controller: &falseVar,
				},
				{
					APIVersion: "thirdpartyVersion",
					Kind:       "thirdpartyKind",
					Name:       "name",
					UID:        "2",
					Controller: &trueVar,
				},
				{
					APIVersion: "thirdpartyVersion",
					Kind:       "thirdpartyKind",
					Name:       "name",
					UID:        "3",
					Controller: &falseVar,
				},
				{
					APIVersion: "thirdpartyVersion",
					Kind:       "thirdpartyKind",
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
					APIVersion: "thirdpartyVersion",
					Kind:       "thirdpartyKind",
					Name:       "name",
					UID:        "1",
					Controller: &falseVar,
				},
				{
					APIVersion: "thirdpartyVersion",
					Kind:       "thirdpartyKind",
					Name:       "name",
					UID:        "2",
					Controller: &trueVar,
				},
				{
					APIVersion: "thirdpartyVersion",
					Kind:       "thirdpartyKind",
					Name:       "name",
					UID:        "3",
					Controller: &trueVar,
				},
				{
					APIVersion: "thirdpartyVersion",
					Kind:       "thirdpartyKind",
					Name:       "name",
					UID:        "4",
				},
			},
			expectError:          true,
			expectedErrorMessage: "Only one reference can have Controller set to true",
		},
	}

	for _, tc := range testCases {
		errs := ValidateObjectMeta(
			&metav1.ObjectMeta{Name: "test", Namespace: "test", OwnerReferences: tc.ownerReferences},
			true,
			func(s string, prefix bool) []string {
				return nil
			},
			field.NewPath("field"))
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
	); len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if errs := ValidateObjectMetaUpdate(
		&metav1.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: metav1.NewTime(time.Unix(10, 0))},
		&metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
		field.NewPath("field"),
	); len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if errs := ValidateObjectMetaUpdate(
		&metav1.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: metav1.NewTime(time.Unix(10, 0))},
		&metav1.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: metav1.NewTime(time.Unix(11, 0))},
		field.NewPath("field"),
	); len(errs) != 0 {
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
		errs := ValidateObjectMeta(&tc.ObjectMeta, false, NameIsDNSSubdomain, field.NewPath("field"))
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
			ExpectedErrs: []string{"field.deletionTimestamp: Invalid value: 1970-01-01 00:16:40 +0000 UTC: field is immutable; may only be changed via deletion"},
		},
		"invalid clear deletionTimestamp": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			ExpectedErrs: []string{}, // no errors, validation copies the old value
		},
		"invalid change deletionTimestamp": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &later},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			ExpectedErrs: []string{}, // no errors, validation copies the old value
		},

		"invalid set deletionGracePeriodSeconds": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedErrs: []string{"field.deletionGracePeriodSeconds: Invalid value: 30: field is immutable; may only be changed via deletion"},
		},
		"invalid clear deletionGracePeriodSeconds": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedErrs: []string{}, // no errors, validation copies the old value
		},
		"invalid change deletionGracePeriodSeconds": {
			Old:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			New:          metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodLong},
			ExpectedNew:  metav1.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodLong},
			ExpectedErrs: []string{"field.deletionGracePeriodSeconds: Invalid value: 40: field is immutable; may only be changed via deletion"},
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
				t.Errorf("%s: error #%d: expected %q, got %q", k, i, tc.ExpectedErrs[i], errs[i].Error())
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
				t.Errorf("%s: error #%d: expected %q, got %q", k, i, tc.ExpectedErrs[i], errList[i])
			}
		}
	}
}

// Ensure trailing slash is allowed in generate name
func TestValidateObjectMetaTrimsTrailingSlash(t *testing.T) {
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
		{"a": strings.Repeat("b", totalAnnotationSizeLimitB-1)},
		{
			"a": strings.Repeat("b", totalAnnotationSizeLimitB/2-1),
			"c": strings.Repeat("d", totalAnnotationSizeLimitB/2-1),
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
		{"a": strings.Repeat("b", totalAnnotationSizeLimitB)},
		{
			"a": strings.Repeat("b", totalAnnotationSizeLimitB/2),
			"c": strings.Repeat("d", totalAnnotationSizeLimitB/2),
		},
	}
	for i := range totalSizeErrorCases {
		errs := ValidateAnnotations(totalSizeErrorCases[i], field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		}
	}
}
