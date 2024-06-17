/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateLabels(t *testing.T) {
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
		{"UpperCaseAreOK123": "bar"},
		{"goodvalue": "123_-.BaR"},
	}
	for i := range successCases {
		errs := ValidateLabels(successCases[i], field.NewPath("field"))
		if len(errs) != 0 {
			t.Errorf("case[%d] expected success, got %#v", i, errs)
		}
	}

	namePartErrMsg := "name part must consist of"
	nameErrMsg := "a qualified name must consist of"
	labelErrMsg := "a valid label must be an empty string or consist of"
	maxLengthErrMsg := "must be no more than"

	labelNameErrorCases := []struct {
		labels map[string]string
		expect string
	}{
		{map[string]string{"nospecialchars^=@": "bar"}, namePartErrMsg},
		{map[string]string{"cantendwithadash-": "bar"}, namePartErrMsg},
		{map[string]string{"only/one/slash": "bar"}, nameErrMsg},
		{map[string]string{strings.Repeat("a", 254): "bar"}, maxLengthErrMsg},
	}
	for i := range labelNameErrorCases {
		errs := ValidateLabels(labelNameErrorCases[i].labels, field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d]: expected failure", i)
		} else {
			if !strings.Contains(errs[0].Detail, labelNameErrorCases[i].expect) {
				t.Errorf("case[%d]: error details do not include %q: %q", i, labelNameErrorCases[i].expect, errs[0].Detail)
			}
		}
	}

	labelValueErrorCases := []struct {
		labels map[string]string
		expect string
	}{
		{map[string]string{"toolongvalue": strings.Repeat("a", 64)}, maxLengthErrMsg},
		{map[string]string{"backslashesinvalue": "some\\bad\\value"}, labelErrMsg},
		{map[string]string{"nocommasallowed": "bad,value"}, labelErrMsg},
		{map[string]string{"strangecharsinvalue": "?#$notsogood"}, labelErrMsg},
	}
	for i := range labelValueErrorCases {
		errs := ValidateLabels(labelValueErrorCases[i].labels, field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d]: expected failure", i)
		} else {
			if !strings.Contains(errs[0].Detail, labelValueErrorCases[i].expect) {
				t.Errorf("case[%d]: error details do not include %q: %q", i, labelValueErrorCases[i].expect, errs[0].Detail)
			}
		}
	}
}

func TestValidDryRun(t *testing.T) {
	tests := [][]string{
		{},
		{"All"},
		{"All", "All"},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%v", test), func(t *testing.T) {
			if errs := ValidateDryRun(field.NewPath("dryRun"), test); len(errs) != 0 {
				t.Errorf("%v should be a valid dry-run value: %v", test, errs)
			}
		})
	}
}

func TestInvalidDryRun(t *testing.T) {
	tests := [][]string{
		{"False"},
		{"All", "False"},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%v", test), func(t *testing.T) {
			if len(ValidateDryRun(field.NewPath("dryRun"), test)) == 0 {
				t.Errorf("%v shouldn't be a valid dry-run value", test)
			}
		})
	}

}

func boolPtr(b bool) *bool {
	return &b
}

func TestValidPatchOptions(t *testing.T) {
	tests := []struct {
		opts      metav1.PatchOptions
		patchType types.PatchType
	}{{
		opts: metav1.PatchOptions{
			Force:        boolPtr(true),
			FieldManager: "kubectl",
		},
		patchType: types.ApplyPatchType,
	}, {
		opts: metav1.PatchOptions{
			FieldManager: "kubectl",
		},
		patchType: types.ApplyPatchType,
	}, {
		opts:      metav1.PatchOptions{},
		patchType: types.MergePatchType,
	}, {
		opts: metav1.PatchOptions{
			FieldManager: "patcher",
		},
		patchType: types.MergePatchType,
	}}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%v", test.opts), func(t *testing.T) {
			errs := ValidatePatchOptions(&test.opts, test.patchType)
			if len(errs) != 0 {
				t.Fatalf("Expected no failures, got: %v", errs)
			}
		})
	}
}

func TestInvalidPatchOptions(t *testing.T) {
	tests := []struct {
		opts      metav1.PatchOptions
		patchType types.PatchType
	}{
		// missing manager
		{
			opts:      metav1.PatchOptions{},
			patchType: types.ApplyPatchType,
		},
		// force on non-apply
		{
			opts: metav1.PatchOptions{
				Force: boolPtr(true),
			},
			patchType: types.MergePatchType,
		},
		// manager and force on non-apply
		{
			opts: metav1.PatchOptions{
				FieldManager: "kubectl",
				Force:        boolPtr(false),
			},
			patchType: types.MergePatchType,
		},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%v", test.opts), func(t *testing.T) {
			errs := ValidatePatchOptions(&test.opts, test.patchType)
			if len(errs) == 0 {
				t.Fatal("Expected failures, got none.")
			}
		})
	}
}

func TestValidateFieldManagerValid(t *testing.T) {
	tests := []string{
		"filedManager",
		"你好", // Hello
		"🍔",
	}

	for _, test := range tests {
		t.Run(test, func(t *testing.T) {
			errs := ValidateFieldManager(test, field.NewPath("fieldManager"))
			if len(errs) != 0 {
				t.Errorf("Validation failed: %v", errs)
			}
		})
	}
}

func TestValidateFieldManagerInvalid(t *testing.T) {
	tests := []string{
		"field\nmanager", // Contains invalid character \n
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", // Has 129 chars
	}

	for _, test := range tests {
		t.Run(test, func(t *testing.T) {
			errs := ValidateFieldManager(test, field.NewPath("fieldManager"))
			if len(errs) == 0 {
				t.Errorf("Validation should have failed")
			}
		})
	}
}

func TestValidateManagedFieldsInvalid(t *testing.T) {
	tests := []metav1.ManagedFieldsEntry{{
		Operation:  metav1.ManagedFieldsOperationUpdate,
		FieldsType: "RandomVersion",
		APIVersion: "v1",
	}, {
		Operation:  "RandomOperation",
		FieldsType: "FieldsV1",
		APIVersion: "v1",
	}, {
		// Operation is missing
		FieldsType: "FieldsV1",
		APIVersion: "v1",
	}, {
		Operation:  metav1.ManagedFieldsOperationUpdate,
		FieldsType: "FieldsV1",
		// Invalid fieldManager
		Manager:    "field\nmanager",
		APIVersion: "v1",
	}, {
		Operation:   metav1.ManagedFieldsOperationApply,
		FieldsType:  "FieldsV1",
		APIVersion:  "v1",
		Subresource: "TooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLongTooLong",
	}}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v", test), func(t *testing.T) {
			errs := ValidateManagedFields([]metav1.ManagedFieldsEntry{test}, field.NewPath("managedFields"))
			if len(errs) == 0 {
				t.Errorf("Validation should have failed")
			}
		})
	}
}

func TestValidateMangedFieldsValid(t *testing.T) {
	tests := []metav1.ManagedFieldsEntry{{
		Operation:  metav1.ManagedFieldsOperationUpdate,
		APIVersion: "v1",
		// FieldsType is missing
	}, {
		Operation:  metav1.ManagedFieldsOperationUpdate,
		FieldsType: "FieldsV1",
		APIVersion: "v1",
	}, {
		Operation:   metav1.ManagedFieldsOperationApply,
		FieldsType:  "FieldsV1",
		APIVersion:  "v1",
		Subresource: "scale",
	}, {
		Operation:  metav1.ManagedFieldsOperationApply,
		FieldsType: "FieldsV1",
		APIVersion: "v1",
		Manager:    "🍔",
	}}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v", test), func(t *testing.T) {
			err := ValidateManagedFields([]metav1.ManagedFieldsEntry{test}, field.NewPath("managedFields"))
			if err != nil {
				t.Errorf("Validation failed: %v", err)
			}
		})
	}
}

func TestValidateConditions(t *testing.T) {
	tests := []struct {
		name         string
		conditions   []metav1.Condition
		validateErrs func(t *testing.T, errs field.ErrorList)
	}{{
		name: "bunch-of-invalid-fields",
		conditions: []metav1.Condition{{
			Type:               ":invalid",
			Status:             "unknown",
			ObservedGeneration: -1,
			LastTransitionTime: metav1.Time{},
			Reason:             "invalid;val",
			Message:            "",
		}},
		validateErrs: func(t *testing.T, errs field.ErrorList) {
			needle := `status.conditions[0].type: Invalid value: ":invalid": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`
			if !hasError(errs, needle) {
				t.Errorf("missing %q in\n%v", needle, errorsAsString(errs))
			}
			needle = `status.conditions[0].status: Unsupported value: "unknown": supported values: "False", "True", "Unknown"`
			if !hasError(errs, needle) {
				t.Errorf("missing %q in\n%v", needle, errorsAsString(errs))
			}
			needle = `status.conditions[0].observedGeneration: Invalid value: -1: must be greater than or equal to zero`
			if !hasError(errs, needle) {
				t.Errorf("missing %q in\n%v", needle, errorsAsString(errs))
			}
			needle = `status.conditions[0].lastTransitionTime: Required value: must be set`
			if !hasError(errs, needle) {
				t.Errorf("missing %q in\n%v", needle, errorsAsString(errs))
			}
			needle = `status.conditions[0].reason: Invalid value: "invalid;val": a condition reason must start with alphabetic character, optionally followed by a string of alphanumeric characters or '_,:', and must end with an alphanumeric character or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName',  or 'ReasonA,ReasonB',  or 'ReasonA:ReasonB', regex used for validation is '[A-Za-z]([A-Za-z0-9_,:]*[A-Za-z0-9_])?')`
			if !hasError(errs, needle) {
				t.Errorf("missing %q in\n%v", needle, errorsAsString(errs))
			}
		},
	}, {
		name: "duplicates",
		conditions: []metav1.Condition{{
			Type: "First",
		}, {
			Type: "Second",
		}, {
			Type: "First",
		}},
		validateErrs: func(t *testing.T, errs field.ErrorList) {
			needle := `status.conditions[2].type: Duplicate value: "First"`
			if !hasError(errs, needle) {
				t.Errorf("missing %q in\n%v", needle, errorsAsString(errs))
			}
		},
	}, {
		name: "colon-allowed-in-reason",
		conditions: []metav1.Condition{{
			Type:   "First",
			Reason: "valid:val",
		}},
		validateErrs: func(t *testing.T, errs field.ErrorList) {
			needle := `status.conditions[0].reason`
			if hasPrefixError(errs, needle) {
				t.Errorf("has %q in\n%v", needle, errorsAsString(errs))
			}
		},
	}, {
		name: "comma-allowed-in-reason",
		conditions: []metav1.Condition{{
			Type:   "First",
			Reason: "valid,val",
		}},
		validateErrs: func(t *testing.T, errs field.ErrorList) {
			needle := `status.conditions[0].reason`
			if hasPrefixError(errs, needle) {
				t.Errorf("has %q in\n%v", needle, errorsAsString(errs))
			}
		},
	}, {
		name: "reason-does-not-end-in-delimiter",
		conditions: []metav1.Condition{{
			Type:   "First",
			Reason: "valid,val:",
		}},
		validateErrs: func(t *testing.T, errs field.ErrorList) {
			needle := `status.conditions[0].reason: Invalid value: "valid,val:": a condition reason must start with alphabetic character, optionally followed by a string of alphanumeric characters or '_,:', and must end with an alphanumeric character or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName',  or 'ReasonA,ReasonB',  or 'ReasonA:ReasonB', regex used for validation is '[A-Za-z]([A-Za-z0-9_,:]*[A-Za-z0-9_])?')`
			if !hasError(errs, needle) {
				t.Errorf("missing %q in\n%v", needle, errorsAsString(errs))
			}
		},
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateConditions(test.conditions, field.NewPath("status").Child("conditions"))
			test.validateErrs(t, errs)
		})
	}
}

func TestLabelSelectorMatchExpression(t *testing.T) {
	testCases := []struct {
		name            string
		labelSelector   *metav1.LabelSelector
		wantErrorNumber int
		validateErrs    func(t *testing.T, errs field.ErrorList)
	}{{
		name: "Valid LabelSelector",
		labelSelector: &metav1.LabelSelector{
			MatchExpressions: []metav1.LabelSelectorRequirement{{
				Key:      "key",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"value"},
			}},
		},
		wantErrorNumber: 0,
		validateErrs:    nil,
	}, {
		name: "MatchExpression's key name isn't valid",
		labelSelector: &metav1.LabelSelector{
			MatchExpressions: []metav1.LabelSelectorRequirement{{
				Key:      "-key",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"value"},
			}},
		},
		wantErrorNumber: 1,
		validateErrs: func(t *testing.T, errs field.ErrorList) {
			errMessage := "name part must consist of alphanumeric characters"
			if !partStringInErrorMessage(errs, errMessage) {
				t.Errorf("missing %q in\n%v", errMessage, errorsAsString(errs))
			}
		},
	}, {
		name: "MatchExpression's operator isn't valid",
		labelSelector: &metav1.LabelSelector{
			MatchExpressions: []metav1.LabelSelectorRequirement{{
				Key:      "key",
				Operator: "abc",
				Values:   []string{"value"},
			}},
		},
		wantErrorNumber: 1,
		validateErrs: func(t *testing.T, errs field.ErrorList) {
			errMessage := "not a valid selector operator"
			if !partStringInErrorMessage(errs, errMessage) {
				t.Errorf("missing %q in\n%v", errMessage, errorsAsString(errs))
			}
		},
	}, {
		name: "MatchExpression's value name isn't valid",
		labelSelector: &metav1.LabelSelector{
			MatchExpressions: []metav1.LabelSelectorRequirement{{
				Key:      "key",
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{"-value"},
			}},
		},
		wantErrorNumber: 1,
		validateErrs: func(t *testing.T, errs field.ErrorList) {
			errMessage := "a valid label must be an empty string or consist of"
			if !partStringInErrorMessage(errs, errMessage) {
				t.Errorf("missing %q in\n%v", errMessage, errorsAsString(errs))
			}
		},
	}}
	for index, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			allErrs := ValidateLabelSelector(testCase.labelSelector, LabelSelectorValidationOptions{false}, field.NewPath("labelSelector"))
			if len(allErrs) != testCase.wantErrorNumber {
				t.Errorf("case[%d]: expected failure", index)
			}
			if len(allErrs) >= 1 && testCase.validateErrs != nil {
				testCase.validateErrs(t, allErrs)
			}
		})
	}
}

func hasError(errs field.ErrorList, needle string) bool {
	for _, curr := range errs {
		if curr.Error() == needle {
			return true
		}
	}
	return false
}

func hasPrefixError(errs field.ErrorList, prefix string) bool {
	for _, curr := range errs {
		if strings.HasPrefix(curr.Error(), prefix) {
			return true
		}
	}
	return false
}

func partStringInErrorMessage(errs field.ErrorList, prefix string) bool {
	for _, curr := range errs {
		if strings.Contains(curr.Error(), prefix) {
			return true
		}
	}
	return false
}

func errorsAsString(errs field.ErrorList) string {
	messages := []string{}
	for _, curr := range errs {
		messages = append(messages, curr.Error())
	}
	return strings.Join(messages, "\n")
}
