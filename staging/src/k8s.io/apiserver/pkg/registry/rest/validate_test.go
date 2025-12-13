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

package rest

import (
	"bytes"
	"context"
	"fmt"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/validation"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
)

func TestValidateDeclaratively(t *testing.T) {
	valid := &Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
		},
	}

	invalidRestartPolicy := &Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
		},
		RestartPolicy: "INVALID",
	}

	invalidRestartPolicyErr := field.Invalid(field.NewPath("spec", "restartPolicy"), "", "Invalid value").WithOrigin("invalid-test")
	mutatedRestartPolicyErr := field.Invalid(field.NewPath("spec", "restartPolicy"), "", "Immutable field").WithOrigin("immutable-test")
	invalidStatusErr := field.Invalid(field.NewPath("status", "conditions"), "", "Invalid condition").WithOrigin("invalid-condition")
	invalidIfOptionErr := field.Invalid(field.NewPath("spec", "restartPolicy"), "", "Invalid when option is set").WithOrigin("invalid-when-option-set")
	invalidSubresourceErr := field.InternalError(nil, fmt.Errorf("unexpected error parsing subresource path: %w", fmt.Errorf("invalid subresource path: %s", "invalid/status")))

	testCases := []struct {
		name        string
		object      runtime.Object
		oldObject   runtime.Object
		subresource string
		options     []string
		expected    field.ErrorList
	}{
		{
			name:     "create",
			object:   invalidRestartPolicy,
			expected: field.ErrorList{invalidRestartPolicyErr},
		},
		{
			name:      "update",
			object:    invalidRestartPolicy,
			oldObject: valid,
			expected:  field.ErrorList{invalidRestartPolicyErr, mutatedRestartPolicyErr},
		},
		{
			name:        "update subresource with declarative validation",
			subresource: "status",
			object:      valid,
			oldObject:   valid,
			expected:    field.ErrorList{invalidStatusErr},
		},
		{
			name:        "update subresource without declarative validation",
			subresource: "scale",
			object:      valid,
			oldObject:   valid,
			expected:    field.ErrorList{}, // Expect no errors if there is no registered validation
		},
		{
			name:        "invalid subresource",
			subresource: "/invalid/status",
			object:      valid,
			oldObject:   valid,
			expected:    field.ErrorList{invalidSubresourceErr},
		},
		{
			name:     "update with option",
			options:  []string{"option1"},
			object:   valid,
			expected: field.ErrorList{invalidIfOptionErr},
		},
	}

	ctx := context.Background()

	internalGV := schema.GroupVersion{Group: "", Version: runtime.APIVersionInternal}
	v1GV := schema.GroupVersion{Group: "", Version: "v1"}

	scheme := runtime.NewScheme()
	scheme.AddKnownTypes(internalGV, &Pod{})
	scheme.AddKnownTypes(v1GV, &v1.Pod{})

	scheme.AddValidationFunc(&v1.Pod{}, func(ctx context.Context, op operation.Operation, object, oldObject any) field.ErrorList {
		results := field.ErrorList{}
		if op.HasOption("option1") {
			results = append(results, invalidIfOptionErr)
		}
		if slices.Equal(op.Request.Subresources, []string{"status"}) {
			results = append(results, invalidStatusErr)
		}
		if op.Type == operation.Update && object.(*v1.Pod).Spec.RestartPolicy != oldObject.(*v1.Pod).Spec.RestartPolicy {
			results = append(results, mutatedRestartPolicyErr)
		}
		if object.(*v1.Pod).Spec.RestartPolicy == "INVALID" {
			results = append(results, invalidRestartPolicyErr)
		}
		return results
	})
	err := scheme.AddConversionFunc(&Pod{}, &v1.Pod{}, func(a, b interface{}, scope conversion.Scope) error {
		if in, ok := a.(*Pod); ok {
			if out, ok := b.(*v1.Pod); ok {
				out.APIVersion = in.APIVersion
				out.Kind = in.Kind
				out.Spec.RestartPolicy = v1.RestartPolicy(in.RestartPolicy)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range testCases {
		ctx = genericapirequest.WithRequestInfo(ctx, &genericapirequest.RequestInfo{
			APIGroup:    "",
			APIVersion:  "v1",
			Subresource: tc.subresource,
		})
		t.Run(tc.name, func(t *testing.T) {

			cfg := &validationConfigOption{
				options: tc.options,
			}
			if tc.oldObject == nil {
				cfg.opType = operation.Create
			} else {
				cfg.opType = operation.Update
			}
			results := panicSafeValidateFunc(validateDeclaratively, cfg.takeover, cfg.validationIdentifier)(ctx, scheme, tc.object, tc.oldObject, cfg)
			matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
			matcher.Test(t, tc.expected, results)
		})
	}
}

// Fake internal pod type, since core.Pod cannot be imported by this package
type Pod struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	RestartPolicy     string `json:"restartPolicy"`
}

func (Pod) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }

func (p Pod) DeepCopyObject() runtime.Object {
	return &Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: p.APIVersion,
			Kind:       p.Kind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      p.Name,
			Namespace: p.Namespace,
		},
		RestartPolicy: p.RestartPolicy,
	}
}

// TestGatherDeclarativeValidationMismatches tests all mismatch
// scenarios across imperative and declarative errors for
// the gatherDeclarativeValidationMismatches function
func TestGatherDeclarativeValidationMismatches(t *testing.T) {
	replicasPath := field.NewPath("spec").Child("replicas")
	minReadySecondsPath := field.NewPath("spec").Child("minReadySeconds")
	selectorPath := field.NewPath("spec").Child("selector")

	errA := field.Invalid(replicasPath, nil, "regular error A")
	errB := field.Invalid(minReadySecondsPath, -1, "covered error B").WithOrigin("minimum")
	coveredErrB := field.Invalid(minReadySecondsPath, -1, "covered error B").WithOrigin("minimum")
	errBWithDiffDetail := field.Invalid(minReadySecondsPath, -1, "covered error B - different detail").WithOrigin("minimum")
	errBWithDiffPath := field.Invalid(field.NewPath("spec").Child("fakeminReadySeconds"), -1, "covered error B").WithOrigin("minimum")
	coveredErrB.CoveredByDeclarative = true
	errC := field.Invalid(replicasPath, nil, "covered error C").WithOrigin("minimum")
	coveredErrC := field.Invalid(replicasPath, nil, "covered error C").WithOrigin("minimum")
	coveredErrC.CoveredByDeclarative = true
	errCWithDiffOrigin := field.Invalid(replicasPath, nil, "covered error C").WithOrigin("maximum")
	errD := field.Invalid(selectorPath, nil, "regular error D")

	testCases := []struct {
		name                    string
		imperativeErrors        field.ErrorList
		declarativeErrors       field.ErrorList
		takeover                bool
		expectMismatches        bool
		expectDetailsContaining []string
		normalizedRules         []field.NormalizationRule
	}{
		{
			name:                    "Declarative and imperative return 0 errors - no mismatch",
			imperativeErrors:        field.ErrorList{},
			declarativeErrors:       field.ErrorList{},
			takeover:                false,
			expectMismatches:        false,
			expectDetailsContaining: []string{},
		},
		{
			name: "Declarative returns multiple errors with different origins, errors match - no mismatch",
			imperativeErrors: field.ErrorList{
				errA,
				coveredErrB,
				coveredErrC,
				errD,
			},
			declarativeErrors: field.ErrorList{
				errB,
				errC,
			},
			takeover:                false,
			expectMismatches:        false,
			expectDetailsContaining: []string{},
		},
		{
			name: "Declarative returns multiple errors with different origins, errors don't match - mismatch case",
			imperativeErrors: field.ErrorList{
				errA,
				coveredErrB,
				coveredErrC,
			},
			declarativeErrors: field.ErrorList{
				errB,
				errCWithDiffOrigin,
			},
			takeover:         true,
			expectMismatches: true,
			expectDetailsContaining: []string{
				"Unexpected difference between hand written validation and declarative validation error results",
				"unmatched error(s) found",
				"extra error(s) found",
				"replicas",
				"Consider disabling the DeclarativeValidationTakeover feature gate to keep data persisted in etcd consistent with prior versions of Kubernetes",
			},
		},
		{
			name: "Declarative and imperative return exactly 1 error, errors match - no mismatch",
			imperativeErrors: field.ErrorList{
				coveredErrB,
			},
			declarativeErrors: field.ErrorList{
				errB,
			},
			takeover:                false,
			expectMismatches:        false,
			expectDetailsContaining: []string{},
		},
		{
			name: "Declarative and imperative exactly 1 error, errors don't match - mismatch",
			imperativeErrors: field.ErrorList{
				coveredErrB,
			},
			declarativeErrors: field.ErrorList{
				errC,
			},
			takeover:         false,
			expectMismatches: true,
			expectDetailsContaining: []string{
				"Unexpected difference between hand written validation and declarative validation error results",
				"unmatched error(s) found",
				"minReadySeconds",
				"extra error(s) found",
				"replicas",
				"This difference should not affect system operation since hand written validation is authoritative",
			},
		},
		{
			name: "Declarative returns 0 errors, imperative returns 1 covered error - mismatch",
			imperativeErrors: field.ErrorList{
				coveredErrB,
			},
			declarativeErrors: field.ErrorList{},
			takeover:          true,
			expectMismatches:  true,
			expectDetailsContaining: []string{
				"Unexpected difference between hand written validation and declarative validation error results",
				"unmatched error(s) found",
				"minReadySeconds",
				"Consider disabling the DeclarativeValidationTakeover feature gate to keep data persisted in etcd consistent with prior versions of Kubernetes",
			},
		},
		{
			name: "Declarative returns 0 errors, imperative returns 1 uncovered error - no mismatch",
			imperativeErrors: field.ErrorList{
				errB,
			},
			declarativeErrors:       field.ErrorList{},
			takeover:                false,
			expectMismatches:        false,
			expectDetailsContaining: []string{},
		},
		{
			name:             "Declarative returns 1 error, imperative returns 0 error - mismatch",
			imperativeErrors: field.ErrorList{},
			declarativeErrors: field.ErrorList{
				errB,
			},
			takeover:         false,
			expectMismatches: true,
			expectDetailsContaining: []string{
				"Unexpected difference between hand written validation and declarative validation error results",
				"extra error(s) found",
				"minReadySeconds",
				"This difference should not affect system operation since hand written validation is authoritative",
			},
		},
		{
			name: "Declarative returns 1 error, imperative returns 3 matching errors  - no mismatch",
			imperativeErrors: field.ErrorList{
				coveredErrB,
			},
			declarativeErrors: field.ErrorList{
				errB,
				errB,
				errBWithDiffDetail,
			},
			takeover:                false,
			expectMismatches:        false,
			expectDetailsContaining: []string{},
		},
		{
			name: "Field normalization, errors don't match - mismatch",
			imperativeErrors: field.ErrorList{
				coveredErrB,
			},
			declarativeErrors: field.ErrorList{
				errBWithDiffPath,
			},
			normalizedRules: []field.NormalizationRule{
				{
					Regexp:      regexp.MustCompile(`spec.fakeminReadySeconds`),
					Replacement: "spec.minReadySeconds",
				},
			},
			takeover:                false,
			expectMismatches:        false,
			expectDetailsContaining: []string{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			details := gatherDeclarativeValidationMismatches(tc.imperativeErrors, tc.declarativeErrors, tc.takeover, tc.normalizedRules)
			// Check if mismatches were found if expected
			if tc.expectMismatches && len(details) == 0 {
				t.Errorf("Expected mismatches but got none")
			}
			// Check if details contain expected text
			detailsStr := strings.Join(details, " ")
			for _, expectedContent := range tc.expectDetailsContaining {
				if !strings.Contains(detailsStr, expectedContent) {
					t.Errorf("Expected details to contain: %q, but they didn't.\nDetails were:\n%s",
						expectedContent, strings.Join(details, "\n"))
				}
			}
			// If we don't expect any details, make sure none provided
			if len(tc.expectDetailsContaining) == 0 && len(details) > 0 {
				t.Errorf("Expected no details, but got %d details: %v", len(details), details)
			}
		})
	}
}

// TestCompareDeclarativeErrorsAndEmitMismatches tests expected
// logging of mismatch information given match & mismatch error conditions.
func TestCompareDeclarativeErrorsAndEmitMismatches(t *testing.T) {
	replicasPath := field.NewPath("spec").Child("replicas")
	minReadySecondsPath := field.NewPath("spec").Child("minReadySeconds")

	errA := field.Invalid(replicasPath, nil, "regular error A")
	errB := field.Invalid(minReadySecondsPath, -1, "covered error B").WithOrigin("minimum")
	coveredErrB := field.Invalid(minReadySecondsPath, -1, "covered error B").WithOrigin("minimum")
	coveredErrB.CoveredByDeclarative = true

	testCases := []struct {
		name            string
		imperativeErrs  field.ErrorList
		declarativeErrs field.ErrorList
		takeover        bool
		expectLogs      bool
		expectedRegex   string
	}{
		{
			name:            "mismatched errors, log info",
			imperativeErrs:  field.ErrorList{coveredErrB},
			declarativeErrs: field.ErrorList{errA},
			takeover:        true,
			expectLogs:      true,
			// logs have a prefix of the form - E0309 21:05:33.865030 1926106 validate.go:199]
			expectedRegex: "E.*Unexpected difference between hand written validation and declarative validation error results.*Consider disabling the DeclarativeValidationTakeover feature gate to keep data persisted in etcd consistent with prior versions of Kubernetes",
		},
		{
			name:            "matching errors, don't log info",
			imperativeErrs:  field.ErrorList{coveredErrB},
			declarativeErrs: field.ErrorList{errB},
			takeover:        true,
			expectLogs:      false,
			expectedRegex:   "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			klog.LogToStderr(false)
			defer klog.LogToStderr(true)
			ctx := context.Background()

			compareDeclarativeErrorsAndEmitMismatches(ctx, tc.imperativeErrs, tc.declarativeErrs, tc.takeover, "test_validationIdentifier", nil)

			klog.Flush()
			logOutput := buf.String()

			if tc.expectLogs {
				matched, err := regexp.MatchString(tc.expectedRegex, logOutput)
				if err != nil {
					t.Fatalf("Bad regex: %v", err)
				}
				if !matched {
					t.Errorf("Expected log output to match %q, but got:\n%s", tc.expectedRegex, logOutput)
				}
			} else if len(logOutput) > 0 {
				t.Errorf("Expected no mismatch logs, but found: %s", logOutput)
			}
		})
	}
}

func TestWithRecover(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	var options []string
	obj := &runtime.Unknown{}

	testCases := []struct {
		name            string
		validateFn      func(context.Context, *runtime.Scheme, runtime.Object, runtime.Object, *validationConfigOption) field.ErrorList
		takeoverEnabled bool
		wantErrs        field.ErrorList
		expectLogRegex  string
	}{
		{
			name: "no panic",
			validateFn: func(context.Context, *runtime.Scheme, runtime.Object, runtime.Object, *validationConfigOption) field.ErrorList {
				return field.ErrorList{
					field.Invalid(field.NewPath("field"), "value", "reason"),
				}
			},
			takeoverEnabled: false,
			wantErrs: field.ErrorList{
				field.Invalid(field.NewPath("field"), "value", "reason"),
			},
			expectLogRegex: "",
		},
		{
			name: "panic with takeover disabled",
			validateFn: func(context.Context, *runtime.Scheme, runtime.Object, runtime.Object, *validationConfigOption) field.ErrorList {
				panic("test panic")
			},
			takeoverEnabled: false,
			wantErrs:        nil,
			// logs have a prefix of the form - E0309 21:05:33.865030 1926106 validate.go:199]
			expectLogRegex: "E.*panic during declarative validation: test panic",
		},
		{
			name: "panic with takeover enabled",
			validateFn: func(context.Context, *runtime.Scheme, runtime.Object, runtime.Object, *validationConfigOption) field.ErrorList {
				panic("test panic")
			},
			takeoverEnabled: true,
			wantErrs: field.ErrorList{
				field.InternalError(nil, fmt.Errorf("panic during declarative validation: test panic")),
			},
			expectLogRegex: "",
		},
		{
			name: "nil return, no panic",
			validateFn: func(context.Context, *runtime.Scheme, runtime.Object, runtime.Object, *validationConfigOption) field.ErrorList {
				return nil
			},
			takeoverEnabled: false,
			wantErrs:        nil,
			expectLogRegex:  "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			klog.LogToStderr(false)
			defer klog.LogToStderr(true)

			// Pass the takeover flag to panicSafeValidateFunc instead of relying on the feature gate
			wrapped := panicSafeValidateFunc(tc.validateFn, tc.takeoverEnabled, "test_validationIdentifier")
			gotErrs := wrapped(ctx, scheme, obj, nil, &validationConfigOption{opType: operation.Create, options: options, takeover: tc.takeoverEnabled})

			klog.Flush()
			logOutput := buf.String()

			// Compare gotErrs vs. tc.wantErrs
			if !equalErrorLists(gotErrs, tc.wantErrs) {
				t.Errorf("panicSafeValidateFunc() gotErrs = %#v, want %#v", gotErrs, tc.wantErrs)
			}

			// Check logs if needed
			if tc.expectLogRegex != "" {
				matched, err := regexp.MatchString(tc.expectLogRegex, logOutput)
				if err != nil {
					t.Fatalf("Bad regex: %v", err)
				}
				if !matched {
					t.Errorf("Expected log output %q, but got:\n%s", tc.expectLogRegex, logOutput)
				}
			} else if strings.Contains(logOutput, "panic during declarative validation") {
				t.Errorf("Unexpected panic log found: %s", logOutput)
			}
		})
	}
}

func TestWithRecoverUpdate(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	var options []string
	obj := &runtime.Unknown{}
	oldObj := &runtime.Unknown{}

	testCases := []struct {
		name            string
		validateFn      func(context.Context, *runtime.Scheme, runtime.Object, runtime.Object, *validationConfigOption) field.ErrorList
		takeoverEnabled bool
		wantErrs        field.ErrorList
		expectLogRegex  string
	}{
		{
			name: "no panic",
			validateFn: func(context.Context, *runtime.Scheme, runtime.Object, runtime.Object, *validationConfigOption) field.ErrorList {
				return field.ErrorList{
					field.Invalid(field.NewPath("field"), "value", "reason"),
				}
			},
			takeoverEnabled: false,
			wantErrs: field.ErrorList{
				field.Invalid(field.NewPath("field"), "value", "reason"),
			},
			expectLogRegex: "",
		},
		{
			name: "panic with takeover disabled",
			validateFn: func(context.Context, *runtime.Scheme, runtime.Object, runtime.Object, *validationConfigOption) field.ErrorList {
				panic("test update panic")
			},
			takeoverEnabled: false,
			wantErrs:        nil,
			// logs have a prefix of the form - E0309 21:05:33.865030 1926106 validate.go:199]
			expectLogRegex: "E.*panic during declarative validation: test update panic",
		},
		{
			name: "panic with takeover enabled",
			validateFn: func(context.Context, *runtime.Scheme, runtime.Object, runtime.Object, *validationConfigOption) field.ErrorList {
				panic("test update panic")
			},
			takeoverEnabled: true,
			wantErrs: field.ErrorList{
				field.InternalError(nil, fmt.Errorf("panic during declarative validation: test update panic")),
			},
			expectLogRegex: "",
		},
		{
			name: "nil return, no panic",
			validateFn: func(context.Context, *runtime.Scheme, runtime.Object, runtime.Object, *validationConfigOption) field.ErrorList {
				return nil
			},
			takeoverEnabled: false,
			wantErrs:        nil,
			expectLogRegex:  "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			klog.LogToStderr(false)
			defer klog.LogToStderr(true)

			// Pass the takeover flag to panicSafeValidateUpdateFunc instead of relying on the feature gate
			wrapped := panicSafeValidateFunc(tc.validateFn, tc.takeoverEnabled, "test_validationIdentifier")
			gotErrs := wrapped(ctx, scheme, obj, oldObj, &validationConfigOption{opType: operation.Update, options: options, takeover: tc.takeoverEnabled})

			klog.Flush()
			logOutput := buf.String()

			// Compare gotErrs with wantErrs
			if !equalErrorLists(gotErrs, tc.wantErrs) {
				t.Errorf("panicSafeValidateUpdateFunc() gotErrs = %#v, want %#v", gotErrs, tc.wantErrs)
			}

			// Verify log output
			if tc.expectLogRegex != "" {
				matched, err := regexp.MatchString(tc.expectLogRegex, logOutput)
				if err != nil {
					t.Fatalf("Bad regex: %v", err)
				}
				if !matched {
					t.Errorf("Expected log pattern %q, but got:\n%s", tc.expectLogRegex, logOutput)
				}
			} else if strings.Contains(logOutput, "panic during declarative validation") {
				t.Errorf("Unexpected panic log found: %s", logOutput)
			}
		})
	}
}

func TestRecordDuplicateValidationErrors(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name           string
		qualifiedKind  schema.GroupKind
		errs           field.ErrorList
		expectedMetric string
	}{
		{
			name:          "detect duplicates and increment metric",
			qualifiedKind: schema.GroupKind{Group: "apps", Kind: "ReplicaSet"},
			errs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("replicas"), -1, "must be greater than or equal to 0").WithOrigin("minimum"),
				field.Invalid(field.NewPath("spec").Child("replicas"), -1, "must be greater than or equal to 0").WithOrigin("minimum"),
				field.Invalid(field.NewPath("spec").Child("selector"), &metav1.LabelSelector{MatchLabels: map[string]string{}, MatchExpressions: []metav1.LabelSelectorRequirement{}}, "empty selector is invalid for deployment"),
				field.Invalid(field.NewPath("spec").Child("selector"), &metav1.LabelSelector{MatchLabels: map[string]string{}, MatchExpressions: []metav1.LabelSelectorRequirement{}}, "empty selector is invalid for deployment"),
			},
			expectedMetric: `
			# HELP apiserver_validation_duplicate_validation_error_total [INTERNAL] Number of duplicate validation errors during validation.
			# TYPE apiserver_validation_duplicate_validation_error_total counter
			apiserver_validation_duplicate_validation_error_total 2
			`,
		},
		{
			name:          "detect duplicates with all fields but origin being equal",
			qualifiedKind: schema.GroupKind{Group: "apps", Kind: "ReplicaSet"},
			errs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("replicas"), -1, "must be greater than or equal to 0").WithOrigin("minimum"),
				field.Invalid(field.NewPath("spec").Child("replicas"), -1, "must be greater than or equal to 0").WithOrigin("min"),
			},
			expectedMetric: `
			# HELP apiserver_validation_duplicate_validation_error_total [INTERNAL] Number of duplicate validation errors during validation.
			# TYPE apiserver_validation_duplicate_validation_error_total counter
			apiserver_validation_duplicate_validation_error_total 1
			`,
		},
		{
			name:          "no duplicates",
			qualifiedKind: schema.GroupKind{Group: "apps", Kind: "ReplicaSet"},
			errs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("replicas"), -1, "must be greater than or equal to 0").WithOrigin("minimum"),
				field.Invalid(field.NewPath("spec").Child("selector"), &metav1.LabelSelector{MatchLabels: map[string]string{}, MatchExpressions: []metav1.LabelSelectorRequirement{}}, "empty selector is invalid for deployment"),
			},
			expectedMetric: `
			# HELP apiserver_validation_duplicate_validation_error_total [INTERNAL] Number of duplicate validation errors during validation.
			# TYPE apiserver_validation_duplicate_validation_error_total counter
			apiserver_validation_duplicate_validation_error_total 0
			`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer legacyregistry.Reset()
			defer validation.ResetValidationMetricsInstance()
			RecordDuplicateValidationErrors(ctx, tc.qualifiedKind, tc.errs)

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.expectedMetric), "apiserver_validation_duplicate_validation_error_total"); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func equalErrorLists(a, b field.ErrorList) bool {
	// If both are nil, consider them equal
	if a == nil && b == nil {
		return true
	}
	// If one is nil and the other not, they're different
	if (a == nil && b != nil) || (a != nil && b == nil) {
		return false
	}
	// Both non-nil: do a normal DeepEqual
	return reflect.DeepEqual(a, b)
}

func TestMetricIdentifier(t *testing.T) {
	scheme := runtime.NewScheme()
	scheme.AddKnownTypes(schema.GroupVersion{Version: "v1"}, &v1.Pod{})

	testCases := []struct {
		name        string
		opType      operation.Type
		obj         runtime.Object
		scheme      *runtime.Scheme
		subresource string
		expected    string
		expectErr   bool
	}{
		{
			name:        "with subresource",
			opType:      operation.Create,
			obj:         &v1.Pod{TypeMeta: metav1.TypeMeta{Kind: "Pod"}},
			scheme:      scheme,
			subresource: "status",
			expected:    "pod_status_create",
			expectErr:   false,
		},
		{
			name:      "without subresource",
			opType:    operation.Update,
			obj:       &v1.Pod{TypeMeta: metav1.TypeMeta{Kind: "Pod"}},
			scheme:    scheme,
			expected:  "pod_update",
			expectErr: false,
		},
		{
			name:      "unknown operation",
			opType:    3, // not a valid operation.Type
			obj:       &v1.Pod{TypeMeta: metav1.TypeMeta{Kind: "Pod"}},
			scheme:    scheme,
			expected:  "pod_unknown_op",
			expectErr: true,
		},
		{
			name:      "no request info and no kind",
			opType:    operation.Create,
			obj:       nil,
			expected:  "unknown_resource_create",
			expectErr: true,
		},
		{
			name:      "known type without kind",
			opType:    operation.Update,
			obj:       &v1.Pod{},
			scheme:    scheme,
			expected:  "pod_update",
			expectErr: false,
		},
		{
			name:      "unknown type with scheme",
			opType:    operation.Create,
			obj:       &runtime.Unknown{}, // Not registered in the scheme
			scheme:    scheme,
			expected:  "unknown_resource_create",
			expectErr: true,
		},
		{
			name:      "unknown type without scheme",
			opType:    operation.Type(4),
			obj:       &runtime.Unknown{}, // Not registered in the scheme
			expected:  "unknown_resource_unknown_op",
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			if tc.obj != nil {
				ctx = genericapirequest.WithRequestInfo(ctx, &genericapirequest.RequestInfo{
					Subresource: tc.subresource,
				})
			}

			result, err := metricIdentifier(ctx, tc.scheme, tc.obj, tc.opType)
			if (err != nil) != tc.expectErr {
				t.Errorf("expected error: %v, got: %v", tc.expectErr, err)
			}
			if result != tc.expected {
				t.Errorf("expected: %s, got: %s", tc.expected, result)
			}
		})
	}
}
