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
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
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
		options     sets.Set[string]
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
			name:        "update subresource",
			subresource: "/status",
			object:      valid,
			oldObject:   valid,
			expected:    field.ErrorList{invalidStatusErr},
		},
		{
			name:        "invalid subresource",
			subresource: "invalid/status",
			object:      valid,
			oldObject:   valid,
			expected:    field.ErrorList{invalidSubresourceErr},
		},
		{
			name:     "update with option",
			options:  sets.New("option1"),
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

	scheme.AddValidationFunc(&v1.Pod{}, func(ctx context.Context, op operation.Operation, object, oldObject interface{}, subresources ...string) field.ErrorList {
		results := field.ErrorList{}
		if op.Options.Has("option1") {
			results = append(results, invalidIfOptionErr)
		}
		if len(subresources) == 1 && subresources[0] == "status" {
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
			var results field.ErrorList
			if tc.oldObject == nil {
				results = ValidateDeclaratively(ctx, tc.options, scheme, tc.object)
			} else {
				results = ValidateUpdateDeclaratively(ctx, tc.options, scheme, tc.object, tc.oldObject)
			}
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
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			details := gatherDeclarativeValidationMismatches(tc.imperativeErrors, tc.declarativeErrors, tc.takeover)
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
			// logs have a prefix of the form - I0309 21:05:33.865030 1926106 validate.go:199]
			expectedRegex: "I.*Unexpected difference between hand written validation and declarative validation error results.*Consider disabling the DeclarativeValidationTakeover feature gate to keep data persisted in etcd consistent with prior versions of Kubernetes",
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

			CompareDeclarativeErrorsAndEmitMismatches(ctx, tc.imperativeErrs, tc.declarativeErrs, tc.takeover)

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
	options := sets.New[string]()
	obj := &runtime.Unknown{}

	testCases := []struct {
		name            string
		validateFn      func(context.Context, sets.Set[string], *runtime.Scheme, runtime.Object) field.ErrorList
		takeoverEnabled bool
		wantErrs        field.ErrorList
		expectLogRegex  string
	}{
		{
			name: "no panic",
			validateFn: func(context.Context, sets.Set[string], *runtime.Scheme, runtime.Object) field.ErrorList {
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
			validateFn: func(context.Context, sets.Set[string], *runtime.Scheme, runtime.Object) field.ErrorList {
				panic("test panic")
			},
			takeoverEnabled: false,
			wantErrs:        nil,
			// logs have a prefix of the form - I0309 21:05:33.865030 1926106 validate.go:199]
			expectLogRegex: "I.*panic during declarative validation: test panic",
		},
		{
			name: "panic with takeover enabled",
			validateFn: func(context.Context, sets.Set[string], *runtime.Scheme, runtime.Object) field.ErrorList {
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
			validateFn: func(context.Context, sets.Set[string], *runtime.Scheme, runtime.Object) field.ErrorList {
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

			// Pass the takeover flag to withRecover instead of relying on the feature gate
			wrapped := withRecover(tc.validateFn, tc.takeoverEnabled)
			gotErrs := wrapped(ctx, options, scheme, obj)

			klog.Flush()
			logOutput := buf.String()

			// Compare gotErrs vs. tc.wantErrs
			if !equalErrorLists(gotErrs, tc.wantErrs) {
				t.Errorf("withRecover() gotErrs = %#v, want %#v", gotErrs, tc.wantErrs)
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
	options := sets.New[string]()
	obj := &runtime.Unknown{}
	oldObj := &runtime.Unknown{}

	testCases := []struct {
		name            string
		validateFn      func(context.Context, sets.Set[string], *runtime.Scheme, runtime.Object, runtime.Object) field.ErrorList
		takeoverEnabled bool
		wantErrs        field.ErrorList
		expectLogRegex  string
	}{
		{
			name: "no panic",
			validateFn: func(context.Context, sets.Set[string], *runtime.Scheme, runtime.Object, runtime.Object) field.ErrorList {
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
			validateFn: func(context.Context, sets.Set[string], *runtime.Scheme, runtime.Object, runtime.Object) field.ErrorList {
				panic("test update panic")
			},
			takeoverEnabled: false,
			wantErrs:        nil,
			// logs have a prefix of the form - I0309 21:05:33.865030 1926106 validate.go:199]
			expectLogRegex: "I.*panic during declarative validation: test update panic",
		},
		{
			name: "panic with takeover enabled",
			validateFn: func(context.Context, sets.Set[string], *runtime.Scheme, runtime.Object, runtime.Object) field.ErrorList {
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
			validateFn: func(context.Context, sets.Set[string], *runtime.Scheme, runtime.Object, runtime.Object) field.ErrorList {
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

			// Pass the takeover flag to withRecoverUpdate instead of relying on the feature gate
			wrapped := withRecoverUpdate(tc.validateFn, tc.takeoverEnabled)
			gotErrs := wrapped(ctx, options, scheme, obj, oldObj)

			klog.Flush()
			logOutput := buf.String()

			// Compare gotErrs with wantErrs
			if !equalErrorLists(gotErrs, tc.wantErrs) {
				t.Errorf("withRecoverUpdate() gotErrs = %#v, want %#v", gotErrs, tc.wantErrs)
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

func TestValidateDeclarativelyWithRecovery(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	options := sets.New[string]()
	obj := &runtime.Unknown{}

	// Simple test for the ValidateDeclarativelyWithRecovery function
	t.Run("with takeover disabled", func(t *testing.T) {
		errs := ValidateDeclarativelyWithRecovery(ctx, options, scheme, obj, false)
		if errs == nil {
			// This is expected to error since the request info is missing
			t.Errorf("Expected errors but got nil")
		}
	})

	t.Run("with takeover enabled", func(t *testing.T) {
		errs := ValidateDeclarativelyWithRecovery(ctx, options, scheme, obj, true)
		if errs == nil {
			// This is expected to error since the request info is missing
			t.Errorf("Expected errors but got nil")
		}
	})
}

func TestValidateUpdateDeclarativelyWithRecovery(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	options := sets.New[string]()
	obj := &runtime.Unknown{}
	oldObj := &runtime.Unknown{}

	// Simple test for the ValidateUpdateDeclarativelyWithRecovery function
	t.Run("with takeover disabled", func(t *testing.T) {
		errs := ValidateUpdateDeclarativelyWithRecovery(ctx, options, scheme, obj, oldObj, false)
		if errs == nil {
			// This is expected to error since the request info is missing
			t.Errorf("Expected errors but got nil")
		}
	})

	t.Run("with takeover enabled", func(t *testing.T) {
		errs := ValidateUpdateDeclarativelyWithRecovery(ctx, options, scheme, obj, oldObj, true)
		if errs == nil {
			// This is expected to error since the request info is missing
			t.Errorf("Expected errors but got nil")
		}
	})
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
