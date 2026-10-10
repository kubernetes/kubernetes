/*
Copyright The Kubernetes Authors.

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

package equivalents

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	admissiontesting "k8s.io/apiserver/pkg/admission/testing"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

var widgets = schema.GroupVersionResource{Group: "example.io", Version: "v1", Resource: "widgets"}

func target(sub string, op admission.Operation) ResourceOperation {
	return ResourceOperation{resource: widgets, subresource: sub, operation: op}
}

// turboUpdate returns attributes for an UPDATE of widgets/turbo that record annotations.
func turboUpdate() *recordingAttributes {
	return &recordingAttributes{Attributes: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "example.io", Version: "v1", Kind: "Widget"},
		"ns", "w1", widgets, "turbo", admission.Update, nil, false, nil)}
}

type recordingAttributes struct {
	admission.Attributes
	annotations map[string]string
}

func (r *recordingAttributes) AddAnnotation(key, value string) error {
	if r.annotations == nil {
		r.annotations = map[string]string{}
	}
	r.annotations[key] = value
	return nil
}

func TestExpand(t *testing.T) {
	testcases := []struct {
		name   string
		source ResourceOperation
		eqs    []admission.Equivalent
		expect []ResourceOperation
	}{{
		name:   "none",
		source: target("turbo", admission.Update),
	}, {
		name:   "nil operations use the request operation",
		source: target("turbo", admission.Update),
		eqs:    []admission.Equivalent{{Subresource: "resize"}},
		expect: []ResourceOperation{target("resize", admission.Update)},
	}, {
		name:   "empty operations use the request operation",
		source: target("turbo", admission.Delete),
		eqs:    []admission.Equivalent{{Subresource: "resize", Operations: []admission.Operation{}}},
		expect: []ResourceOperation{target("resize", admission.Delete)},
	}, {
		name:   "explicit operations are independent of the request operation",
		source: target("turbo", admission.Connect),
		eqs:    []admission.Equivalent{{Operations: []admission.Operation{admission.Create, admission.Update}}},
		expect: []ResourceOperation{target("", admission.Create), target("", admission.Update)},
	}, {
		name:   "declaration order, then operation order",
		source: target("turbo", admission.Update),
		eqs: []admission.Equivalent{
			{Operations: []admission.Operation{admission.Update, admission.Create}},
			{Subresource: "resize"},
			{Subresource: "extras"},
		},
		expect: []ResourceOperation{target("", admission.Update), target("", admission.Create), target("resize", admission.Update), target("extras", admission.Update)},
	}, {
		// Self-references are rejected at installation; duplicates are harmless.
		name:   "no deduplication",
		source: target("turbo", admission.Update),
		eqs: []admission.Equivalent{
			{Operations: []admission.Operation{admission.Update, admission.Update}},
			{},
		},
		expect: []ResourceOperation{target("", admission.Update), target("", admission.Update), target("", admission.Update)},
	}}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			got := expand(tc.source, tc.eqs)
			if diff := cmp.Diff(tc.expect, got, cmp.AllowUnexported(ResourceOperation{})); diff != "" {
				t.Errorf("unexpected targets (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAttrWithResourceOperation(t *testing.T) {
	attr := turboUpdate()
	wrapped := &attrWithResourceOperation{Attributes: attr, target: target("", admission.Create)}
	if got := resourceOperationOf(wrapped); got != target("", admission.Create) {
		t.Errorf("expected overridden resource operation, got %v", got)
	}
	// Matchers must not annotate the real request on behalf of a hypothetical one.
	if err := wrapped.AddAnnotation("example.io/key", "v"); err == nil {
		t.Error("expected AddAnnotation to fail")
	}
	if err := wrapped.AddAnnotationWithLevel("example.io/key", "v", auditinternal.LevelMetadata); err == nil {
		t.Error("expected AddAnnotationWithLevel to fail")
	}
	if len(attr.annotations) != 0 {
		t.Errorf("expected no annotations on the request, got %v", attr.annotations)
	}
}

func TestTargetsWithoutDeclarations(t *testing.T) {
	attr := turboUpdate()
	if got := Targets(attr, admissiontesting.ObjectInterfacesWithEquivalents{}); got != nil {
		t.Errorf("expected no targets without declarations, got %v", got)
	}
	if got := Targets(attr, admission.NewObjectInterfacesFromScheme(runtime.NewScheme())); got != nil {
		t.Errorf("expected no targets for ObjectInterfaces without EquivalentsGetter, got %v", got)
	}
}

func TestCheck(t *testing.T) {
	// A StatusError, which NewForbidden would return verbatim if it were passed through.
	errEval := apierrors.NewForbidden(schema.GroupResource{Resource: "namespaces"}, "ns", errors.New("nope"))
	// matchesOnly returns a matcher selecting exactly the given resource operations, or failing for errOn.
	matchesOnly := func(errOn []ResourceOperation, selected ...ResourceOperation) func(admission.Attributes) (bool, error) {
		return func(a admission.Attributes) (bool, error) {
			e := resourceOperationOf(a)
			if slices.Contains(errOn, e) {
				return false, errEval
			}
			return slices.Contains(selected, e), nil
		}
	}
	source := target("turbo", admission.Update)
	main, resize := target("", admission.Update), target("resize", admission.Update)

	testcases := []struct {
		name      string
		matches   func(admission.Attributes) (bool, error)
		expectErr string // substring; "" for no error
	}{{
		name:    "matches nothing",
		matches: matchesOnly(nil),
	}, {
		name:    "matches source only",
		matches: matchesOnly(nil, source),
	}, {
		name:    "matches source and targets",
		matches: matchesOnly(nil, source, main, resize),
	}, {
		// Dispatch hits the same error for the request, so it is not treated more leniently.
		name:    "uniform evaluation error is left to dispatch",
		matches: matchesOnly([]ResourceOperation{source, main, resize}),
	}, {
		name:    "source errors, targets do not match",
		matches: matchesOnly([]ResourceOperation{source}),
	}, {
		name:      "evaluation error on the source only",
		matches:   matchesOnly([]ResourceOperation{source}, main, resize),
		expectErr: `ValidatingAdmissionWebhook "h" (configuration "c") applies to widgets (UPDATE) but could not be evaluated for widgets/turbo (UPDATE) (namespaces "ns" is forbidden: nope).`,
	}, {
		name:      "source errors, target matches cleanly",
		matches:   matchesOnly([]ResourceOperation{source, main}, resize),
		expectErr: `ValidatingAdmissionWebhook "h" (configuration "c") applies to widgets/resize (UPDATE) but could not be evaluated for widgets/turbo (UPDATE) (namespaces "ns" is forbidden: nope).`,
	}, {
		name:      "matches a target only",
		matches:   matchesOnly(nil, resize),
		expectErr: `"h" (configuration "c") applies to widgets/resize (UPDATE) but not widgets/turbo (UPDATE). Add`,
	}, {
		name:      "first matching target is reported",
		matches:   matchesOnly(nil, main, resize),
		expectErr: `"h" (configuration "c") applies to widgets (UPDATE) but not widgets/turbo (UPDATE). Add`,
	}, {
		name:      "evaluation error on a target fails closed",
		matches:   matchesOnly([]ResourceOperation{main}),
		expectErr: `applies to widgets (UPDATE) but not widgets/turbo (UPDATE) (match criteria could not be evaluated: namespaces "ns" is forbidden: nope). Add`,
	}}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			attr := turboUpdate()
			targets := Targets(attr, admissiontesting.ObjectInterfacesWithEquivalents{Equivalents: []admission.Equivalent{{}, {Subresource: "resize"}}})
			v, ok := Check(attr, targets, Hook{Plugin: "ValidatingAdmissionWebhook", Name: "h", Configuration: "c"}, tc.matches)
			if tc.expectErr == "" {
				if ok {
					t.Errorf("unexpected violation: %s", v.message(source))
				}
				return
			}
			if !ok {
				t.Fatalf("expected violation containing %q, got none", tc.expectErr)
			}
			if err := Reject(context.Background(), attr, []Violation{v}); err == nil || !strings.Contains(err.Error(), tc.expectErr) {
				t.Errorf("expected error containing %q, got %v", tc.expectErr, err)
			}
		})
	}
}

func TestRejectWithoutViolations(t *testing.T) {
	attr := turboUpdate()
	// err is compared to untyped nil, so a typed nil would fail.
	if err := Reject(context.Background(), attr, nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(attr.annotations) != 0 {
		t.Errorf("expected no annotations, got %v", attr.annotations)
	}
}

func TestReject(t *testing.T) {
	attr := turboUpdate()
	err := Reject(context.Background(), attr, []Violation{{
		hook:   Hook{Plugin: "ValidatingAdmissionWebhook", Name: "check.example.com", Configuration: "widget-checks"},
		target: target("", admission.Update),
	}, {
		hook:   Hook{Plugin: "ValidatingAdmissionPolicy", Name: "no-privileged", Binding: "no-privileged-prod"},
		target: target("resize", admission.Update),
	}})
	if !apierrors.IsForbidden(err) {
		t.Fatalf("expected Forbidden, got %v", err)
	}
	const violations = `ValidatingAdmissionWebhook "check.example.com" (configuration "widget-checks") applies to widgets (UPDATE) but not widgets/turbo (UPDATE); ` +
		`ValidatingAdmissionPolicy "no-privileged" (binding "no-privileged-prod") applies to widgets/resize (UPDATE) but not widgets/turbo (UPDATE)`
	const expectMsg = `widgets.example.io "w1" is forbidden: request to widgets/turbo is not covered by admission hooks that apply to its admission equivalents: ` +
		violations + `. Add "widgets/turbo" to the rules of these hooks, or remove their rules for the listed resources.`
	if diff := cmp.Diff(expectMsg, err.Error()); diff != "" {
		t.Errorf("unexpected message (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(violations, attr.annotations[AuditAnnotationKey]); diff != "" {
		t.Errorf("unexpected audit annotation (-want +got):\n%s", diff)
	}

	// Other tests in this package also record rejections; the hook names above are unique to
	// this test.
	for plugin, name := range map[string]string{"ValidatingAdmissionWebhook": "check.example.com", "ValidatingAdmissionPolicy": "no-privileged"} {
		got, err := testutil.GetCounterValuesFromGatherer(legacyregistry.DefaultGatherer, "apiserver_admission_equivalent_coverage_rejections_total",
			map[string]string{"plugin": plugin, "resource": "widgets.example.io", "subresource": "turbo"}, "name")
		if err != nil {
			t.Fatal(err)
		}
		if got[name] != 1 {
			t.Errorf("expected 1 rejection for %s %q, got %v", plugin, name, got)
		}
	}
}

func TestRejectHint(t *testing.T) {
	const hint = `. Add "widgets/turbo" to the rules of these hooks`
	errEval := errors.New("bad selector")
	sourceErrViolation := Violation{
		hook:      Hook{Plugin: "ValidatingAdmissionWebhook", Name: "source-error", Configuration: "cfg"},
		target:    target("", admission.Update),
		sourceErr: errEval,
	}
	targetErrViolation := Violation{
		hook:      Hook{Plugin: "ValidatingAdmissionWebhook", Name: "target-error", Configuration: "cfg"},
		target:    target("", admission.Update),
		targetErr: errEval,
	}
	testcases := []struct {
		name       string
		violations []Violation
		expectMsg  string
	}{{
		// Adding the request to the hook's rules would not fix an evaluation error.
		name:       "source errors only",
		violations: []Violation{sourceErrViolation},
		expectMsg: `widgets.example.io "w1" is forbidden: request to widgets/turbo is not covered by admission hooks that apply to its admission equivalents: ` +
			`ValidatingAdmissionWebhook "source-error" (configuration "cfg") applies to widgets (UPDATE) but could not be evaluated for widgets/turbo (UPDATE) (bad selector).`,
	}, {
		name:       "source error and coverage violation",
		violations: []Violation{sourceErrViolation, targetErrViolation},
		expectMsg: `widgets.example.io "w1" is forbidden: request to widgets/turbo is not covered by admission hooks that apply to its admission equivalents: ` +
			`ValidatingAdmissionWebhook "source-error" (configuration "cfg") applies to widgets (UPDATE) but could not be evaluated for widgets/turbo (UPDATE) (bad selector); ` +
			`ValidatingAdmissionWebhook "target-error" (configuration "cfg") applies to widgets (UPDATE) but not widgets/turbo (UPDATE) (match criteria could not be evaluated: bad selector)` +
			hint + `, or remove their rules for the listed resources.`,
	}}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			err := Reject(context.Background(), turboUpdate(), tc.violations)
			if err == nil {
				t.Fatal("expected error")
			}
			if diff := cmp.Diff(tc.expectMsg, err.Error()); diff != "" {
				t.Errorf("unexpected message (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRejectTruncation(t *testing.T) {
	attr := turboUpdate()
	const n = maxViolationsInMessage + 2
	var violations []Violation
	for i := 0; i < n; i++ {
		violations = append(violations, Violation{
			hook:   Hook{Plugin: "ValidatingAdmissionWebhook", Name: fmt.Sprintf("hook-%d", i), Configuration: "cfg"},
			target: target("", admission.Update),
		})
	}
	msg := Reject(context.Background(), attr, violations).Error()
	annotation := attr.annotations[AuditAnnotationKey]
	for i := 0; i < n; i++ {
		hook := fmt.Sprintf(`"hook-%d"`, i)
		if got, want := strings.Contains(msg, hook), i < maxViolationsInMessage; got != want {
			t.Errorf("hook-%d in message: got %v, want %v: %s", i, got, want, msg)
		}
		if !strings.Contains(annotation, hook) {
			t.Errorf("expected hook-%d in audit annotation: %s", i, annotation)
		}
	}
	if !strings.Contains(msg, `hook-4" (configuration "cfg") applies to widgets (UPDATE) but not widgets/turbo (UPDATE); and 2 more. Add`) {
		t.Errorf("expected truncation marker in message: %s", msg)
	}
}
