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

package validating_test

import (
	"context"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utiljson "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/coverage"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/admission/plugin/policy/generic"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/apiserver/pkg/admission/plugin/policy/validating"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/util/compatibility"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/warning"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/utils/ptr"
)

const coverageAnnotationKey = "coverage.admission.k8s.io/validating-admission-policy"

var (
	podsGVR = schema.GroupVersionResource{Version: "v1", Resource: "pods"}
	podGVK  = schema.GroupVersionKind{Version: "v1", Kind: "Pod"}

	createPods       = podRule([]admissionregistrationv1.OperationType{admissionregistrationv1.Create}, "pods")
	createUpdatePods = podRule([]admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update}, "pods")
	// Matches the target directly.
	createUpdatePodsAndDynamic = podRule([]admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update}, "pods", "pods/dynamic")
	updateDynamic              = podRule([]admissionregistrationv1.OperationType{admissionregistrationv1.Update}, "pods/dynamic")
)

// newCoverageRequirements requires UPDATE pods/dynamic to be covered by every hook on CREATE pods.
func newCoverageRequirements(t *testing.T) []coverage.Requirement {
	t.Helper()
	reqs := []coverage.Requirement{{
		Resource: schema.GroupResource{Resource: "pods"},
		Target:   coverage.Shape{Operation: admission.Update, Subresource: "dynamic"},
		Sources:  []coverage.Shape{{Operation: admission.Create}},
	}}
	require.NoError(t, coverage.Validate(reqs))
	return reqs
}

func podRule(ops []admissionregistrationv1.OperationType, resources ...string) admissionregistrationv1.NamedRuleWithOperations {
	return admissionregistrationv1.NamedRuleWithOperations{
		RuleWithOperations: admissionregistrationv1.RuleWithOperations{
			Operations: ops,
			Rule: admissionregistrationv1.Rule{
				APIGroups:   []string{""},
				APIVersions: []string{"v1"},
				Resources:   resources,
				Scope:       ptr.To(admissionregistrationv1.AllScopes),
			},
		},
	}
}

func matchResources(objectLabels map[string]string, rules ...admissionregistrationv1.NamedRuleWithOperations) *admissionregistrationv1.MatchResources {
	return &admissionregistrationv1.MatchResources{
		NamespaceSelector: &metav1.LabelSelector{},
		ObjectSelector:    &metav1.LabelSelector{MatchLabels: objectLabels},
		ResourceRules:     rules,
	}
}

func newPolicy(name string, constraints *admissionregistrationv1.MatchResources) *validating.Policy {
	return &validating.Policy{
		ObjectMeta: metav1.ObjectMeta{Name: name, ResourceVersion: "1"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy:    ptr.To(admissionregistrationv1.Fail),
			MatchConstraints: constraints,
		},
	}
}

func newBinding(name, policyName string, mr *admissionregistrationv1.MatchResources, actions ...admissionregistrationv1.ValidationAction) *validating.PolicyBinding {
	return &validating.PolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: name, ResourceVersion: "1"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        policyName,
			MatchResources:    mr,
			ValidationActions: actions,
		},
	}
}

func podAttributes(subresource string, op admission.Operation, labels map[string]string) *FakeAttributes {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default", Labels: labels}}
	var oldPod runtime.Object
	if op == admission.Update {
		oldPod = pod.DeepCopy()
	}
	return &FakeAttributes{
		Attributes: admission.NewAttributesRecord(pod, oldPod, podGVK, pod.Namespace, pod.Name, podsGVR, subresource, op, nil, false, nil),
	}
}

func admitAll(*validating.Policy) validating.Validator {
	return validateFunc(func(context.Context, schema.GroupVersionResource, *admission.VersionedAttributes, runtime.Object, *v1.Namespace, int64, authorizer.UnconditionalAuthorizer) validating.ValidateResult {
		return validating.ValidateResult{Decisions: []validating.PolicyDecision{{Action: validating.ActionAdmit}}}
	})
}

// setupCoverageTest uses the real matcher, since coverage evaluates selectors
// and resource rules itself.
func setupCoverageTest(t *testing.T, reqs []coverage.Requirement) *generic.PolicyTestContext[*validating.Policy, *validating.PolicyBinding, validating.Validator] {
	t.Cleanup(generic.SetPolicyRefreshIntervalForTests(10 * time.Millisecond))
	testContext, testContextCancel, err := generic.NewPolicyTestContext(
		t,
		validating.NewValidatingAdmissionPolicyAccessor,
		validating.NewValidatingAdmissionPolicyBindingAccessor,
		admitAll,
		func(a authorizer.UnconditionalAuthorizer, m *matching.Matcher, client kubernetes.Interface) generic.Dispatcher[validating.PolicyHook] {
			return validating.NewDispatcher(a, generic.NewPolicyMatcher(m))
		},
		// Policies that match the request are evaluated with its namespace.
		[]runtime.Object{&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "default"}}},
		[]meta.RESTMapping{{
			Resource:         paramsGVK.GroupVersion().WithResource("paramsconfigs"),
			GroupVersionKind: paramsGVK,
			Scope:            meta.RESTScopeNamespace,
		}, {
			Resource:         v1.SchemeGroupVersion.WithResource("namespaces"),
			GroupVersionKind: v1.SchemeGroupVersion.WithKind("Namespace"),
			Scope:            meta.RESTScopeRoot,
		}},
		generic.WithAdmissionCoverage(reqs),
	)
	require.NoError(t, err)
	t.Cleanup(testContextCancel)
	require.NoError(t, testContext.Start())
	return testContext
}

// coverageMetrics returns the coverage violation counter keyed by
// policy name and the policy check counter's coverage_violation series
// keyed by "binding/enforcement_action".
func coverageMetrics(t *testing.T) (violations, checks map[string]float64) {
	t.Helper()
	families, err := legacyregistry.DefaultGatherer.Gather()
	require.NoError(t, err)
	violations, checks = map[string]float64{}, map[string]float64{}
	for _, family := range families {
		for _, m := range family.GetMetric() {
			labels := map[string]string{}
			for _, l := range m.GetLabel() {
				labels[l.GetName()] = l.GetValue()
			}
			switch {
			case family.GetName() == "apiserver_admission_coverage_violations_total" && labels["kind"] == "policy" && labels["type"] == "validate":
				violations[labels["name"]] += m.GetCounter().GetValue()
			case family.GetName() == "apiserver_validating_admission_policy_check_total" && labels["error_type"] == "coverage_violation":
				checks[labels["policy_binding"]+"/"+labels["enforcement_action"]] += m.GetCounter().GetValue()
			}
		}
	}
	return violations, checks
}

// delta returns the non-zero increments from before to after.
func delta(before, after map[string]float64) map[string]float64 {
	d := map[string]float64{}
	for k, v := range after {
		if v != before[k] {
			d[k] = v - before[k]
		}
	}
	return d
}

func TestPluginCoverage(t *testing.T) {
	const (
		violationClause = `intercepts CREATE pods but not UPDATE pods/dynamic; add a rule for UPDATE on "pods/dynamic" (or "pods/*") to the policy's matchConstraints (and the binding's matchResources, if set), or exclude it explicitly via excludeResourceRules, to allow this request`
	)
	// The existing format of Warn action warnings.
	warnText := func(policy, binding string) string {
		return "Validation failed for ValidatingAdmissionPolicy '" + policy + "' with binding '" + binding + "': " + violationClause
	}
	entry := func(policy, binding string) coverage.AuditAnnotationEntry {
		return coverage.AuditAnnotationEntry{Name: policy, Binding: binding, Source: "CREATE pods", Target: "UPDATE pods/dynamic"}
	}
	deny := admissionregistrationv1.Deny
	warn := admissionregistrationv1.Warn
	audit := admissionregistrationv1.Audit

	// Fails with a ConfigurationError under failurePolicy: Fail.
	failingBadParamPolicy := newPolicy("p", matchResources(nil, createUpdatePodsAndDynamic))
	failingBadParamPolicy.Spec.ParamKind = &admissionregistrationv1.ParamKind{APIVersion: paramsGVK.GroupVersion().String(), Kind: "BadParamKind"}

	invalidSelectorPolicy := newPolicy("p", matchResources(nil, createPods))
	invalidSelectorPolicy.Spec.MatchConstraints.ObjectSelector.MatchExpressions = []metav1.LabelSelectorRequirement{{Key: "app", Operator: "Bogus", Values: []string{"x"}}}

	testCases := []struct {
		name     string
		reqs     func(*testing.T) []coverage.Requirement
		objects  []runtime.Object
		attr     *FakeAttributes
		wantDeny string // substring of the 403 message; empty means allowed
		// The aggregated coverage annotation entries, in any order.
		wantEntries  []coverage.AuditAnnotationEntry
		wantWarnings []string
		wantFailures []validating.ValidationFailureValue
	}{
		{
			name:        "Deny rejects",
			objects:     []runtime.Object{newPolicy("p", matchResources(nil, createUpdatePods)), newBinding("b", "p", nil, deny)},
			attr:        podAttributes("dynamic", admission.Update, nil),
			wantDeny:    "ValidatingAdmissionPolicy 'p' with binding 'b' denied request: " + violationClause,
			wantEntries: []coverage.AuditAnnotationEntry{entry("p", "b")},
		},
		{
			name: "Warn warns, Audit audits, both allow and share one annotation",
			objects: []runtime.Object{
				newPolicy("p1", matchResources(nil, createUpdatePods)),
				newBinding("b1", "p1", nil, warn),
				newBinding("b2", "p1", nil, audit),
				newPolicy("p2", matchResources(nil, createPods)),
				newBinding("b3", "p2", nil, warn, audit),
			},
			attr:         podAttributes("dynamic", admission.Update, nil),
			wantEntries:  []coverage.AuditAnnotationEntry{entry("p1", "b1"), entry("p1", "b2"), entry("p2", "b3")},
			wantWarnings: []string{warnText("p1", "b1"), warnText("p2", "b3")},
			wantFailures: []validating.ValidationFailureValue{
				{Message: violationClause, Policy: "p1", Binding: "b2", ExpressionIndex: -1, ValidationActions: []admissionregistrationv1.ValidationAction{audit}},
				{Message: violationClause, Policy: "p2", Binding: "b3", ExpressionIndex: -1, ValidationActions: []admissionregistrationv1.ValidationAction{warn, audit}},
			},
		},
		{
			name: "binding-level violation",
			objects: []runtime.Object{
				newPolicy("p", matchResources(nil, createUpdatePodsAndDynamic)),
				newBinding("b", "p", matchResources(nil, createPods), deny),
			},
			attr:        podAttributes("dynamic", admission.Update, nil),
			wantDeny:    violationClause,
			wantEntries: []coverage.AuditAnnotationEntry{entry("p", "b")},
		},
		{
			name:    "policy that matches the target is evaluated normally",
			objects: []runtime.Object{newPolicy("p", matchResources(nil, createUpdatePodsAndDynamic)), newBinding("b", "p", nil, deny)},
			attr:    podAttributes("dynamic", admission.Update, nil),
		},
		{
			name:    "source requests are unaffected",
			objects: []runtime.Object{newPolicy("p", matchResources(nil, updateDynamic)), newBinding("b", "p", nil, deny)},
			attr:    podAttributes("", admission.Create, nil),
		},
		{
			// The coverage denial is recorded before the configuration error's,
			// so it determines the response.
			name:        "coverage takes precedence over a failing ConfigurationError",
			objects:     []runtime.Object{failingBadParamPolicy, newBinding("b", "p", matchResources(nil, createPods), deny)},
			attr:        podAttributes("dynamic", admission.Update, nil),
			wantDeny:    violationClause,
			wantEntries: []coverage.AuditAnnotationEntry{entry("p", "b")},
		},
		{
			name:         "undetermined coverage is handled per validationActions",
			objects:      []runtime.Object{invalidSelectorPolicy, newBinding("b", "p", nil, warn)},
			attr:         podAttributes("dynamic", admission.Update, nil),
			wantEntries:  []coverage.AuditAnnotationEntry{{Name: "p", Binding: "b", Target: "UPDATE pods/dynamic"}},
			wantWarnings: []string{`Validation failed for ValidatingAdmissionPolicy 'p' with binding 'b': unable to determine admission coverage for UPDATE pods/dynamic: Internal error occurred: "Bogus" is not a valid label selector operator`},
		},
		{
			name:    "no coverage requirements",
			reqs:    func(*testing.T) []coverage.Requirement { return nil },
			objects: []runtime.Object{newPolicy("p", matchResources(nil, createUpdatePods)), newBinding("b", "p", nil, deny)},
			attr:    podAttributes("dynamic", admission.Update, nil),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			reqs := newCoverageRequirements(t)
			if tc.reqs != nil {
				reqs = tc.reqs(t)
			}
			testContext := setupCoverageTest(t, reqs)
			require.NoError(t, testContext.UpdateAndWait(tc.objects...))

			warnings := newWarningRecorder()
			violationsBefore, checksBefore := coverageMetrics(t)
			// Policies that match the request are evaluated, which needs a real converter.
			err := testContext.Plugin.Dispatch(warning.WithWarningRecorder(testContext, warnings), tc.attr, admission.NewObjectInterfacesFromScheme(clientgoscheme.Scheme))
			violationsAfter, checksAfter := coverageMetrics(t)

			// Every action of a violating binding is counted once.
			wantViolations, wantChecks := map[string]float64{}, map[string]float64{}
			for _, e := range tc.wantEntries {
				wantViolations[e.Name]++
				for _, obj := range tc.objects {
					if b, ok := obj.(*validating.PolicyBinding); ok && b.Name == e.Binding {
						for _, action := range b.Spec.ValidationActions {
							wantChecks[e.Binding+"/"+strings.ToLower(string(action))]++
						}
					}
				}
			}
			require.Equal(t, wantViolations, delta(violationsBefore, violationsAfter), "coverage violations metric increments")
			require.Equal(t, wantChecks, delta(checksBefore, checksAfter), "policy check metric coverage_violation increments")

			if tc.wantDeny == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, tc.wantDeny)
				var statusErr *apierrors.StatusError
				require.ErrorAs(t, err, &statusErr)
				require.Equal(t, int32(http.StatusForbidden), statusErr.Status().Code)
				require.Equal(t, metav1.StatusReasonForbidden, statusErr.Status().Reason)
			}

			require.Equal(t, len(tc.wantWarnings), warnings.len())
			for _, w := range tc.wantWarnings {
				require.True(t, warnings.hasWarning(w), "missing warning %q", w)
			}

			annotations := tc.attr.GetAnnotations(auditinternal.LevelMetadata)
			if len(tc.wantEntries) == 0 {
				require.NotContains(t, annotations, coverageAnnotationKey)
			} else {
				var entries []coverage.AuditAnnotationEntry
				require.NoError(t, utiljson.Unmarshal([]byte(annotations[coverageAnnotationKey]), &entries))
				require.ElementsMatch(t, tc.wantEntries, entries)
			}
			if len(tc.wantFailures) == 0 {
				require.NotContains(t, annotations, "validation.policy.admission.k8s.io/validation_failure")
			} else {
				var failures []validating.ValidationFailureValue
				require.NoError(t, utiljson.Unmarshal([]byte(annotations["validation.policy.admission.k8s.io/validation_failure"]), &failures))
				require.ElementsMatch(t, tc.wantFailures, failures)
			}
		})
	}
}

// TestCoveragePluginStaticPolicy shows that the plugin builds gaps for
// static policies and enforces them. API policy wiring is covered by the
// integration test.
func TestCoveragePluginStaticPolicy(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	plugin, err := validating.NewPlugin(nil)
	require.NoError(t, err)
	plugin.SetStaticManifestsDir(t.TempDir())
	plugin.SetManifestLoaders(&initializer.ManifestLoaders{
		LoadValidatingPolicyManifests: func(string) ([]*admissionregistrationv1.ValidatingAdmissionPolicy, []*admissionregistrationv1.ValidatingAdmissionPolicyBinding, string, error) {
			return []*admissionregistrationv1.ValidatingAdmissionPolicy{newPolicy("p.static.k8s.io", matchResources(nil, createUpdatePods))},
				[]*admissionregistrationv1.ValidatingAdmissionPolicyBinding{newBinding("b.static.k8s.io", "p.static.k8s.io", nil, admissionregistrationv1.Deny)},
				"hash", nil
		},
	})

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	client := fake.NewClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	initializer.New(
		client,
		dynamicfake.NewSimpleDynamicClient(runtime.NewScheme()),
		informerFactory,
		authorizerfactory.NewAlwaysAllowAuthorizer(),
		utilfeature.DefaultFeatureGate,
		compatibility.DefaultBuildEffectiveVersion(),
		ctx.Done(),
		meta.NewDefaultRESTMapper(nil),
	).Initialize(plugin)
	plugin.SetAdmissionCoverage(newCoverageRequirements(t))
	require.NoError(t, plugin.ValidateInitialization())
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())

	err = plugin.Validate(ctx, podAttributes("dynamic", admission.Update, nil), admission.NewObjectInterfacesFromScheme(clientgoscheme.Scheme))
	require.ErrorContains(t, err, "ValidatingAdmissionPolicy 'p.static.k8s.io' with binding 'b.static.k8s.io' denied request: intercepts CREATE pods but not UPDATE pods/dynamic")
	require.True(t, apierrors.IsForbidden(err))
}
