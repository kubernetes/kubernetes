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

package mutating

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/coverage"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/admission/plugin/policy/matching"
	"k8s.io/apiserver/pkg/admission/plugin/policy/mutating/patch"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/util/compatibility"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/discovery"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/openapitest"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/utils/ptr"
)

// annotationRecorder records the annotations added to the attributes it wraps.
type annotationRecorder struct {
	admission.Attributes
	annotations map[string]string
}

func (r *annotationRecorder) AddAnnotation(key, value string) error {
	if r.annotations == nil {
		r.annotations = map[string]string{}
	}
	r.annotations[key] = value
	return r.Attributes.AddAnnotation(key, value)
}

// panicInformer fails the test if params are collected for its hook.
type panicInformer struct{}

func (panicInformer) Informer() cache.SharedIndexInformer { panic("params must not be collected") }
func (panicInformer) Lister() cache.GenericLister         { panic("params must not be collected") }

func coverageViolations(t *testing.T, policyName string) float64 {
	t.Helper()
	values, err := testutil.GetCounterValuesFromGatherer(legacyregistry.DefaultGatherer, "apiserver_admission_coverage_violations_total",
		map[string]string{"name": policyName, "kind": "policy", "type": "admit"}, "operation")
	if err != nil {
		// The vector is not gathered until its first increment.
		return 0
	}
	return values["UPDATE"]
}

// coverageChecks returns the policy check metric's coverage_violation count for
// the (policy, binding) pair.
func coverageChecks(t *testing.T, policyName, bindingName string) float64 {
	t.Helper()
	values, err := testutil.GetCounterValuesFromGatherer(legacyregistry.DefaultGatherer, "apiserver_mutating_admission_policy_check_total",
		map[string]string{"policy": policyName, "policy_binding": bindingName}, "error_type")
	if err != nil {
		// The vector is not gathered until its first increment.
		return 0
	}
	return values["coverage_violation"]
}

func TestDispatcherCoverage(t *testing.T) {
	pods := schema.GroupResource{Resource: "pods"}
	coverageReqs := []coverage.Requirement{{
		Resource: pods,
		Target:   coverage.Shape{Operation: admission.Update, Subresource: "dynamic"},
		Sources:  []coverage.Shape{{Operation: admission.Create}, {Operation: admission.Update}},
	}}
	podGVK := corev1.SchemeGroupVersion.WithKind("Pod")
	podGVR := corev1.SchemeGroupVersion.WithResource("pods")

	rules := func(resources ...string) []admissionregistrationv1.NamedRuleWithOperations {
		return []admissionregistrationv1.NamedRuleWithOperations{{
			RuleWithOperations: admissionregistrationv1.RuleWithOperations{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update},
				Rule: admissionregistrationv1.Rule{
					APIGroups:   []string{""},
					APIVersions: []string{"v1"},
					Resources:   resources,
				},
			},
		}}
	}
	matchResources := func(resourceRules []admissionregistrationv1.NamedRuleWithOperations) *admissionregistrationv1.MatchResources {
		return &admissionregistrationv1.MatchResources{
			NamespaceSelector: &metav1.LabelSelector{},
			ObjectSelector:    &metav1.LabelSelector{},
			ResourceRules:     resourceRules,
		}
	}
	withObjectSelector := func(mr *admissionregistrationv1.MatchResources, app string) *admissionregistrationv1.MatchResources {
		mr.ObjectSelector = &metav1.LabelSelector{MatchLabels: map[string]string{"app": app}}
		return mr
	}
	// Each policy labels the object with its name, so tests can tell whether it ran.
	mutatingPolicy := func(name string, mr *admissionregistrationv1.MatchResources) *Policy {
		return mutations(matchConstraints(policy(name), mr), admissionregistrationv1.Mutation{
			PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
			ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
				Expression: `Object{metadata: Object.metadata{labels: {"mutated-by": "` + name + `"}}}`,
			},
		})
	}
	ignoreFailures := func(p *Policy) *Policy {
		p.Spec.FailurePolicy = ptr.To(admissionregistrationv1.Ignore)
		return p
	}
	binding := func(name, policyName string, mr *admissionregistrationv1.MatchResources) *PolicyBinding {
		return &PolicyBinding{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec:       admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{PolicyName: policyName, MatchResources: mr},
		}
	}
	hook := func(p *Policy, bindings ...*PolicyBinding) PolicyHook {
		return PolicyHook{Policy: p, Bindings: bindings}
	}

	testCases := []struct {
		name           string
		subresource    string
		noRequirements bool
		reinvoke       bool
		hooks          []PolicyHook
		// wantViolation is the expected error message; empty means the request is allowed.
		wantViolation string
		// undetermined means a selector failed to evaluate, so the annotation names no source.
		undetermined  bool
		wantMutatedBy string
	}{
		{
			name:          "violation rejected despite failurePolicy Ignore",
			subresource:   "dynamic",
			hooks:         []PolicyHook{hook(ignoreFailures(mutatingPolicy("p", matchResources(rules("pods")))), binding("b", "p", nil))},
			wantViolation: `pods "pod" is forbidden: MutatingAdmissionPolicy "p" (binding "b") intercepts CREATE pods but not UPDATE pods/dynamic; add a rule for UPDATE on "pods/dynamic" (or "pods/*") to the policy's matchConstraints (and the binding's matchResources, if set), or exclude it explicitly via excludeResourceRules, to allow this request`,
		},
		{
			name:           "no coverage requirements",
			subresource:    "dynamic",
			noRequirements: true,
			hooks:          []PolicyHook{hook(mutatingPolicy("p", matchResources(rules("pods"))), binding("b", "p", nil))},
		},
		{
			name:        "reinvocation is exempt",
			subresource: "dynamic",
			reinvoke:    true,
			hooks:       []PolicyHook{hook(mutatingPolicy("p", matchResources(rules("pods"))), binding("b", "p", nil))},
		},
		{
			name:          "source request is not affected",
			hooks:         []PolicyHook{hook(mutatingPolicy("p", matchResources(rules("pods"))), binding("b", "p", nil))},
			wantMutatedBy: "p",
		},
		{
			name:  "policy intercepting only the target ignores source requests",
			hooks: []PolicyHook{hook(mutatingPolicy("p", matchResources(rules("pods/dynamic"))), binding("b", "p", nil))},
		},
		{
			name:          "policy intercepting the target is called",
			subresource:   "dynamic",
			hooks:         []PolicyHook{hook(mutatingPolicy("p", matchResources(rules("pods", "pods/dynamic"))), binding("b", "p", nil))},
			wantMutatedBy: "p",
		},
		{
			name:          "binding intercepts source but not target",
			subresource:   "dynamic",
			hooks:         []PolicyHook{hook(mutatingPolicy("p", matchResources(rules("pods", "pods/dynamic"))), binding("b", "p", matchResources(rules("pods"))))},
			wantViolation: `MutatingAdmissionPolicy "p" (binding "b") intercepts CREATE pods but not UPDATE pods/dynamic`,
		},
		{
			name:        "policy-level violation of a binding other than the first",
			subresource: "dynamic",
			hooks: []PolicyHook{
				hook(mutatingPolicy("p", matchResources(rules("pods"))),
					binding("b-excluded", "p", withObjectSelector(matchResources(nil), "other")),
					binding("b", "p", nil)),
			},
			wantViolation: `MutatingAdmissionPolicy "p" (binding "b") intercepts CREATE pods but not UPDATE pods/dynamic`,
		},
		{
			// Each binding is checked with its own gaps. The first one
			// intercepts no source, so its gaps never report a violation.
			name:        "binding-level violation of a binding other than the first",
			subresource: "dynamic",
			hooks: []PolicyHook{
				hook(mutatingPolicy("p", matchResources(rules("pods", "pods/dynamic"))),
					binding("b-excluded", "p", withObjectSelector(matchResources(rules("pods/dynamic")), "other")),
					binding("b", "p", matchResources(rules("pods")))),
			},
			wantViolation: `MutatingAdmissionPolicy "p" (binding "b") intercepts CREATE pods but not UPDATE pods/dynamic`,
		},
		{
			name:        "violation short-circuits before params and mutations",
			subresource: "dynamic",
			hooks: []PolicyHook{
				hook(mutatingPolicy("first", matchResources(rules("pods/dynamic"))), binding("b-first", "first", nil)),
				hook(mutatingPolicy("p", matchResources(rules("pods"))), binding("b", "p", nil)),
				{
					Policy: paramKind(mutatingPolicy("last", matchResources(rules("pods/dynamic"))), &admissionregistrationv1.ParamKind{APIVersion: "v1", Kind: "ConfigMap"}),
					Bindings: []*PolicyBinding{func() *PolicyBinding {
						b := binding("b-last", "last", nil)
						b.Spec.ParamRef = &admissionregistrationv1.ParamRef{Name: "param", Namespace: "default"}
						return b
					}()},
					ParamInformer: panicInformer{},
				},
			},
			wantViolation: `MutatingAdmissionPolicy "p" (binding "b") intercepts CREATE pods but not UPDATE pods/dynamic`,
		},
		{
			name:        "undetermined coverage is rejected",
			subresource: "dynamic",
			hooks: []PolicyHook{hook(func() *Policy {
				mr := matchResources(rules("pods"))
				mr.ObjectSelector.MatchExpressions = []metav1.LabelSelectorRequirement{{Key: "app", Operator: "Bogus", Values: []string{"x"}}}
				return mutatingPolicy("p", mr)
			}(), binding("b", "p", nil))},
			wantViolation: `MutatingAdmissionPolicy "p" (binding "b"): unable to determine admission coverage for UPDATE pods/dynamic: `,
			undetermined:  true,
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	tcManager := patch.NewTypeConverterManager(nil, openapitest.NewEmbeddedFileClient())
	go tcManager.Run(ctx)
	err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, time.Second, true, func(context.Context) (done bool, err error) {
		return tcManager.GetTypeConverter(podGVK) != nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	objectInterfaces := admission.NewObjectInterfacesFromScheme(scheme)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewClientset(&corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "default"}})
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			matcher := matching.NewMatcher(informerFactory.Core().V1().Namespaces().Lister(), client)
			paramInformer, err := informerFactory.ForResource(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"})
			if err != nil {
				t.Fatal(err)
			}
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			reqs := coverageReqs
			if tc.noRequirements {
				reqs = nil
			}
			for i := range tc.hooks {
				h := &tc.hooks[i]
				if h.ParamInformer == nil {
					h.ParamInformer = paramInformer
				}
				h.ParamScope = testParamScope{}
				h.Evaluator = compilePolicy(h.Policy)
				if h.CoverageGaps != nil {
					continue
				}
				for _, b := range h.Bindings {
					h.CoverageGaps = append(h.CoverageGaps, coverage.ForPolicyBinding(reqs, h.Policy.Spec.MatchConstraints, b.Spec.MatchResources))
				}
			}

			dispatcher := NewDispatcher(fakeAuthorizer{}, matcher, tcManager)
			if err := dispatcher.Start(ctx); err != nil {
				t.Fatalf("error starting dispatcher: %v", err)
			}

			pod := &corev1.Pod{
				TypeMeta:   metav1.TypeMeta{Kind: "Pod", APIVersion: "v1"},
				ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "default", Labels: map[string]string{"app": "x"}},
			}
			attrs := &annotationRecorder{Attributes: admission.NewAttributesRecord(pod, pod.DeepCopy(), podGVK, "default", "pod", podGVR,
				tc.subresource, admission.Update, &metav1.UpdateOptions{}, false, nil)}
			if tc.reinvoke {
				attrs.GetReinvocationContext().SetIsReinvoke()
			}

			violationsBefore, checksBefore := coverageViolations(t, "p"), coverageChecks(t, "p", "b")
			err = dispatcher.Dispatch(ctx, attrs, objectInterfaces, tc.hooks)
			violations := coverageViolations(t, "p") - violationsBefore
			checks := coverageChecks(t, "p", "b") - checksBefore

			annotation, annotated := attrs.annotations[coverage.AuditAnnotationKey(coverage.PolicyMutating)]
			if tc.wantViolation == "" {
				if err != nil {
					t.Fatalf("Dispatch() returned unexpected error: %v", err)
				}
				if annotated {
					t.Errorf("unexpected coverage audit annotation %s", annotation)
				}
				if violations != 0 || checks != 0 {
					t.Errorf("coverage violations metric increased by %v and policy check metric by %v, want 0", violations, checks)
				}
			} else {
				if !apierrors.IsForbidden(err) {
					t.Fatalf("Dispatch() returned %v, want a Forbidden error", err)
				}
				if got := err.Error(); !strings.Contains(got, tc.wantViolation) {
					t.Errorf("Dispatch() returned %q, want it to contain %q", got, tc.wantViolation)
				}
				if violations != 1 || checks != 1 {
					t.Errorf("coverage violations metric increased by %v and policy check metric by %v, want 1", violations, checks)
				}
				var entries []coverage.AuditAnnotationEntry
				if err := json.Unmarshal([]byte(annotation), &entries); err != nil {
					t.Fatalf("invalid coverage audit annotation %q: %v", annotation, err)
				}
				want := []coverage.AuditAnnotationEntry{{
					Name:    "p",
					Binding: "b",
					Source:  "CREATE pods",
					Target:  "UPDATE pods/dynamic",
				}}
				if tc.undetermined {
					want[0].Source = ""
				}
				if !reflect.DeepEqual(entries, want) {
					t.Errorf("coverage audit annotation = %+v, want %+v", entries, want)
				}
			}
			if got := pod.Labels["mutated-by"]; got != tc.wantMutatedBy {
				t.Errorf("object mutated by %q, want %q", got, tc.wantMutatedBy)
			}
		})
	}
}

// openAPIClientset serves embedded OpenAPI schemas, which the plugin's
// dispatcher factory reads and the fake clientset doesn't implement.
type openAPIClientset struct {
	*fake.Clientset
}

func (c openAPIClientset) Discovery() discovery.DiscoveryInterfaces {
	return openAPIDiscovery{c.Clientset.Discovery()}
}

type openAPIDiscovery struct {
	discovery.DiscoveryInterfaces
}

func (openAPIDiscovery) OpenAPIV3() openapi.Client {
	return openapitest.NewEmbeddedFileClient()
}

// TestCoveragePluginStaticPolicy shows that the plugin builds gaps for
// static policies and enforces them. API policy wiring is covered by the
// integration test.
func TestCoveragePluginStaticPolicy(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MutatingAdmissionPolicy, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	coverageReqs := []coverage.Requirement{{
		Resource: schema.GroupResource{Resource: "pods"},
		Target:   coverage.Shape{Operation: admission.Update, Subresource: "dynamic"},
		Sources:  []coverage.Shape{{Operation: admission.Create}},
	}}

	plugin, err := NewPlugin(nil)
	if err != nil {
		t.Fatal(err)
	}
	plugin.SetStaticManifestsDir(t.TempDir())
	plugin.SetManifestLoaders(&initializer.ManifestLoaders{
		LoadMutatingPolicyManifests: func(string) ([]*admissionregistrationv1.MutatingAdmissionPolicy, []*admissionregistrationv1.MutatingAdmissionPolicyBinding, string, error) {
			// Intercepts the source CREATE pods but not the target.
			p := mutations(matchConstraints(policy("p.static.k8s.io"), &admissionregistrationv1.MatchResources{
				NamespaceSelector: &metav1.LabelSelector{},
				ObjectSelector:    &metav1.LabelSelector{},
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
					RuleWithOperations: admissionregistrationv1.RuleWithOperations{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"pods"}},
					},
				}},
			}), admissionregistrationv1.Mutation{
				PatchType:          admissionregistrationv1.PatchTypeApplyConfiguration,
				ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{Expression: `Object{}`},
			})
			b := &admissionregistrationv1.MutatingAdmissionPolicyBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "b.static.k8s.io"},
				Spec:       admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{PolicyName: p.Name},
			}
			return []*admissionregistrationv1.MutatingAdmissionPolicy{p}, []*admissionregistrationv1.MutatingAdmissionPolicyBinding{b}, "hash", nil
		},
	})

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	client := openAPIClientset{fake.NewClientset()}
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
	plugin.SetAdmissionCoverage(coverageReqs)
	if err := plugin.ValidateInitialization(); err != nil {
		t.Fatal(err)
	}
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())

	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "default"}}
	attrs := admission.NewAttributesRecord(pod, pod.DeepCopy(), corev1.SchemeGroupVersion.WithKind("Pod"), "default", "pod",
		corev1.SchemeGroupVersion.WithResource("pods"), "dynamic", admission.Update, &metav1.UpdateOptions{}, false, nil)
	err = plugin.Admit(ctx, attrs, admission.NewObjectInterfacesFromScheme(scheme))
	if !apierrors.IsForbidden(err) {
		t.Fatalf("Admit() returned %v, want a Forbidden error", err)
	}
	want := `MutatingAdmissionPolicy "p.static.k8s.io" (binding "b.static.k8s.io") intercepts CREATE pods but not UPDATE pods/dynamic`
	if got := err.Error(); !strings.Contains(got, want) {
		t.Errorf("Admit() returned %q, want it to contain %q", got, want)
	}
}
