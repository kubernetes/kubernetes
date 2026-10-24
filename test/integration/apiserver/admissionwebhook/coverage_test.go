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

package admissionwebhook

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"

	admissionreviewv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/coverage"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/admission/plugin/namespace/lifecycle"
	mutatingadmissionpolicy "k8s.io/apiserver/pkg/admission/plugin/policy/mutating"
	validatingadmissionpolicy "k8s.io/apiserver/pkg/admission/plugin/policy/validating"
	mutatingwebhook "k8s.io/apiserver/pkg/admission/plugin/webhook/mutating"
	webhooktesting "k8s.io/apiserver/pkg/admission/plugin/webhook/testing"
	validatingwebhook "k8s.io/apiserver/pkg/admission/plugin/webhook/validating"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	"k8s.io/kubernetes/test/integration/framework"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

const (
	coverageWebhookName = "coverage.integration.k8s.io"

	// Each namespace has two pods, so that a request allowed by a selector is
	// known to have been evaluated with a configuration that rejects the other.
	coveragePodX = "pod-x" // app=x
	coveragePodY = "pod-y" // app=y

	// coverageHint is part of every rejection for coverageRequirement, whatever
	// the hook kind.
	coverageHint = `add a rule for UPDATE on "pods/status" (or "pods/*")`
)

// coverageRequirement is test-only: kube-apiserver declares no coverage
// requirements yet, and nothing else writes pods/status in this test. CREATE
// is the first source, so it's the one named in rejections.
var coverageRequirement = coverage.Requirement{
	Resource: schema.GroupResource{Resource: "pods"},
	Target:   coverage.Shape{Operation: admission.Update, Subresource: "status"},
	Sources:  []coverage.Shape{{Operation: admission.Create}, {Operation: admission.Update}},
}

// coverageHookScope is the part of a hook's configuration that the test varies.
type coverageHookScope struct {
	resources      []string
	objectSelector *metav1.LabelSelector
}

var (
	// sourceOnlyScope intercepts the requirement's sources but not its target.
	sourceOnlyScope = coverageHookScope{resources: []string{"pods"}}
	// withTargetScope also intercepts the target.
	withTargetScope = coverageHookScope{resources: []string{"pods", "pods/status"}}
	// podYOnlyScope intercepts only the sources, and only for pod-y.
	podYOnlyScope = coverageHookScope{
		resources:      []string{"pods"},
		objectSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "y"}},
	}
)

// TestAdmissionCoverage runs a kube-apiserver with a test-only coverage
// requirement (hooks on CREATE/UPDATE pods must also intercept UPDATE
// pods/status) and checks that the requirement reaches each kind of hook. The
// coverage semantics are unit-tested in k8s.io/apiserver.
func TestAdmissionCoverage(t *testing.T) {
	reqs := []coverage.Requirement{coverageRequirement}
	if err := coverage.Validate(reqs); err != nil {
		t.Fatalf("invalid coverage requirement: %v", err)
	}
	webhookURL := startCoverageWebhookServer(t)

	client, _, tearDown := framework.StartTestServer(t.Context(), t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Don't depend on the host having a default route to pick an
			// address. Like kubeapiservertesting, disable endpoint
			// reconciliation, which rejects a loopback address.
			opts.GenericServerRunOptions.AdvertiseAddress = netutils.ParseIPSloppy("127.0.0.1")
			opts.EndpointReconcilerType = "none"
			// No service account controller runs, so pod creation would fail.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount"}
			opts.Admission.GenericAdmission.Plugins = pluginsWithCoverage(t, opts.Admission.GenericAdmission.Plugins, reqs)
		},
	})
	t.Cleanup(tearDown)

	admissionregistration := client.AdmissionregistrationV1()
	webhookRejection := func(ns string) []string {
		return []string{fmt.Sprintf("admission webhook %q (configuration %q) intercepts CREATE pods but not UPDATE pods/status", coverageWebhookName, ns)}
	}
	hooks := []struct {
		name string
		// apply creates or updates the hook under test, scoped to namespace ns.
		apply func(ctx context.Context, t *testing.T, ns string, scope coverageHookScope)
		// wantRejection returns the hook-specific parts of the rejection message.
		wantRejection func(ns string) []string
	}{
		{
			name: "validating-webhook",
			apply: func(ctx context.Context, t *testing.T, ns string, scope coverageHookScope) {
				createOrUpdateCoverageHook(ctx, t, admissionregistration.ValidatingWebhookConfigurations(), &admissionregistrationv1.ValidatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{Name: ns},
					Webhooks:   []admissionregistrationv1.ValidatingWebhook{coverageValidatingWebhook(coverageWebhookName, webhookURL, ns, scope)},
				})
			},
			wantRejection: webhookRejection,
		},
		{
			name: "mutating-webhook",
			apply: func(ctx context.Context, t *testing.T, ns string, scope coverageHookScope) {
				createOrUpdateCoverageHook(ctx, t, admissionregistration.MutatingWebhookConfigurations(), &admissionregistrationv1.MutatingWebhookConfiguration{
					ObjectMeta: metav1.ObjectMeta{Name: ns},
					Webhooks: webhooktesting.ConvertToMutatingWebhooks([]admissionregistrationv1.ValidatingWebhook{
						coverageValidatingWebhook(coverageWebhookName, webhookURL, ns, scope),
					}),
				})
			},
			wantRejection: webhookRejection,
		},
		{
			name: "validating-admission-policy",
			// The server observes the policy and binding separately. Writing
			// the binding first means that every intermediate configuration
			// is at least as permissive as the final one for pod-x: a binding
			// without its policy is no hook, and the selector reaches the
			// binding while the policy still intercepts the target.
			apply: func(ctx context.Context, t *testing.T, ns string, scope coverageHookScope) {
				createOrUpdateCoverageHook(ctx, t, admissionregistration.ValidatingAdmissionPolicyBindings(), coverageVAPBinding(ns, ns, ns, scope))
				createOrUpdateCoverageHook(ctx, t, admissionregistration.ValidatingAdmissionPolicies(), coverageVAP(ns, scope))
			},
			wantRejection: func(ns string) []string {
				return []string{"ValidatingAdmissionPolicy '" + ns + "' with binding '" + ns + "'", "intercepts CREATE pods but not UPDATE pods/status"}
			},
		},
		{
			name: "mutating-admission-policy",
			// Binding first, for the same reason as validating-admission-policy.
			apply: func(ctx context.Context, t *testing.T, ns string, scope coverageHookScope) {
				createOrUpdateCoverageHook(ctx, t, admissionregistration.MutatingAdmissionPolicyBindings(), coverageMAPBinding(ns, ns, ns, scope))
				createOrUpdateCoverageHook(ctx, t, admissionregistration.MutatingAdmissionPolicies(), coverageMAP(ns, scope))
			},
			wantRejection: func(ns string) []string {
				return []string{fmt.Sprintf("MutatingAdmissionPolicy %q (binding %q) intercepts CREATE pods but not UPDATE pods/status", ns, ns)}
			},
		},
	}
	for _, hook := range hooks {
		t.Run(hook.name, func(t *testing.T) {
			ctx := t.Context()
			ns := "coverage-" + hook.name
			createCoverageNamespaceAndPods(t, client, ns)
			for _, pod := range []string{coveragePodX, coveragePodY} {
				if err := updateCoveragePodStatus(ctx, client, ns, pod); err != nil {
					t.Fatalf("UPDATE pods/status of %s failed before any hook exists: %v", pod, err)
				}
			}

			// failurePolicy is Ignore and, for policies, the binding's
			// matchResources has only selectors; neither suppresses coverage.
			hook.apply(ctx, t, ns, sourceOnlyScope)
			expectCoverageRejection(t, waitForPodStatusUpdate(t, client, ns, coveragePodY, true), hook.wantRejection(ns)...)
			expectCoverageRejection(t, updateCoveragePodStatus(ctx, client, ns, coveragePodX), hook.wantRejection(ns)...)
			// Coverage is directional: the violating hook's own sources are unaffected.
			if err := updateCoveragePod(ctx, client, ns, coveragePodX); err != nil {
				t.Errorf("UPDATE pods of %s failed while the violating hook intercepts it: %v", coveragePodX, err)
			}

			// The hook is updated in place, so the request can only be allowed
			// once the server has observed the target in the rules.
			hook.apply(ctx, t, ns, withTargetScope)
			_ = waitForPodStatusUpdate(t, client, ns, coveragePodX, false)
			if err := updateCoveragePodStatus(ctx, client, ns, coveragePodY); err != nil {
				t.Errorf("UPDATE pods/status of %s failed with the target in the rules: %v", coveragePodY, err)
			}

			// Only the configuration with the selector both rejects pod-y and,
			// since the selector doesn't match pod-x, allows pod-x. Waiting
			// for both in the same round doesn't depend on the order in which
			// the server observes the hook's objects.
			hook.apply(ctx, t, ns, podYOnlyScope)
			errs := waitForPodStatusUpdates(t, client, ns, map[string]bool{coveragePodY: true, coveragePodX: false})
			expectCoverageRejection(t, errs[coveragePodY], hook.wantRejection(ns)...)
		})
	}
}

// pluginsWithCoverage returns a registry with the same plugins as defaults, whose
// webhook and policy plugins enforce reqs. A production server would inject reqs with
// a plugin initializer; the factories set them at the same point, before
// ValidateInitialization, without changing kube-apiserver.
func pluginsWithCoverage(t *testing.T, defaults *admission.Plugins, reqs []coverage.Requirement) *admission.Plugins {
	t.Helper()
	plugins := admission.NewPlugins()
	lifecycle.Register(plugins)
	plugins.Register(validatingwebhook.PluginName, withAdmissionCoverage(validatingwebhook.NewValidatingAdmissionWebhook, reqs))
	plugins.Register(mutatingwebhook.PluginName, withAdmissionCoverage(mutatingwebhook.NewMutatingWebhook, reqs))
	plugins.Register(validatingadmissionpolicy.PluginName, withAdmissionCoverage(validatingadmissionpolicy.NewPlugin, reqs))
	plugins.Register(mutatingadmissionpolicy.PluginName, withAdmissionCoverage(mutatingadmissionpolicy.NewPlugin, reqs))
	kubeoptions.RegisterAllAdmissionPlugins(plugins)

	// Fail loudly rather than silently drop a plugin added to the default registry.
	got, want := sets.New(plugins.Registered()...), sets.New(defaults.Registered()...)
	if !got.Equal(want) {
		t.Fatalf("admission plugin registry differs from the default: missing %v, unexpected %v",
			sets.List(want.Difference(got)), sets.List(got.Difference(want)))
	}
	return plugins
}

func withAdmissionCoverage[P interface {
	admission.Interface
	initializer.WantsAdmissionCoverage
}](newPlugin func(io.Reader) (P, error), reqs []coverage.Requirement) admission.Factory {
	return func(config io.Reader) (admission.Interface, error) {
		plugin, err := newPlugin(config)
		if err != nil {
			return nil, err
		}
		plugin.SetAdmissionCoverage(reqs)
		return plugin, nil
	}
}

// startCoverageWebhookServer starts a webhook that allows every request and
// returns its URL.
func startCoverageWebhookServer(t *testing.T) string {
	t.Helper()
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		review := &admissionreviewv1.AdmissionReview{}
		if err := json.NewDecoder(r.Body).Decode(review); err != nil || review.Request == nil {
			http.Error(w, fmt.Sprintf("invalid AdmissionReview: %v", err), http.StatusBadRequest)
			return
		}
		review.Response = &admissionreviewv1.AdmissionResponse{UID: review.Request.UID, Allowed: true}
		review.Request = nil
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(review); err != nil {
			t.Errorf("failed to write AdmissionReview response: %v", err)
		}
	}))
	server.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
	server.StartTLS()
	t.Cleanup(server.Close)
	return server.URL
}

func coverageNamespaceSelector(ns string) *metav1.LabelSelector {
	return &metav1.LabelSelector{MatchLabels: map[string]string{corev1.LabelMetadataName: ns}}
}

func coverageRules(scope coverageHookScope) []admissionregistrationv1.RuleWithOperations {
	return []admissionregistrationv1.RuleWithOperations{{
		Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update},
		Rule: admissionregistrationv1.Rule{
			APIGroups:   []string{""},
			APIVersions: []string{"v1"},
			Resources:   scope.resources,
		},
	}}
}

func coverageMatchConstraints(scope coverageHookScope) *admissionregistrationv1.MatchResources {
	return &admissionregistrationv1.MatchResources{
		ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{RuleWithOperations: coverageRules(scope)[0]}},
	}
}

// coverageBindingMatchResources has only selectors: the binding is scoped to ns
// but doesn't narrow the policy's resource rules.
func coverageBindingMatchResources(ns string, scope coverageHookScope) *admissionregistrationv1.MatchResources {
	return &admissionregistrationv1.MatchResources{
		NamespaceSelector: coverageNamespaceSelector(ns),
		ObjectSelector:    scope.objectSelector,
	}
}

func coverageValidatingWebhook(name, url, ns string, scope coverageHookScope) admissionregistrationv1.ValidatingWebhook {
	return admissionregistrationv1.ValidatingWebhook{
		Name:                    name,
		ClientConfig:            admissionregistrationv1.WebhookClientConfig{URL: &url, CABundle: localhostCert},
		Rules:                   coverageRules(scope),
		FailurePolicy:           ptr.To(admissionregistrationv1.Ignore),
		SideEffects:             ptr.To(admissionregistrationv1.SideEffectClassNone),
		AdmissionReviewVersions: []string{"v1"},
		NamespaceSelector:       coverageNamespaceSelector(ns),
		ObjectSelector:          scope.objectSelector,
	}
}

func coverageVAP(name string, scope coverageHookScope) *admissionregistrationv1.ValidatingAdmissionPolicy {
	return &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			MatchConstraints: coverageMatchConstraints(scope),
			Validations:      []admissionregistrationv1.Validation{{Expression: "true"}},
			FailurePolicy:    ptr.To(admissionregistrationv1.Ignore),
		},
	}
}

func coverageVAPBinding(name, policyName, ns string, scope coverageHookScope) *admissionregistrationv1.ValidatingAdmissionPolicyBinding {
	return &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        policyName,
			ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
			MatchResources:    coverageBindingMatchResources(ns, scope),
		},
	}
}

func coverageMAP(name string, scope coverageHookScope) *admissionregistrationv1.MutatingAdmissionPolicy {
	return &admissionregistrationv1.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
			MatchConstraints: coverageMatchConstraints(scope),
			Mutations: []admissionregistrationv1.Mutation{{
				PatchType:          admissionregistrationv1.PatchTypeApplyConfiguration,
				ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{Expression: `Object{metadata: Object.metadata{labels: {"coverage-mutated": "true"}}}`},
			}},
			FailurePolicy:      ptr.To(admissionregistrationv1.Ignore),
			ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
		},
	}
}

func coverageMAPBinding(name, policyName, ns string, scope coverageHookScope) *admissionregistrationv1.MutatingAdmissionPolicyBinding {
	return &admissionregistrationv1.MutatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{
			PolicyName:     policyName,
			MatchResources: coverageBindingMatchResources(ns, scope),
		},
	}
}

// coverageHookClient is the subset of a typed client used to manage hooks.
type coverageHookClient[T any] interface {
	Get(ctx context.Context, name string, opts metav1.GetOptions) (T, error)
	Create(ctx context.Context, obj T, opts metav1.CreateOptions) (T, error)
	Update(ctx context.Context, obj T, opts metav1.UpdateOptions) (T, error)
	Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error
}

// createOrUpdateCoverageHook updates hooks in place, rather than recreating
// them, so that the server never observes a moment without the hook, which
// would let requests through for the wrong reason.
func createOrUpdateCoverageHook[T metav1.Object](ctx context.Context, t *testing.T, c coverageHookClient[T], obj T) {
	t.Helper()
	existing, err := c.Get(ctx, obj.GetName(), metav1.GetOptions{})
	switch {
	case apierrors.IsNotFound(err):
		_, err = c.Create(ctx, obj, metav1.CreateOptions{})
		if err == nil {
			t.Cleanup(func() {
				// t.Context() is already canceled when cleanups run.
				if err := c.Delete(context.Background(), obj.GetName(), metav1.DeleteOptions{}); err != nil {
					t.Errorf("failed to delete %s: %v", obj.GetName(), err)
				}
			})
		}
	case err == nil:
		obj.SetResourceVersion(existing.GetResourceVersion())
		_, err = c.Update(ctx, obj, metav1.UpdateOptions{})
	}
	if err != nil {
		t.Fatalf("failed to create or update %s: %v", obj.GetName(), err)
	}
}

func createCoverageNamespaceAndPods(t *testing.T, client clientset.Interface, ns string) {
	t.Helper()
	ctx := t.Context()
	if _, err := client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create namespace: %v", err)
	}
	t.Cleanup(func() {
		if err := client.CoreV1().Namespaces().Delete(context.Background(), ns, metav1.DeleteOptions{}); err != nil {
			t.Errorf("failed to delete namespace %s: %v", ns, err)
		}
	})
	for name, app := range map[string]string{coveragePodX: "x", coveragePodY: "y"} {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"app": app}},
			Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "c", Image: "registry.k8s.io/pause"}}},
		}
		if _, err := client.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create pod %s: %v", name, err)
		}
	}
}

// updateCoveragePodStatus sends the coverage target request, UPDATE pods/status.
// The update is unconditional, so a concurrent write can't turn the expected
// response into a conflict.
func updateCoveragePodStatus(ctx context.Context, client clientset.Interface, ns, name string) error {
	pod, err := client.CoreV1().Pods(ns).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	pod.ResourceVersion = ""
	pod.Status.Message = "probe " + time.Now().Format(time.RFC3339Nano)
	_, err = client.CoreV1().Pods(ns).UpdateStatus(ctx, pod, metav1.UpdateOptions{})
	return err
}

// updateCoveragePod sends a coverage source request, UPDATE pods, unconditionally
// like updateCoveragePodStatus.
func updateCoveragePod(ctx context.Context, client clientset.Interface, ns, name string) error {
	pod, err := client.CoreV1().Pods(ns).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	pod.ResourceVersion = ""
	metav1.SetMetaDataAnnotation(&pod.ObjectMeta, "coverage-probe", time.Now().Format(time.RFC3339Nano))
	_, err = client.CoreV1().Pods(ns).Update(ctx, pod, metav1.UpdateOptions{})
	return err
}

// waitForPodStatusUpdate waits until UPDATE pods/status of the named pod is
// forbidden, or allowed, since hook configuration changes reach admission
// asynchronously. It returns the last error. Any other error fails the test.
func waitForPodStatusUpdate(t *testing.T, client clientset.Interface, ns, name string, forbidden bool) error {
	t.Helper()
	return waitForPodStatusUpdates(t, client, ns, map[string]bool{name: forbidden})[name]
}

// waitForPodStatusUpdates is like waitForPodStatusUpdate, for several pods at
// once: it waits until, in the same round of requests, UPDATE pods/status of
// each pod in want is forbidden (true) or allowed (false). It returns the last
// error for each pod.
func waitForPodStatusUpdates(t *testing.T, client clientset.Interface, ns string, want map[string]bool) map[string]error {
	t.Helper()
	pods := sets.List(sets.KeySet(want))
	// Probe the pods expected to be forbidden first: configuration changes
	// only move forward, so an allowed response that follows a rejection in
	// the same round comes from a configuration at least as new.
	sort.SliceStable(pods, func(i, j int) bool { return want[pods[i]] && !want[pods[j]] })
	lastErrs := make(map[string]error, len(pods))
	err := wait.PollUntilContextTimeout(t.Context(), 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		done := true
		for _, name := range pods {
			err := updateCoveragePodStatus(ctx, client, ns, name)
			if err != nil && ctx.Err() != nil {
				// The deadline cut the request short; keep the last
				// response from the server for the failure message.
				return false, nil
			}
			lastErrs[name] = err
			if err != nil && !apierrors.IsForbidden(err) {
				return false, err
			}
			done = done && apierrors.IsForbidden(err) == want[name]
		}
		return done, nil
	})
	if err != nil {
		var got []string
		for _, name := range pods {
			lastErr, ok := lastErrs[name]
			if !ok {
				got = append(got, fmt.Sprintf("%s: want forbidden=%v, no response", name, want[name]))
				continue
			}
			got = append(got, fmt.Sprintf("%s: want forbidden=%v, last error: %v", name, want[name], lastErr))
		}
		t.Fatalf("waiting for UPDATE pods/status: %v; %s", err, strings.Join(got, "; "))
	}
	return lastErrs
}

// expectCoverageRejection checks that err rejects UPDATE pods/status for a
// coverage violation.
func expectCoverageRejection(t *testing.T, err error, wantMessages ...string) {
	t.Helper()
	if !apierrors.IsForbidden(err) {
		t.Fatalf("UPDATE pods/status returned %v, want a Forbidden error", err)
	}
	for _, want := range append([]string{"UPDATE pods/status", coverageHint}, wantMessages...) {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("rejection %q does not contain %q", err.Error(), want)
		}
	}
}
