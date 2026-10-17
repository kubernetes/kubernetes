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

package admissionequivalents

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/testapigroup/install"
	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/equivalents"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	"k8s.io/component-base/metrics/testutil"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/controlplane"
	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
	carpstore "k8s.io/kubernetes/pkg/registry/testapigroup/carp/storage"
	testapigrouprest "k8s.io/kubernetes/pkg/registry/testapigroup/rest"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils"
	"k8s.io/utils/ptr"
)

func init() {
	install.Install(legacyscheme.Scheme)
}

const (
	// caseLabel scopes every binding and webhook to the namespaces of a single test case, so
	// cases neither interfere with each other nor need to wait for cleanup to propagate.
	caseLabel = "admissionequivalents.test/case"

	// denyMarker makes policies deny a request: as a label key on carps (the only part of a carp
	// that survives an update of the resource itself), or as status.message on carps/turbo (the
	// only part that survives an update of a status-like subresource).
	denyMarker = "deny"
)

var carpsGVR = testapigroupv1.SchemeGroupVersion.WithResource("carps")

// turboREST is a status-like subresource that declares carps (CREATE, UPDATE) and carps/status
// as its admission equivalents, standing in for a real subresource (like pods/resize) that can
// make changes also reachable through other endpoints.
type turboREST struct {
	*carpstore.StatusREST
}

var _ rest.AdmissionEquivalentsProvider = &turboREST{}

func (*turboREST) AdmissionEquivalents() []admission.Equivalent {
	return []admission.Equivalent{
		{Subresource: "", Operations: []admission.Operation{admission.Create, admission.Update}},
		{Subresource: "status"},
	}
}

// plainREST is identical to turboREST but declares no admission equivalents. It is the control
// showing that enforcement is opt-in per endpoint.
type plainREST struct {
	*carpstore.StatusREST
}

// equivalentsStorageProvider serves carps/turbo and carps/plain in addition to the regular
// testapigroup storage.
type equivalentsStorageProvider struct {
	testapigrouprest.RESTStorageProvider
}

func (p equivalentsStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	info, err := p.RESTStorageProvider.NewRESTStorage(apiResourceConfigSource, restOptionsGetter)
	if err != nil {
		return genericapiserver.APIGroupInfo{}, err
	}
	storage := info.VersionedResourcesStorageMap[testapigroupv1.SchemeGroupVersion.Version]
	status, ok := storage["carps/status"].(*carpstore.StatusREST)
	if !ok {
		return genericapiserver.APIGroupInfo{}, fmt.Errorf("carps/status storage is %T, want *storage.StatusREST; is %s enabled?", storage["carps/status"], carpsGVR)
	}
	storage["carps/turbo"] = &turboREST{status}
	storage["carps/plain"] = &plainREST{status}
	return info, nil
}

type testEnv struct {
	ctx     context.Context
	client  kubernetes.Interface
	dynamic dynamic.Interface
	// auditLog receives Metadata-level ResponseComplete events for carps/turbo only.
	auditLog string
}

const auditPolicy = `
apiVersion: audit.k8s.io/v1
kind: Policy
omitStages: ["RequestReceived", "ResponseStarted"]
rules:
- level: Metadata
  resources:
  - group: ` + testapigroupv1.GroupName + `
    resources: ["carps/turbo"]
- level: None
`

func TestAdmissionEquivalents(t *testing.T) {
	// Sanity check. Not protected against concurrent access, but integration tests also run
	// with race detection, which would catch that.
	if controlplane.AdditionalStorageProvidersForTests != nil {
		t.Fatal("cannot set AdditionalStorageProvidersForTests, already set")
	}
	t.Cleanup(func() {
		controlplane.AdditionalStorageProvidersForTests = nil
	})
	controlplane.AdditionalStorageProvidersForTests = func(client *kubernetes.Clientset) []controlplaneapiserver.RESTStorageProvider {
		return []controlplaneapiserver.RESTStorageProvider{
			equivalentsStorageProvider{testapigrouprest.RESTStorageProvider{NamespaceClient: client.CoreV1().Namespaces()}},
		}
	}

	auditDir := t.TempDir()
	auditPolicyFile := filepath.Join(auditDir, "policy.yaml")
	if err := os.WriteFile(auditPolicyFile, []byte(auditPolicy), 0o600); err != nil {
		t.Fatal(err)
	}
	auditLog := filepath.Join(auditDir, "audit.log")

	flags := append(framework.DefaultTestServerFlags(),
		"--runtime-config="+testapigroupv1.SchemeGroupVersion.String()+"=true",
		"--audit-policy-file="+auditPolicyFile,
		"--audit-log-path="+auditLog,
		// Events are written before the response is sent, so they can be read right after.
		"--audit-log-mode=blocking",
		"--audit-log-version="+auditv1.SchemeGroupVersion.String(),
	)
	server := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), flags, framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	env := testEnv{
		ctx:      t.Context(),
		client:   kubernetes.NewForConfigOrDie(server.ClientConfig),
		dynamic:  dynamic.NewForConfigOrDie(server.ClientConfig),
		auditLog: auditLog,
	}

	t.Run("uncovered policy rejects the equivalent until it is covered", env.testPolicyCoverage)
	t.Run("same-operation equivalent", env.testSameOperationEquivalent)
	t.Run("wildcard subresource rule covers the source", env.testWildcardSubresource)
	t.Run("excluded source is a violation", env.testExcludedSource)
	t.Run("audit and warn only binding is enforced", env.testNonDenyBinding)
	t.Run("object selector", env.testObjectSelector)
	t.Run("namespace selector", env.testNamespaceSelector)
	t.Run("fail-open validating webhook is enforced", env.testValidatingWebhook)
	t.Run("fail-open mutating webhook is enforced", env.testMutatingWebhook)
	t.Run("fail-open mutating admission policy is enforced", env.testMutatingPolicy)
}

func (env testEnv) testPolicyCoverage(t *testing.T) {
	ns, carps := env.newCase(t, "coverage")

	policy, binding := newVAP("coverage", ns, []string{"carps"}, admissionregistrationv1.Update)
	rejectionsBefore := env.coverageRejectionsTotal(t, "ValidatingAdmissionPolicy", policy.Name)
	env.createVAP(t, policy, binding)

	carps.requireCoverageErrorEventually("c", vapViolation(policy.Name, binding.Name, "carps (UPDATE)"))

	// PATCH is admitted as UPDATE, so it must not be a way around the check.
	_, err := carps.client.Patch(env.ctx, "c", types.MergePatchType, []byte(`{"status":{"message":"patched"}}`), metav1.PatchOptions{}, "turbo")
	requireCoverageError(t, err, vapViolation(policy.Name, binding.Name, "carps (UPDATE)"))

	// The policy is loaded and evaluated for the resource it names.
	if _, err := carps.update("c", "", withLabel("benign")); err != nil {
		t.Fatalf("benign update of carps: %v", err)
	}
	_, err = carps.update("c", "", withLabel(denyMarker))
	requirePolicyDenial(t, err, policy.Name)

	// Without a declaration there is nothing to enforce, even for an otherwise identical endpoint.
	if _, err := carps.update("c", "plain", withStatusMessage(denyMarker)); err != nil {
		t.Fatalf("update of carps/plain, which declares no equivalents: %v", err)
	}

	// One rejection ends requireCoverageErrorEventually; the other is the PATCH.
	if got := env.coverageRejectionsTotal(t, "ValidatingAdmissionPolicy", policy.Name) - rejectionsBefore; got != 2 {
		t.Errorf("coverage rejections metric for ValidatingAdmissionPolicy %q increased by %v, want 2", policy.Name, got)
	}

	env.requireAuditAnnotation(t, ns, "c", vapViolation(policy.Name, binding.Name, "carps (UPDATE)"))

	// Covering the source resolves the violation and subjects it to the policy.
	policies := env.client.AdmissionregistrationV1().ValidatingAdmissionPolicies()
	if err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		current, err := policies.Get(env.ctx, policy.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		current.Spec.MatchConstraints.ResourceRules[0].Resources = []string{"carps", "carps/turbo"}
		_, err = policies.Update(env.ctx, current, metav1.UpdateOptions{})
		return err
	}); err != nil {
		t.Fatal(err)
	}
	err = eventually(t, env.ctx, "policy denial on carps/turbo", isPolicyDenial, func() error {
		_, err := carps.update("c", "turbo", withStatusMessage(denyMarker))
		return err
	})
	requirePolicyDenial(t, err, policy.Name)
	if _, err := carps.update("c", "turbo", withStatusMessage("benign")); err != nil {
		t.Fatalf("benign update of covered carps/turbo: %v", err)
	}
}

// testSameOperationEquivalent checks the {Subresource: "status"} declaration, which has no
// operations and so expands to the incoming request's operation.
func (env testEnv) testSameOperationEquivalent(t *testing.T) {
	ns, carps := env.newCase(t, "status")

	policy, binding := newVAP("status", ns, []string{"carps/status"}, admissionregistrationv1.Update)
	env.createVAP(t, policy, binding)

	carps.requireCoverageErrorEventually("c", vapViolation(policy.Name, binding.Name, "carps/status (UPDATE)"))

	// The policy is loaded and evaluated for the subresource it names.
	_, err := carps.update("c", "status", withStatusMessage(denyMarker))
	requirePolicyDenial(t, err, policy.Name)
}

func (env testEnv) testWildcardSubresource(t *testing.T) {
	ns, carps := env.newCase(t, "wildcard")

	policy, binding := newVAP("wildcard", ns, []string{"carps", "carps/*"}, admissionregistrationv1.Update)
	env.createVAP(t, policy, binding)

	// A policy denial rather than a coverage error proves carps/turbo is covered by carps/*.
	err := eventually(t, env.ctx, "policy denial on carps/turbo", isPolicyDenial, func() error {
		_, err := carps.update("c", "turbo", withStatusMessage(denyMarker))
		return err
	})
	requirePolicyDenial(t, err, policy.Name)
	if _, err := carps.update("c", "turbo", withStatusMessage("benign")); err != nil {
		t.Fatalf("benign update of covered carps/turbo: %v", err)
	}
}

func (env testEnv) testExcludedSource(t *testing.T) {
	ns, carps := env.newCase(t, "exclude")

	policy, binding := newVAP("exclude", ns, []string{"carps", "carps/*"}, admissionregistrationv1.Update)
	policy.Spec.MatchConstraints.ExcludeResourceRules = []admissionregistrationv1.NamedRuleWithOperations{
		carpsRule([]string{"carps/turbo"}, admissionregistrationv1.OperationAll),
	}
	env.createVAP(t, policy, binding)

	carps.requireCoverageErrorEventually("c", vapViolation(policy.Name, binding.Name, "carps (UPDATE)"))
}

func (env testEnv) testNonDenyBinding(t *testing.T) {
	ns, carps := env.newCase(t, "nondeny")

	policy, binding := newVAP("nondeny", ns, []string{"carps"}, admissionregistrationv1.Update)
	binding.Spec.ValidationActions = []admissionregistrationv1.ValidationAction{admissionregistrationv1.Warn, admissionregistrationv1.Audit}
	env.createVAP(t, policy, binding)

	// Skipping a Warn or Audit binding would hide exactly the requests its owner asked to see.
	carps.requireCoverageErrorEventually("c", vapViolation(policy.Name, binding.Name, "carps (UPDATE)"))

	if _, err := carps.update("c", "", withLabel(denyMarker)); err != nil {
		t.Fatalf("update of carps under a non-Deny binding: %v", err)
	}
}

func (env testEnv) testObjectSelector(t *testing.T) {
	ns, carps := env.newCase(t, "objectselector")
	carps.create("selected", map[string]string{"probe": "true"})
	carps.create("unselected", nil)

	policy, binding := newVAP("objectselector", ns, []string{"carps"}, admissionregistrationv1.Update)
	binding.Spec.MatchResources.ObjectSelector = &metav1.LabelSelector{MatchLabels: map[string]string{"probe": "true"}}
	env.createVAP(t, policy, binding)

	carps.requireCoverageErrorEventually("selected", vapViolation(policy.Name, binding.Name, "carps (UPDATE)"))

	if _, err := carps.update("unselected", "turbo", withStatusMessage("hello")); err != nil {
		t.Fatalf("update of carps/turbo for an object the binding does not select: %v", err)
	}
}

func (env testEnv) testNamespaceSelector(t *testing.T) {
	selected, selectedCarps := env.newCase(t, "nsselector-selected")
	_, unselectedCarps := env.newCase(t, "nsselector-unselected")

	policy, binding := newVAP("nsselector", selected, []string{"carps"}, admissionregistrationv1.Update)
	env.createVAP(t, policy, binding)

	selectedCarps.requireCoverageErrorEventually("c", vapViolation(policy.Name, binding.Name, "carps (UPDATE)"))

	if _, err := unselectedCarps.update("c", "turbo", withStatusMessage("hello")); err != nil {
		t.Fatalf("update of carps/turbo in a namespace the binding does not select: %v", err)
	}
}

// Nothing listens on port 1, so every call fails and failurePolicy Ignore admits the request.
// A fail-open webhook still has to see the request: if it is up, it expects to.
const (
	unreachableWebhookURL = "https://127.0.0.1:1/"
	webhookName           = "carps.admissionequivalents.example.com"
)

func (env testEnv) testValidatingWebhook(t *testing.T) {
	ns, carps := env.newCase(t, "validatingwebhook")

	cfg := &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "admissionequivalents-validatingwebhook"},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name:                    webhookName,
			ClientConfig:            admissionregistrationv1.WebhookClientConfig{URL: ptr.To(unreachableWebhookURL)},
			Rules:                   []admissionregistrationv1.RuleWithOperations{carpsRule([]string{"carps"}, admissionregistrationv1.OperationAll).RuleWithOperations},
			FailurePolicy:           ptr.To(admissionregistrationv1.Ignore),
			SideEffects:             ptr.To(admissionregistrationv1.SideEffectClassNone),
			AdmissionReviewVersions: []string{"v1"},
			TimeoutSeconds:          ptr.To[int32](1),
			NamespaceSelector:       caseSelector(ns),
		}},
	}
	if _, err := env.client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(env.ctx, cfg, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	env.requireFailOpenWebhookEnforced(t, carps, "ValidatingAdmissionWebhook", cfg.Name)
}

func (env testEnv) testMutatingWebhook(t *testing.T) {
	ns, carps := env.newCase(t, "mutatingwebhook")

	cfg := &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "admissionequivalents-mutatingwebhook"},
		Webhooks: []admissionregistrationv1.MutatingWebhook{{
			Name:                    webhookName,
			ClientConfig:            admissionregistrationv1.WebhookClientConfig{URL: ptr.To(unreachableWebhookURL)},
			Rules:                   []admissionregistrationv1.RuleWithOperations{carpsRule([]string{"carps"}, admissionregistrationv1.OperationAll).RuleWithOperations},
			FailurePolicy:           ptr.To(admissionregistrationv1.Ignore),
			SideEffects:             ptr.To(admissionregistrationv1.SideEffectClassNone),
			AdmissionReviewVersions: []string{"v1"},
			TimeoutSeconds:          ptr.To[int32](1),
			NamespaceSelector:       caseSelector(ns),
		}},
	}
	if _, err := env.client.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(env.ctx, cfg, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	env.requireFailOpenWebhookEnforced(t, carps, "MutatingAdmissionWebhook", cfg.Name)
}

// requireFailOpenWebhookEnforced checks that webhookName in configuration cfg, which matches
// carps with operations "*" and is unreachable, blocks carps/turbo but admits carps.
func (env testEnv) requireFailOpenWebhookEnforced(t *testing.T, carps carpClient, plugin, cfg string) {
	t.Helper()
	// Operations "*" match CREATE first, which is the first declared target.
	carps.requireCoverageErrorEventually("c", fmt.Sprintf("%s %q (configuration %q) applies to carps (CREATE) but not carps/turbo (UPDATE)", plugin, webhookName, cfg))

	if _, err := carps.update("c", "", withLabel("benign")); err != nil {
		t.Fatalf("update of carps with an unreachable fail-open webhook: %v", err)
	}
}

func (env testEnv) testMutatingPolicy(t *testing.T) {
	ns, carps := env.newCase(t, "map")

	policy := &admissionregistrationv1.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "admissionequivalents-map"},
		Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{carpsRule([]string{"carps"}, admissionregistrationv1.Update)},
			},
			Mutations: []admissionregistrationv1.Mutation{{
				PatchType: admissionregistrationv1.PatchTypeJSONPatch,
				JSONPatch: &admissionregistrationv1.JSONPatch{
					Expression: `[JSONPatch{op: "add", path: "/metadata/annotations", value: {"mutated": "true"}}]`,
				},
			}},
			FailurePolicy:      ptr.To(admissionregistrationv1.Ignore),
			ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
		},
	}
	binding := &admissionregistrationv1.MutatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "admissionequivalents-map"},
		Spec: admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{
			PolicyName:     policy.Name,
			MatchResources: &admissionregistrationv1.MatchResources{NamespaceSelector: caseSelector(ns)},
		},
	}
	if _, err := env.client.AdmissionregistrationV1().MutatingAdmissionPolicies().Create(env.ctx, policy, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := env.client.AdmissionregistrationV1().MutatingAdmissionPolicyBindings().Create(env.ctx, binding, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	carps.requireCoverageErrorEventually("c", fmt.Sprintf("MutatingAdmissionPolicy %q (binding %q) applies to carps (UPDATE) but not carps/turbo (UPDATE)", policy.Name, binding.Name))

	// The policy is loaded and mutates the resource it names.
	updated, err := carps.update("c", "", withLabel("benign"))
	if err != nil {
		t.Fatalf("update of carps: %v", err)
	}
	if got := updated.GetAnnotations()["mutated"]; got != "true" {
		t.Errorf("carps update was not mutated by %s: annotations %v", policy.Name, updated.GetAnnotations())
	}
}

func vapViolation(policy, binding, target string) string {
	return fmt.Sprintf("ValidatingAdmissionPolicy %q (binding %q) applies to %s but not carps/turbo (UPDATE)", policy, binding, target)
}

func carpsRule(resources []string, ops ...admissionregistrationv1.OperationType) admissionregistrationv1.NamedRuleWithOperations {
	return admissionregistrationv1.NamedRuleWithOperations{
		RuleWithOperations: admissionregistrationv1.RuleWithOperations{
			Operations: ops,
			Rule: admissionregistrationv1.Rule{
				APIGroups:   []string{testapigroupv1.GroupName},
				APIVersions: []string{testapigroupv1.SchemeGroupVersion.Version},
				Resources:   resources,
			},
		},
	}
}

func caseSelector(ns string) *metav1.LabelSelector {
	return &metav1.LabelSelector{MatchLabels: map[string]string{caseLabel: ns}}
}

// newVAP returns a Fail policy matching resources that denies denyMarker, and a Deny binding
// scoped to namespace ns.
func newVAP(name, ns string, resources []string, ops ...admissionregistrationv1.OperationType) (*admissionregistrationv1.ValidatingAdmissionPolicy, *admissionregistrationv1.ValidatingAdmissionPolicyBinding) {
	name = "admissionequivalents-" + name
	policy := &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy: ptr.To(admissionregistrationv1.Fail),
			MatchConstraints: &admissionregistrationv1.MatchResources{
				ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{carpsRule(resources, ops...)},
			},
			Validations: []admissionregistrationv1.Validation{{
				Expression: fmt.Sprintf("!(has(object.metadata.labels) && %[1]q in object.metadata.labels) && "+
					"!(has(object.status) && has(object.status.message) && object.status.message == %[1]q)", denyMarker),
				Message: "deny marker present",
			}},
		},
	}
	binding := &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName:        name,
			ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
			MatchResources:    &admissionregistrationv1.MatchResources{NamespaceSelector: caseSelector(ns)},
		},
	}
	return policy, binding
}

func (env testEnv) createVAP(t *testing.T, policy *admissionregistrationv1.ValidatingAdmissionPolicy, binding *admissionregistrationv1.ValidatingAdmissionPolicyBinding) {
	t.Helper()
	if _, err := env.client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Create(env.ctx, policy, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := env.client.AdmissionregistrationV1().ValidatingAdmissionPolicyBindings().Create(env.ctx, binding, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
}

// newCase creates a namespace for a test case, and a carp "c" in it.
func (env testEnv) newCase(t *testing.T, name string) (string, carpClient) {
	t.Helper()
	name = "admissionequivalents-" + name
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{caseLabel: name}}}
	if _, err := env.client.CoreV1().Namespaces().Create(env.ctx, ns, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	carps := carpClient{t: t, ctx: env.ctx, client: env.dynamic.Resource(carpsGVR).Namespace(name)}
	carps.create("c", nil)
	return name, carps
}

// coverageRejectionsTotal returns the value of the coverage rejection metric for carps/turbo, as
// served by /metrics.
func (env testEnv) coverageRejectionsTotal(t *testing.T, plugin, hook string) float64 {
	t.Helper()
	raw, err := env.client.CoreV1().RESTClient().Get().AbsPath("/metrics").DoRaw(env.ctx)
	if err != nil {
		t.Fatal(err)
	}
	families, err := testutil.TextToMetricFamilies(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("parsing /metrics: %v", err)
	}
	const name = "apiserver_admission_equivalent_coverage_rejections_total"
	family, ok := families[name]
	if !ok {
		return 0
	}
	labels := map[string]string{"plugin": plugin, "name": hook, "resource": carpsGVR.GroupResource().String(), "subresource": "turbo"}
	var total float64
	for _, m := range family.GetMetric() {
		if testutil.LabelsMatch(m, labels) {
			total += m.GetCounter().GetValue()
		}
	}
	return total
}

// requireAuditAnnotation checks that there are 403 audit events for carps/turbo of the named
// carp for both update and patch, and that every such event carries the coverage annotation
// with want.
func (env testEnv) requireAuditAnnotation(t *testing.T, ns, name, want string) {
	t.Helper()
	f, err := os.Open(env.auditLog)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	report, err := utils.CheckAuditLinesFiltered(f, nil, auditv1.SchemeGroupVersion, func(key, _ string) bool {
		return key == equivalents.AuditAnnotationKey
	})
	if err != nil {
		t.Fatalf("reading audit log: %v", err)
	}

	uri := fmt.Sprintf("/apis/%s/namespaces/%s/carps/%s/turbo", testapigroupv1.SchemeGroupVersion, ns, name)
	verbs := sets.New[string]()
	for _, event := range report.AllEvents {
		if event.RequestURI != uri || event.Code != http.StatusForbidden {
			continue
		}
		verbs.Insert(event.Verb)
		value, ok := event.CustomAuditAnnotations[equivalents.AuditAnnotationKey]
		if !ok {
			t.Errorf("%s %s audit event has no %s annotation", event.Verb, uri, equivalents.AuditAnnotationKey)
			continue
		}
		if value != want {
			t.Errorf("%s %s audit event: %s annotation is %q, want %q", event.Verb, uri, equivalents.AuditAnnotationKey, value, want)
		}
	}
	if wantVerbs := sets.New("update", "patch"); !verbs.IsSuperset(wantVerbs) {
		t.Errorf("%d audit events for %s have verbs %v, want at least %v", http.StatusForbidden, uri, sets.List(verbs), sets.List(wantVerbs))
	}
}

type carpClient struct {
	t      *testing.T
	ctx    context.Context
	client dynamic.ResourceInterface
}

func (c carpClient) create(name string, labels map[string]string) {
	c.t.Helper()
	obj := &unstructured.Unstructured{}
	obj.SetAPIVersion(testapigroupv1.SchemeGroupVersion.String())
	obj.SetKind("Carp")
	obj.SetName(name)
	obj.SetLabels(labels)
	if _, err := c.client.Create(c.ctx, obj, metav1.CreateOptions{}); err != nil {
		c.t.Fatalf("creating carp %s: %v", name, err)
	}
}

type mutation func(*unstructured.Unstructured) error

// update applies mutate to the current carp and writes it to subresource ("" for the carp
// itself). It returns only the error of the write; everything else is fatal.
func (c carpClient) update(name, subresource string, mutate mutation) (*unstructured.Unstructured, error) {
	c.t.Helper()
	obj, err := c.client.Get(c.ctx, name, metav1.GetOptions{})
	if err != nil {
		c.t.Fatalf("getting carp %s: %v", name, err)
	}
	if err := mutate(obj); err != nil {
		c.t.Fatalf("mutating carp %s: %v", name, err)
	}
	var subresources []string
	if subresource != "" {
		subresources = []string{subresource}
	}
	return c.client.Update(c.ctx, obj, metav1.UpdateOptions{}, subresources...)
}

func withLabel(key string) mutation {
	return func(obj *unstructured.Unstructured) error {
		labels := obj.GetLabels()
		if labels == nil {
			labels = map[string]string{}
		}
		labels[key] = "true"
		obj.SetLabels(labels)
		return nil
	}
}

func withStatusMessage(msg string) mutation {
	return func(obj *unstructured.Unstructured) error {
		return unstructured.SetNestedField(obj.Object, msg, "status", "message")
	}
}

// eventually retries attempt until done accepts its result, and returns that result. Admission
// configuration propagates asynchronously, so every case waits for a positive signal that its
// configuration is in effect before asserting anything else.
//
// attempt may call t.Fatal (carpClient does). That is only safe because PollUntilContextTimeout
// runs the condition on the calling goroutine, and t.Fatal's runtime.Goexit is not a panic, so
// the poll loop's crash handler does not intercept it.
func eventually(t *testing.T, ctx context.Context, what string, done func(error) bool, attempt func() error) error {
	t.Helper()
	var last error
	if err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
		last = attempt()
		return done(last), nil
	}); err != nil {
		t.Fatalf("timed out waiting for %s; last result: %v", what, last)
	}
	return last
}

const coveragePrefix = "request to carps/turbo is not covered by admission hooks that apply to its admission equivalents: "

func isCoverageError(err error) bool {
	return apierrors.IsForbidden(err) && strings.Contains(err.Error(), coveragePrefix)
}

func isPolicyDenial(err error) bool {
	return apierrors.IsInvalid(err) && strings.Contains(err.Error(), "denied request")
}

func requireCoverageError(t *testing.T, err error, violation string) {
	t.Helper()
	if !isCoverageError(err) {
		t.Fatalf("want coverage error, got: %v", err)
	}
	if !strings.Contains(err.Error(), violation) {
		t.Errorf("coverage error message does not contain %q: %v", violation, err)
	}
}

// requireCoverageErrorEventually waits for an update of carps/turbo of the named carp to be
// rejected for coverage, then checks that the rejection names violation.
func (c carpClient) requireCoverageErrorEventually(name, violation string) {
	c.t.Helper()
	err := eventually(c.t, c.ctx, "coverage error on carps/turbo", isCoverageError, func() error {
		_, err := c.update(name, "turbo", withStatusMessage("hello"))
		return err
	})
	requireCoverageError(c.t, err, violation)
}

func requirePolicyDenial(t *testing.T, err error, policy string) {
	t.Helper()
	if !isPolicyDenial(err) || !strings.Contains(err.Error(), fmt.Sprintf("'%s'", policy)) {
		t.Fatalf("want denial by ValidatingAdmissionPolicy %q, got: %v", policy, err)
	}
}
