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

package cost

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/ktesting"
	"k8s.io/utils/ptr"
)

const (
	admissionPolicyScenarioKey  = "bench.k8s.io/scenario"
	admissionPolicyPodsPath     = "/api/v1/namespaces/" + metav1.NamespaceDefault + "/pods"
	admissionPolicyVAPPath      = "/apis/admissionregistration.k8s.io/v1/validatingadmissionpolicies"
	admissionPolicyMAPPath      = "/apis/admissionregistration.k8s.io/v1/mutatingadmissionpolicies"
	admissionPolicyDryRunQuery  = "dryRun=All&fieldManager=bench"
	admissionPolicyReadyTimeout = time.Minute

	// The package defaults to -benchtime=1x and allocations are measured
	// process-wide, so a single request is easily dominated by background
	// apiserver work. All admission policy requests are dry-run, so a fixed
	// batch of them is sent per iteration and the per-request average is
	// reported.
	admissionPolicyRequestsPerOp = 50

	admissionPolicyNone         = "none"
	admissionPolicyVAP          = "vap"
	admissionPolicyMAPApply     = "map-apply"
	admissionPolicyMAPJSONPatch = "map-jsonpatch"
)

// CEL evaluation cost does not depend on the wire format, so only JSON is
// measured to keep the number of apiserver instances started low.
var admissionPolicyFormats = []contentType{{name: "json", accept: acceptJSON}}

// BenchmarkAdmissionPolicyRequestCost measures per-request heap allocations
// for kube-apiserver requests evaluated by CEL-based admission policies,
// using the same harness as BenchmarkRequestCost:
//
//   - create of the exemplar Pod validated by a ValidatingAdmissionPolicy or
//     mutated by a MutatingAdmissionPolicy (applyConfiguration and JSONPatch),
//     with an unmatched Pod as baseline, scaled by the number of containers
//     iterated by the policy expressions (CEL evaluation);
//   - create of a ValidatingAdmissionPolicy / MutatingAdmissionPolicy, whose
//     validation parses, type-checks and cost-estimates every expression
//     (CEL compilation).
//
// All requests are dry-run so the same request can be repeated without
// changing cluster state.
func BenchmarkAdmissionPolicyRequestCost(b *testing.B) {
	var scenarios []requestScenario
	for _, n := range []int{1, 10, 100} {
		for _, policy := range []string{admissionPolicyNone, admissionPolicyVAP, admissionPolicyMAPApply, admissionPolicyMAPJSONPatch} {
			scenarios = append(scenarios, admissionPolicyScenario(b,
				fmt.Sprintf("create policy=%s containers=%d", policy, n),
				admissionPolicyPodsPath, admissionPolicyPod(policy, n), setupAdmissionPolicy(policy)))
		}
	}
	scenarios = append(scenarios,
		admissionPolicyScenario(b, "ValidatingAdmissionPolicy",
			admissionPolicyVAPPath, benchVAP("bench-vap-dryrun"), nil),
		admissionPolicyScenario(b, "MutatingAdmissionPolicy applyConfiguration",
			admissionPolicyMAPPath, benchMAPApply("bench-map-apply-dryrun"), nil),
		admissionPolicyScenario(b, "MutatingAdmissionPolicy jsonPatch",
			admissionPolicyMAPPath, benchMAPJSONPatch("bench-map-jsonpatch-dryrun"), nil),
	)
	benchmarkRequestScenarios(b, scenarios)
}

// admissionPolicyScenario returns a dry-run POST of obj to path. setup, if
// set, installs fixtures before the shared harness warms up and measures.
func admissionPolicyScenario(b *testing.B, options, path string, obj kruntime.Object, setup func(ktesting.TContext, http.Handler)) requestScenario {
	return requestScenario{
		method:             "POST",
		options:            options,
		httpMethod:         http.MethodPost,
		path:               path,
		query:              admissionPolicyDryRunQuery,
		acceptContentTypes: admissionPolicyFormats,
		body:               staticObjectBody(b, obj),
		requestsPerOp:      admissionPolicyRequestsPerOp,
		setup:              setup,
	}
}

// setupAdmissionPolicy returns a setup function that creates the admission
// policy and binding for the given scenario and waits until it is enforced.
func setupAdmissionPolicy(policy string) func(ktesting.TContext, http.Handler) {
	return func(tCtx ktesting.TContext, server http.Handler) {
		tCtx.Helper()
		client := tCtx.Client().AdmissionregistrationV1()
		switch policy {
		case admissionPolicyNone:
		case admissionPolicyVAP:
			if _, err := client.ValidatingAdmissionPolicies().Create(tCtx, benchVAP("bench-vap"), metav1.CreateOptions{}); err != nil {
				tCtx.Fatalf("failed to create ValidatingAdmissionPolicy: %v", err)
			}
			if _, err := client.ValidatingAdmissionPolicyBindings().Create(tCtx, &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "bench-vap"},
				Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
					PolicyName:        "bench-vap",
					ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
				},
			}, metav1.CreateOptions{}); err != nil {
				tCtx.Fatalf("failed to create ValidatingAdmissionPolicyBinding: %v", err)
			}
			// The policy rejects ':latest' images once it is enforced.
			denied := admissionPolicyPod(policy, 1)
			denied.Spec.Containers[0].Image = "registry.k8s.io/pause:latest"
			waitForDenial(tCtx, server, encodeObject(tCtx, acceptJSON, denied))
		case admissionPolicyMAPApply, admissionPolicyMAPJSONPatch:
			p := benchMAPApply("bench-" + policy)
			if policy == admissionPolicyMAPJSONPatch {
				p = benchMAPJSONPatch("bench-" + policy)
			}
			if _, err := client.MutatingAdmissionPolicies().Create(tCtx, p, metav1.CreateOptions{}); err != nil {
				tCtx.Fatalf("failed to create MutatingAdmissionPolicy %s: %v", p.Name, err)
			}
			if _, err := client.MutatingAdmissionPolicyBindings().Create(tCtx, &admissionregistrationv1.MutatingAdmissionPolicyBinding{
				ObjectMeta: metav1.ObjectMeta{Name: p.Name},
				Spec:       admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{PolicyName: p.Name},
			}, metav1.CreateOptions{}); err != nil {
				tCtx.Fatalf("failed to create MutatingAdmissionPolicyBinding %s: %v", p.Name, err)
			}
			waitForMutation(tCtx, server, encodeObject(tCtx, acceptJSON, admissionPolicyPod(policy, 1)))
		default:
			tCtx.Fatalf("unknown admission policy scenario %q", policy)
		}
	}
}

func postRecorded(server http.Handler, path string, body []byte) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	req.Header.Set("Content-Type", acceptJSON)
	req.Header.Set("Accept", acceptJSON)
	rec := httptest.NewRecorder()
	server.ServeHTTP(rec, req)
	return rec
}

func waitForDenial(tCtx ktesting.TContext, server http.Handler, body []byte) {
	tCtx.Helper()
	var lastCode int
	var lastBody string
	if err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, admissionPolicyReadyTimeout, true, func(_ context.Context) (bool, error) {
		rec := postRecorded(server, admissionPolicyPodsPath+"?"+admissionPolicyDryRunQuery, body)
		lastCode, lastBody = rec.Code, rec.Body.String()
		return lastCode == http.StatusUnprocessableEntity, nil
	}); err != nil {
		tCtx.Fatalf("waiting for ValidatingAdmissionPolicy to deny, last got HTTP %d: %s", lastCode, lastBody)
	}
}

func waitForMutation(tCtx ktesting.TContext, server http.Handler, body []byte) {
	tCtx.Helper()
	var lastBody string
	if err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, admissionPolicyReadyTimeout, true, func(_ context.Context) (bool, error) {
		rec := postRecorded(server, admissionPolicyPodsPath+"?"+admissionPolicyDryRunQuery, body)
		lastBody = rec.Body.String()
		if rec.Code != http.StatusCreated {
			return false, nil
		}
		var out v1.Pod
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			return false, err
		}
		return out.Labels["mutated-by"] == "map-bench", nil
	}); err != nil {
		tCtx.Fatalf("waiting for MutatingAdmissionPolicy to apply, last response: %s", lastBody)
	}
}

// admissionPolicyPod returns the exemplar pod labeled for the given policy
// scenario, with its containers replaced by the given number of copies of the
// exemplar's main container.
func admissionPolicyPod(policy string, containers int) *v1.Pod {
	pod := exemplarPod.DeepCopy()
	pod.Namespace = metav1.NamespaceDefault
	pod.Name = "pod-new"
	pod.Labels[admissionPolicyScenarioKey] = policy
	main := pod.Spec.Containers[0]
	cs := make([]v1.Container, containers)
	for i := range cs {
		cs[i] = *main.DeepCopy()
		if i > 0 {
			cs[i].Name = fmt.Sprintf("c-%d", i)
		}
	}
	pod.Spec.Containers = cs
	return pod
}

// matchBenchPods matches Pods labeled with the given scenario, so each policy
// only applies to its own requests.
func matchBenchPods(policy string) *admissionregistrationv1.MatchResources {
	return &admissionregistrationv1.MatchResources{
		ObjectSelector: &metav1.LabelSelector{MatchLabels: map[string]string{admissionPolicyScenarioKey: policy}},
		ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
			RuleWithOperations: admissionregistrationv1.RuleWithOperations{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update},
				Rule: admissionregistrationv1.Rule{
					APIGroups:   []string{""},
					APIVersions: []string{"v1"},
					Resources:   []string{"pods"},
				},
			},
		}},
	}
}

// benchVAP is a ValidatingAdmissionPolicy shaped like policies seen in
// practice: composition variables, several validations iterating over the
// pod's containers, a messageExpression, a regex check and an audit
// annotation.
func benchVAP(name string) *admissionregistrationv1.ValidatingAdmissionPolicy {
	return &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy:    ptr.To(admissionregistrationv1.Fail),
			MatchConstraints: matchBenchPods(admissionPolicyVAP),
			Variables: []admissionregistrationv1.Variable{
				{Name: "containers", Expression: "object.spec.containers"},
				{Name: "images", Expression: "variables.containers.map(c, c.image)"},
			},
			Validations: []admissionregistrationv1.Validation{
				{
					Expression:        "size(variables.containers) <= 200",
					MessageExpression: "'pod has ' + string(size(variables.containers)) + ' containers, more than 200'",
				},
				{Expression: "variables.containers.all(c, c.image.startsWith('registry.k8s.io/'))"},
				{Expression: "variables.containers.all(c, has(c.resources) && has(c.resources.requests) && 'memory' in c.resources.requests)"},
				{Expression: "!variables.images.exists(i, i.endsWith(':latest'))"},
				{Expression: "object.metadata.name.matches('^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')"},
			},
			AuditAnnotations: []admissionregistrationv1.AuditAnnotation{
				{Key: "container-count", ValueExpression: "string(size(variables.containers))"},
			},
		},
	}
}

func benchMAP(name, policy string, mutation admissionregistrationv1.Mutation) *admissionregistrationv1.MutatingAdmissionPolicy {
	return &admissionregistrationv1.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
			FailurePolicy:      ptr.To(admissionregistrationv1.Fail),
			ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
			MatchConstraints:   matchBenchPods(policy),
			Mutations:          []admissionregistrationv1.Mutation{mutation},
		},
	}
}

// benchMAPApply mutates every container with a server-side-apply style
// mutation built from typed objects in CEL.
func benchMAPApply(name string) *admissionregistrationv1.MutatingAdmissionPolicy {
	return benchMAP(name, admissionPolicyMAPApply, admissionregistrationv1.Mutation{
		PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
		ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{Expression: `Object{
			metadata: Object.metadata{labels: {"mutated-by": "map-bench"}},
			spec: Object.spec{
				containers: object.spec.containers.map(c,
					Object.spec.containers{name: c.name, imagePullPolicy: "Always"})
			}
		}`},
	})
}

// benchMAPJSONPatch mutates every container with JSON patch operations built
// in CEL.
func benchMAPJSONPatch(name string) *admissionregistrationv1.MutatingAdmissionPolicy {
	return benchMAP(name, admissionPolicyMAPJSONPatch, admissionregistrationv1.Mutation{
		PatchType: admissionregistrationv1.PatchTypeJSONPatch,
		JSONPatch: &admissionregistrationv1.JSONPatch{Expression: `[
			JSONPatch{op: "add", path: "/metadata/labels/mutated-by", value: "map-bench"},
			JSONPatch{op: "replace", path: "/spec/containers",
				value: object.spec.containers.map(c,
					Object.spec.containers{name: c.name, image: c.image, imagePullPolicy: "Always"})}
		]`},
	})
}
