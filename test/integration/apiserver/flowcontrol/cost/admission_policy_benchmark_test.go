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
	"os"
	"runtime"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/ktesting"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

const (
	admissionPolicyScenarioKey  = "bench.k8s.io/scenario"
	admissionPolicyDeployPath   = "/apis/apps/v1/namespaces/" + metav1.NamespaceDefault + "/deployments"
	admissionPolicyVAPPath      = "/apis/admissionregistration.k8s.io/v1/validatingadmissionpolicies"
	admissionPolicyMAPPath      = "/apis/admissionregistration.k8s.io/v1/mutatingadmissionpolicies"
	admissionPolicyDryRunQuery  = "?dryRun=All&fieldManager=bench"
	admissionPolicyReadyTimeout = time.Minute

	// The package defaults to -benchtime=1x and allocations are measured
	// process-wide, so a single request is easily dominated by background
	// apiserver work. A fixed batch of requests is sent per iteration and the
	// per-request average is reported.
	admissionPolicyRequestsPerOp = 50
)

// BenchmarkAdmissionPolicyRequestCost measures per-request heap allocations
// for kube-apiserver requests evaluated by CEL-based admission policies:
//
//   - create of a Deployment validated by a ValidatingAdmissionPolicy or
//     mutated by a MutatingAdmissionPolicy (applyConfiguration and JSONPatch),
//     with an unmatched Deployment as baseline, scaled by the number of
//     containers iterated by the policy expressions (CEL evaluation);
//   - create of a ValidatingAdmissionPolicy / MutatingAdmissionPolicy, whose
//     validation parses, type-checks and cost-estimates every expression
//     (CEL compilation).
//
// Requests run against the same in-process kube-apiserver handler as
// BenchmarkRequestCost. All requests are dry-run so the same request can be
// repeated without changing cluster state. Results are written to $ARTIFACTS
// in the perf-dash GenericPrometheusQuery format with the same labels as
// BenchmarkRequestCost.
func BenchmarkAdmissionPolicyRequestCost(b *testing.B) {
	framework.FixBenchmarkJSONOutput(b)
	tCtx := ktesting.Init(b)
	tCtx, server := setupAPIServer(tCtx)
	setupAdmissionPolicyFixtures(tCtx, server)

	var scenarios []admissionPolicyScenario
	for _, n := range []int{1, 10, 100} {
		for _, policy := range []string{"none", "vap", "map-apply", "map-jsonpatch"} {
			scenarios = append(scenarios, admissionPolicyScenario{
				request: fmt.Sprintf("POST Deployment policy=%s containers=%d", policy, n),
				path:    admissionPolicyDeployPath + admissionPolicyDryRunQuery,
				body:    mustJSON(tCtx, admissionPolicyDeployment(policy, n, 3)),
			})
		}
	}
	scenarios = append(scenarios,
		admissionPolicyScenario{
			request: "POST ValidatingAdmissionPolicy",
			path:    admissionPolicyVAPPath + admissionPolicyDryRunQuery,
			body:    mustJSON(tCtx, benchVAP("bench-vap-dryrun")),
		},
		admissionPolicyScenario{
			request: "POST MutatingAdmissionPolicy applyConfiguration",
			path:    admissionPolicyMAPPath + admissionPolicyDryRunQuery,
			body:    mustJSON(tCtx, benchMAPApply("bench-map-apply-dryrun")),
		},
		admissionPolicyScenario{
			request: "POST MutatingAdmissionPolicy jsonPatch",
			path:    admissionPolicyMAPPath + admissionPolicyDryRunQuery,
			body:    mustJSON(tCtx, benchMAPJSONPatch("bench-map-jsonpatch-dryrun")),
		},
	)

	report := perfData{Version: "v1"}
	for _, sc := range scenarios {
		b.Run("Request="+sc.request, func(b *testing.B) {
			framework.FixBenchmarkJSONOutput(b)
			report.DataItems = append(report.DataItems, benchmarkAdmissionPolicyScenario(b, server, sc)...)
		})
	}
	if artifacts := os.Getenv("ARTIFACTS"); artifacts != "" && len(report.DataItems) > 0 {
		writePerfData(b, artifacts, report)
	}
}

type admissionPolicyScenario struct {
	request string
	path    string
	body    []byte
}

func benchmarkAdmissionPolicyScenario(b *testing.B, server http.Handler, sc admissionPolicyScenario) []dataItem {
	const format = "json"
	b.ReportAllocs()

	// Warm up once so lazily initialized state (type converters, CEL
	// environments, caches) is not attributed to the measured requests.
	runHTTP(b, server, http.MethodPost, sc.path, acceptJSON, acceptJSON, sc.body)

	var startMem, endMem runtime.MemStats
	var responseBytes int
	for i := 0; b.Loop(); i++ {
		if i == 0 {
			runtime.ReadMemStats(&startMem)
		}
		for range admissionPolicyRequestsPerOp {
			responseBytes = runHTTP(b, server, http.MethodPost, sc.path, acceptJSON, acceptJSON, sc.body)
		}
	}
	runtime.ReadMemStats(&endMem)

	b.ReportMetric(float64(len(sc.body)), "request-B/op")
	b.ReportMetric(float64(responseBytes), "response-B/op")
	if b.N == 0 {
		return nil
	}
	requests := float64(b.N * admissionPolicyRequestsPerOp)
	allocBytes := float64(endMem.TotalAlloc-startMem.TotalAlloc) / requests
	allocs := float64(endMem.Mallocs-startMem.Mallocs) / requests
	b.ReportMetric(allocBytes, "alloc-B/request")
	b.ReportMetric(allocs, "allocs/request")

	labels := func(metric string) map[string]string {
		return map[string]string{"Request": sc.request, "Format": format, "Metric": metric}
	}
	return []dataItem{
		{Labels: labels("Allocated bytes per operation"), Unit: "B", Data: map[string]float64{format: allocBytes}},
		{Labels: labels("Allocations per operation"), Unit: "1", Data: map[string]float64{format: allocs}},
		{Labels: labels("Request body bytes per operation"), Unit: "B", Data: map[string]float64{format: float64(len(sc.body))}},
		{Labels: labels("Response body bytes per operation"), Unit: "B", Data: map[string]float64{format: float64(responseBytes)}},
	}
}

// setupAdmissionPolicyFixtures creates the admission policies used by the
// benchmark and waits until each of them is enforced.
func setupAdmissionPolicyFixtures(tCtx ktesting.TContext, server http.Handler) {
	tCtx.Helper()
	client := tCtx.Client().AdmissionregistrationV1()
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
	// The policy rejects more than 50 replicas once it is enforced.
	waitForStatus(tCtx, server, admissionPolicyDeployPath+admissionPolicyDryRunQuery,
		mustJSON(tCtx, admissionPolicyDeployment("vap", 1, 100)), http.StatusUnprocessableEntity)

	for _, p := range []*admissionregistrationv1.MutatingAdmissionPolicy{
		benchMAPApply("bench-map-apply"),
		benchMAPJSONPatch("bench-map-jsonpatch"),
	} {
		if _, err := client.MutatingAdmissionPolicies().Create(tCtx, p, metav1.CreateOptions{}); err != nil {
			tCtx.Fatalf("failed to create MutatingAdmissionPolicy %s: %v", p.Name, err)
		}
		if _, err := client.MutatingAdmissionPolicyBindings().Create(tCtx, &admissionregistrationv1.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{Name: p.Name},
			Spec:       admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{PolicyName: p.Name},
		}, metav1.CreateOptions{}); err != nil {
			tCtx.Fatalf("failed to create MutatingAdmissionPolicyBinding %s: %v", p.Name, err)
		}
	}
	for _, policy := range []string{"map-apply", "map-jsonpatch"} {
		waitForMutation(tCtx, server, mustJSON(tCtx, admissionPolicyDeployment(policy, 1, 3)))
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

func waitForStatus(tCtx ktesting.TContext, server http.Handler, path string, body []byte, want int) {
	tCtx.Helper()
	var lastCode int
	var lastBody string
	if err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, admissionPolicyReadyTimeout, true, func(_ context.Context) (bool, error) {
		rec := postRecorded(server, path, body)
		lastCode, lastBody = rec.Code, rec.Body.String()
		return lastCode == want, nil
	}); err != nil {
		tCtx.Fatalf("POST %s: waiting for HTTP %d, last got HTTP %d: %s", path, want, lastCode, lastBody)
	}
}

func waitForMutation(tCtx ktesting.TContext, server http.Handler, body []byte) {
	tCtx.Helper()
	var lastBody string
	if err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, admissionPolicyReadyTimeout, true, func(_ context.Context) (bool, error) {
		rec := postRecorded(server, admissionPolicyDeployPath+admissionPolicyDryRunQuery, body)
		lastBody = rec.Body.String()
		if rec.Code != http.StatusCreated {
			return false, nil
		}
		var out appsv1.Deployment
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			return false, err
		}
		return out.Labels["mutated-by"] == "map-bench", nil
	}); err != nil {
		tCtx.Fatalf("waiting for MutatingAdmissionPolicy to apply, last response: %s", lastBody)
	}
}

func mustJSON(tCtx ktesting.TContext, obj interface{}) []byte {
	tCtx.Helper()
	data, err := json.Marshal(obj)
	if err != nil {
		tCtx.Fatalf("failed to marshal %T: %v", obj, err)
	}
	return data
}

func admissionPolicyDeployment(policy string, containers int, replicas int32) *appsv1.Deployment {
	cs := make([]corev1.Container, containers)
	for i := range containers {
		cs[i] = corev1.Container{
			Name:  fmt.Sprintf("c-%d", i),
			Image: fmt.Sprintf("registry.example.com/app-%d:v1", i),
			Resources: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{
					corev1.ResourceMemory: resource.MustParse("128Mi"),
					corev1.ResourceCPU:    resource.MustParse("100m"),
				},
			},
		}
	}
	podLabels := map[string]string{"app": "bench"}
	return &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bench-deploy",
			Namespace: metav1.NamespaceDefault,
			Labels:    map[string]string{"app": "bench", admissionPolicyScenarioKey: policy},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To(replicas),
			Selector: &metav1.LabelSelector{MatchLabels: podLabels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: podLabels},
				Spec:       corev1.PodSpec{Containers: cs},
			},
		},
	}
}

// matchBenchDeployments matches Deployments labeled with the given scenario,
// so each policy only applies to its own requests.
func matchBenchDeployments(policy string) *admissionregistrationv1.MatchResources {
	return &admissionregistrationv1.MatchResources{
		ObjectSelector: &metav1.LabelSelector{MatchLabels: map[string]string{admissionPolicyScenarioKey: policy}},
		ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{{
			RuleWithOperations: admissionregistrationv1.RuleWithOperations{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update},
				Rule: admissionregistrationv1.Rule{
					APIGroups:   []string{"apps"},
					APIVersions: []string{"v1"},
					Resources:   []string{"deployments"},
				},
			},
		}},
	}
}

// benchVAP is a ValidatingAdmissionPolicy shaped like policies seen in
// practice: composition variables, several validations iterating over the pod
// template's containers, a messageExpression, a regex check and an audit
// annotation.
func benchVAP(name string) *admissionregistrationv1.ValidatingAdmissionPolicy {
	return &admissionregistrationv1.ValidatingAdmissionPolicy{
		TypeMeta:   metav1.TypeMeta{APIVersion: "admissionregistration.k8s.io/v1", Kind: "ValidatingAdmissionPolicy"},
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy:    ptr.To(admissionregistrationv1.Fail),
			MatchConstraints: matchBenchDeployments("vap"),
			Variables: []admissionregistrationv1.Variable{
				{Name: "containers", Expression: "object.spec.template.spec.containers"},
				{Name: "images", Expression: "variables.containers.map(c, c.image)"},
			},
			Validations: []admissionregistrationv1.Validation{
				{
					Expression:        "object.spec.replicas <= 50",
					MessageExpression: "'replicas ' + string(object.spec.replicas) + ' exceeds 50'",
				},
				{Expression: "variables.containers.all(c, c.image.startsWith('registry.example.com/'))"},
				{Expression: "variables.containers.all(c, has(c.resources) && has(c.resources.limits) && 'memory' in c.resources.limits)"},
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
		TypeMeta:   metav1.TypeMeta{APIVersion: "admissionregistration.k8s.io/v1", Kind: "MutatingAdmissionPolicy"},
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
			FailurePolicy:      ptr.To(admissionregistrationv1.Fail),
			ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
			MatchConstraints:   matchBenchDeployments(policy),
			Mutations:          []admissionregistrationv1.Mutation{mutation},
		},
	}
}

// benchMAPApply mutates every container with a server-side-apply style
// mutation built from typed objects in CEL.
func benchMAPApply(name string) *admissionregistrationv1.MutatingAdmissionPolicy {
	return benchMAP(name, "map-apply", admissionregistrationv1.Mutation{
		PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
		ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{Expression: `Object{
			metadata: Object.metadata{labels: {"mutated-by": "map-bench"}},
			spec: Object.spec{template: Object.spec.template{spec: Object.spec.template.spec{
				containers: object.spec.template.spec.containers.map(c,
					Object.spec.template.spec.containers{name: c.name, imagePullPolicy: "IfNotPresent"})
			}}}
		}`},
	})
}

// benchMAPJSONPatch mutates every container with JSON patch operations built
// in CEL.
func benchMAPJSONPatch(name string) *admissionregistrationv1.MutatingAdmissionPolicy {
	return benchMAP(name, "map-jsonpatch", admissionregistrationv1.Mutation{
		PatchType: admissionregistrationv1.PatchTypeJSONPatch,
		JSONPatch: &admissionregistrationv1.JSONPatch{Expression: `[
			JSONPatch{op: "add", path: "/metadata/labels/mutated-by", value: "map-bench"},
			JSONPatch{op: "replace", path: "/spec/template/spec/containers",
				value: object.spec.template.spec.containers.map(c,
					Object.spec.template.spec.containers{name: c.name, image: c.image, imagePullPolicy: "IfNotPresent"})}
		]`},
	})
}
