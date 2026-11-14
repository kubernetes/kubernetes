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
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/ktesting"
	"k8s.io/utils/ptr"
)

// admissionPolicy selects the CEL admission policy installed while a scenario
// is measured.
type admissionPolicy string

const (
	admissionPolicyNone     admissionPolicy = ""
	validatingPolicy        admissionPolicy = "ValidatingAdmissionPolicy"
	mutatingApplyPolicy     admissionPolicy = "MutatingAdmissionPolicy applyConfiguration"
	mutatingJSONPatchPolicy admissionPolicy = "MutatingAdmissionPolicy jsonPatch"
)

const (
	vapPath = "/apis/admissionregistration.k8s.io/v1/validatingadmissionpolicies"
	mapPath = "/apis/admissionregistration.k8s.io/v1/mutatingadmissionpolicies"

	admissionPolicyProbePath    = "/api/v1/namespaces/" + metav1.NamespaceDefault + "/pods?dryRun=All&fieldManager=bench"
	admissionPolicyReadyTimeout = time.Minute
	mutatedByLabel              = "mutated-by"
	mutatedByValue              = "map-bench"
)

// installAdmissionPolicy creates the policy and its binding and waits until
// they are enforced. The returned func removes them and waits until they no
// longer apply, so later scenarios on the shared apiserver are unaffected.
//
// Policies are loaded asynchronously and nothing reports when they take
// effect, so readiness is probed with dry-run pod creates.
func installAdmissionPolicy(tCtx ktesting.TContext, server http.Handler, policy admissionPolicy) (uninstall func()) {
	tCtx.Helper()
	client := tCtx.Client().AdmissionregistrationV1()
	switch policy {
	case admissionPolicyNone:
		return func() {}
	case validatingPolicy:
		p := benchVAP()
		if _, err := client.ValidatingAdmissionPolicies().Create(tCtx, p, metav1.CreateOptions{}); err != nil {
			tCtx.Fatalf("failed to create ValidatingAdmissionPolicy: %v", err)
		}
		if _, err := client.ValidatingAdmissionPolicyBindings().Create(tCtx, &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{Name: p.Name},
			Spec: admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{
				PolicyName:        p.Name,
				ValidationActions: []admissionregistrationv1.ValidationAction{admissionregistrationv1.Deny},
			},
		}, metav1.CreateOptions{}); err != nil {
			tCtx.Fatalf("failed to create ValidatingAdmissionPolicyBinding: %v", err)
		}
		// The policy rejects ':latest' images once it is enforced.
		denied := exemplarPod.DeepCopy()
		denied.Name = "pod-vap-probe"
		denied.Spec.Containers[0].Image = "registry.k8s.io/pause:latest"
		isDenied := func(rec *httptest.ResponseRecorder) (bool, error) {
			return rec.Code == http.StatusUnprocessableEntity, nil
		}
		waitForAdmissionPolicy(tCtx, server, denied, isDenied)
		return func() {
			if err := client.ValidatingAdmissionPolicyBindings().Delete(tCtx, p.Name, metav1.DeleteOptions{}); err != nil {
				tCtx.Fatalf("failed to delete ValidatingAdmissionPolicyBinding: %v", err)
			}
			if err := client.ValidatingAdmissionPolicies().Delete(tCtx, p.Name, metav1.DeleteOptions{}); err != nil {
				tCtx.Fatalf("failed to delete ValidatingAdmissionPolicy: %v", err)
			}
			waitForAdmissionPolicy(tCtx, server, denied, not(isDenied))
		}
	case mutatingApplyPolicy, mutatingJSONPatchPolicy:
		p := benchMAPApply()
		if policy == mutatingJSONPatchPolicy {
			p = benchMAPJSONPatch()
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
		probe := exemplarPod.DeepCopy()
		probe.Name = "pod-map-probe"
		isMutated := func(rec *httptest.ResponseRecorder) (bool, error) {
			if rec.Code != http.StatusCreated {
				return false, fmt.Errorf("unexpected HTTP %d: %s", rec.Code, rec.Body.String())
			}
			var out v1.Pod
			if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
				return false, err
			}
			return out.Labels[mutatedByLabel] == mutatedByValue, nil
		}
		waitForAdmissionPolicy(tCtx, server, probe, isMutated)
		return func() {
			if err := client.MutatingAdmissionPolicyBindings().Delete(tCtx, p.Name, metav1.DeleteOptions{}); err != nil {
				tCtx.Fatalf("failed to delete MutatingAdmissionPolicyBinding %s: %v", p.Name, err)
			}
			if err := client.MutatingAdmissionPolicies().Delete(tCtx, p.Name, metav1.DeleteOptions{}); err != nil {
				tCtx.Fatalf("failed to delete MutatingAdmissionPolicy %s: %v", p.Name, err)
			}
			waitForAdmissionPolicy(tCtx, server, probe, not(isMutated))
		}
	default:
		tCtx.Fatalf("unknown admission policy %q", policy)
		return nil
	}
}

func not(cond func(*httptest.ResponseRecorder) (bool, error)) func(*httptest.ResponseRecorder) (bool, error) {
	return func(rec *httptest.ResponseRecorder) (bool, error) {
		ok, err := cond(rec)
		return !ok, err
	}
}

// waitForAdmissionPolicy sends pod as a dry-run create until cond holds.
func waitForAdmissionPolicy(tCtx ktesting.TContext, server http.Handler, pod *v1.Pod, cond func(*httptest.ResponseRecorder) (bool, error)) {
	tCtx.Helper()
	body := encodeObject(tCtx, acceptJSON, pod)
	var lastCode int
	var lastBody string
	if err := wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, admissionPolicyReadyTimeout, true, func(context.Context) (bool, error) {
		req := httptest.NewRequest(http.MethodPost, admissionPolicyProbePath, bytes.NewReader(body))
		req.Header.Set("Content-Type", acceptJSON)
		req.Header.Set("Accept", acceptJSON)
		rec := httptest.NewRecorder()
		server.ServeHTTP(rec, req)
		lastCode, lastBody = rec.Code, rec.Body.String()
		return cond(rec)
	}); err != nil {
		tCtx.Fatalf("waiting for admission policy state: %v, last got HTTP %d: %s", err, lastCode, lastBody)
	}
}

func matchPodWrites() *admissionregistrationv1.MatchResources {
	return &admissionregistrationv1.MatchResources{
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

// benchVAP is a typical ValidatingAdmissionPolicy: variables, validations
// over the pod's containers, a messageExpression and an audit annotation.
func benchVAP() *admissionregistrationv1.ValidatingAdmissionPolicy {
	return &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "bench-vap"},
		Spec: admissionregistrationv1.ValidatingAdmissionPolicySpec{
			FailurePolicy:    ptr.To(admissionregistrationv1.Fail),
			MatchConstraints: matchPodWrites(),
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

func benchMAP(name string, mutation admissionregistrationv1.Mutation) *admissionregistrationv1.MutatingAdmissionPolicy {
	return &admissionregistrationv1.MutatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
			FailurePolicy:      ptr.To(admissionregistrationv1.Fail),
			ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
			MatchConstraints:   matchPodWrites(),
			Mutations:          []admissionregistrationv1.Mutation{mutation},
		},
	}
}

// benchMAPApply labels the pod and patches every container via applyConfiguration.
func benchMAPApply() *admissionregistrationv1.MutatingAdmissionPolicy {
	return benchMAP("bench-map-apply", admissionregistrationv1.Mutation{
		PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
		ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{Expression: `Object{
			metadata: Object.metadata{labels: {"` + mutatedByLabel + `": "` + mutatedByValue + `"}},
			spec: Object.spec{
				containers: object.spec.containers.map(c,
					Object.spec.containers{name: c.name, imagePullPolicy: "Always"})
			}
		}`},
	})
}

// benchMAPJSONPatch labels the pod and patches every container via jsonPatch.
func benchMAPJSONPatch() *admissionregistrationv1.MutatingAdmissionPolicy {
	return benchMAP("bench-map-jsonpatch", admissionregistrationv1.Mutation{
		PatchType: admissionregistrationv1.PatchTypeJSONPatch,
		JSONPatch: &admissionregistrationv1.JSONPatch{Expression: `[
			JSONPatch{op: "add", path: "/metadata/labels/` + mutatedByLabel + `", value: "` + mutatedByValue + `"},
			JSONPatch{op: "replace", path: "/spec/containers",
				value: object.spec.containers.map(c,
					Object.spec.containers{name: c.name, image: c.image, imagePullPolicy: "Always"})}
		]`},
	})
}
