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

package testing

import (
	"encoding/json"
	"net/http"
	"net/url"
	"sync/atomic"
	"testing"

	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	registrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/coverage"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/utils/ptr"
)

// NewCoverageRequirements returns coverage requirements that require every hook on
// CREATE or UPDATE pods to also intercept UPDATE pods/dynamic.
func NewCoverageRequirements(t testing.TB) []coverage.Requirement {
	t.Helper()
	reqs := []coverage.Requirement{{
		Resource: schema.GroupResource{Resource: "pods"},
		Target:   coverage.Shape{Operation: admission.Update, Subresource: "dynamic"},
		Sources:  []coverage.Shape{{Operation: admission.Create}, {Operation: admission.Update}},
	}}
	if err := coverage.Validate(reqs); err != nil {
		t.Fatal(err)
	}
	return reqs
}

// NewCoverageTestServer starts a webhook backend that allows every request and
// counts the calls, so that any rejection comes from coverage and "not called"
// is observable.
func NewCoverageTestServer(t testing.TB) (*url.URL, *atomic.Int32) {
	t.Helper()
	calls := &atomic.Int32{}
	srv := NewTestServerWithHandler(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		// v1beta1 responses are not checked for GVK or UID.
		_ = json.NewEncoder(w).Encode(&admissionv1beta1.AdmissionReview{Response: &admissionv1beta1.AdmissionResponse{Allowed: true}})
	})
	srv.StartTLS()
	t.Cleanup(srv.Close)
	u, err := url.ParseRequestURI(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	return u, calls
}

// NewCoverageWebhook returns a webhook on CREATE and UPDATE of the given core
// resources. It fails open and matches every namespace and object, so only
// coverage can reject a request.
func NewCoverageWebhook(name string, resources ...string) registrationv1.ValidatingWebhook {
	return registrationv1.ValidatingWebhook{
		Name: name,
		Rules: []registrationv1.RuleWithOperations{{
			Operations: []registrationv1.OperationType{registrationv1.Create, registrationv1.Update},
			Rule:       registrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: resources},
		}},
		ClientConfig:            ccfgSVC("allow"),
		FailurePolicy:           ptr.To(registrationv1.Ignore),
		SideEffects:             &sideEffectsNone,
		NamespaceSelector:       &metav1.LabelSelector{},
		ObjectSelector:          &metav1.LabelSelector{},
		AdmissionReviewVersions: []string{"v1beta1"},
	}
}

// NewPodAttribute returns attributes for op on pods/subresource of pod
// "default/p", recording the annotations added to them.
func NewPodAttribute(op admission.Operation, subresource string, labels map[string]string) *FakeAttributes {
	pod := &corev1.Pod{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{Name: "p", Namespace: "default", Labels: labels},
	}
	var oldPod runtime.Object
	var opts runtime.Object = &metav1.CreateOptions{}
	if op == admission.Update {
		oldPod, opts = pod.DeepCopy(), &metav1.UpdateOptions{}
	}
	return &FakeAttributes{Attributes: admission.NewAttributesRecord(
		pod, oldPod, corev1.SchemeGroupVersion.WithKind("Pod"), "default", "p",
		corev1.SchemeGroupVersion.WithResource("pods"), subresource, op, opts, false,
		&user.DefaultInfo{Name: "coverage-test"})}
}

// CoverageViolations returns the UPDATE coverage violations recorded
// for the named webhook, whose type is "validate" or "admit".
func CoverageViolations(t testing.TB, name, webhookType string) float64 {
	t.Helper()
	values, err := testutil.GetCounterValuesFromGatherer(legacyregistry.DefaultGatherer, "apiserver_admission_coverage_violations_total",
		map[string]string{"name": name, "kind": admissionmetrics.KindWebhook, "type": webhookType}, "operation")
	if err != nil {
		// The vector is not gathered until its first increment.
		return 0
	}
	return values["UPDATE"]
}
