/*
Copyright 2019 The Kubernetes Authors.

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
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/api/admission/v1beta1"
	admissionv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	registrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils"
)

const (
	testReinvocationClientUsername = "webhook-reinvocation-integration-client"
	auditPolicy                    = `
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  - level: Request
    resources:
      - group: "" # core
        resources: ["pods"]
`
)

// TestWebhookReinvocationPolicyWithWatchCache ensures that the admission webhook reinvocation policy is applied correctly with the watch cache enabled.
func TestWebhookReinvocationPolicyWithWatchCache(t *testing.T) {
	testWebhookReinvocationPolicy(t, true)
}

// TestWebhookReinvocationPolicyWithoutWatchCache ensures that the admission webhook reinvocation policy is applied correctly without the watch cache enabled.
func TestWebhookReinvocationPolicyWithoutWatchCache(t *testing.T) {
	testWebhookReinvocationPolicy(t, false)
}

func mutationAnnotationValue(configuration, webhook string, mutated bool) string {
	return fmt.Sprintf(`{"configuration":"%s","webhook":"%s","mutated":%t}`, configuration, webhook, mutated)
}

func patchAnnotationValue(configuration, webhook string, patch string) string {
	return strings.Replace(fmt.Sprintf(`{"configuration": "%s", "webhook": "%s", "patch": %s, "patchType": "JSONPatch"}`, configuration, webhook, patch), " ", "", -1)
}

// testWebhookReinvocationPolicy ensures that the admission webhook reinvocation policy is applied correctly.
func testWebhookReinvocationPolicy(t *testing.T, watchCache bool) {
	reinvokeNever := registrationv1beta1.NeverReinvocationPolicy
	reinvokeIfNeeded := registrationv1beta1.IfNeededReinvocationPolicy

	type testWebhook struct {
		path           string
		policy         *registrationv1beta1.ReinvocationPolicyType
		objectSelector *metav1.LabelSelector
	}

	testCases := []struct {
		name                           string
		initialPriorityClass           string
		webhooks                       []testWebhook
		expectLabels                   map[string]string
		expectInvocations              map[string]int
		expectError                    bool
		errorContains                  string
		expectAuditMutationAnnotations map[string]string
		expectAuditPatchAnnotations    map[string]string
	}{
		{ // in-tree (mutation), webhook (no mutation), no reinvocation required
			name:                 "no reinvocation for in-tree only mutation",
			initialPriorityClass: "low-priority", // trigger initial in-tree mutation
			webhooks: []testWebhook{
				{path: "/noop", policy: &reinvokeIfNeeded},
			},
			expectInvocations: map[string]int{"/noop": 1},
			expectAuditMutationAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue("admission.integration.test-0", "admission.integration.test.0.noop", false),
			},
		},
		{ // in-tree (mutation), webhook (mutation), reinvoke in-tree (no-mutation), no webhook reinvocation required
			name:                 "no webhook reinvocation for webhook when no in-tree reinvocation mutations",
			initialPriorityClass: "low-priority", // trigger initial in-tree mutation
			webhooks: []testWebhook{
				{path: "/addlabel", policy: &reinvokeIfNeeded},
			},
			expectInvocations: map[string]int{"/addlabel": 1},
			expectAuditPatchAnnotations: map[string]string{
				"patch.webhook.admission.k8s.io/round_0_index_0": patchAnnotationValue("admission.integration.test-1", "admission.integration.test.0.addlabel", `[{"op": "add", "path": "/metadata/labels/a", "value": "true"}]`),
			},
			expectAuditMutationAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue("admission.integration.test-1", "admission.integration.test.0.addlabel", true),
			},
		},
		{ // in-tree (mutation), webhook (mutation), reinvoke in-tree (mutation), webhook (no-mutation), both reinvoked
			name:                 "webhook is reinvoked after in-tree reinvocation",
			initialPriorityClass: "low-priority", // trigger initial in-tree mutation
			webhooks: []testWebhook{
				// Priority plugin is ordered to run before mutating webhooks
				{path: "/setpriority", policy: &reinvokeIfNeeded}, // trigger in-tree reinvoke mutation
			},
			expectInvocations: map[string]int{"/setpriority": 2},
			expectAuditPatchAnnotations: map[string]string{
				"patch.webhook.admission.k8s.io/round_0_index_0": patchAnnotationValue("admission.integration.test-2", "admission.integration.test.0.setpriority", `[{"op": "add", "path": "/spec/priorityClassName", "value": "high-priority"},{"op": "remove", "path": "/spec/priority"}]`),
			},
			expectAuditMutationAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue("admission.integration.test-2", "admission.integration.test.0.setpriority", true),
				"mutation.webhook.admission.k8s.io/round_1_index_0": mutationAnnotationValue("admission.integration.test-2", "admission.integration.test.0.setpriority", false),
			},
		},
		{ // in-tree (mutation), webhook A (mutation), webhook B (mutation), reinvoke in-tree (no-mutation), reinvoke webhook A (no-mutation), no reinvocation of webhook B required
			name:                 "no reinvocation of webhook B when in-tree or prior webhook mutations",
			initialPriorityClass: "low-priority", // trigger initial in-tree mutation
			webhooks: []testWebhook{
				{path: "/addlabel", policy: &reinvokeIfNeeded},
				{path: "/conditionaladdlabel", policy: &reinvokeIfNeeded},
			},
			expectLabels:      map[string]string{"x": "true", "a": "true", "b": "true"},
			expectInvocations: map[string]int{"/addlabel": 2, "/conditionaladdlabel": 1},
			expectAuditPatchAnnotations: map[string]string{
				"patch.webhook.admission.k8s.io/round_0_index_0": patchAnnotationValue("admission.integration.test-3", "admission.integration.test.0.addlabel", `[{"op": "add", "path": "/metadata/labels/a", "value": "true"}]`),
				"patch.webhook.admission.k8s.io/round_0_index_1": patchAnnotationValue("admission.integration.test-3", "admission.integration.test.1.conditionaladdlabel", `[{"op": "add", "path": "/metadata/labels/b", "value": "true"}]`),
			},
			expectAuditMutationAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue("admission.integration.test-3", "admission.integration.test.0.addlabel", true),
				"mutation.webhook.admission.k8s.io/round_0_index_1": mutationAnnotationValue("admission.integration.test-3", "admission.integration.test.1.conditionaladdlabel", true),
				"mutation.webhook.admission.k8s.io/round_1_index_0": mutationAnnotationValue("admission.integration.test-3", "admission.integration.test.0.addlabel", false),
			},
		},
		{ // in-tree (mutation), webhook A (mutation), webhook B (mutation), reinvoke in-tree (no-mutation), reinvoke webhook A (mutation), reinvoke webhook B (mutation), both webhooks reinvoked
			name:                 "all webhooks reinvoked when any webhook reinvocation causes mutation",
			initialPriorityClass: "low-priority", // trigger initial in-tree mutation
			webhooks: []testWebhook{
				{path: "/settrue", policy: &reinvokeIfNeeded},
				{path: "/setfalse", policy: &reinvokeIfNeeded},
			},
			expectLabels:      map[string]string{"x": "true", "fight": "false"},
			expectInvocations: map[string]int{"/settrue": 2, "/setfalse": 2},
			expectAuditPatchAnnotations: map[string]string{
				"patch.webhook.admission.k8s.io/round_0_index_0": patchAnnotationValue("admission.integration.test-4", "admission.integration.test.0.settrue", `[{"op": "replace", "path": "/metadata/labels/fight", "value": "true"}]`),
				"patch.webhook.admission.k8s.io/round_0_index_1": patchAnnotationValue("admission.integration.test-4", "admission.integration.test.1.setfalse", `[{"op": "replace", "path": "/metadata/labels/fight", "value": "false"}]`),
				"patch.webhook.admission.k8s.io/round_1_index_0": patchAnnotationValue("admission.integration.test-4", "admission.integration.test.0.settrue", `[{"op": "replace", "path": "/metadata/labels/fight", "value": "true"}]`),
				"patch.webhook.admission.k8s.io/round_1_index_1": patchAnnotationValue("admission.integration.test-4", "admission.integration.test.1.setfalse", `[{"op": "replace", "path": "/metadata/labels/fight", "value": "false"}]`),
			},
			expectAuditMutationAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue("admission.integration.test-4", "admission.integration.test.0.settrue", true),
				"mutation.webhook.admission.k8s.io/round_0_index_1": mutationAnnotationValue("admission.integration.test-4", "admission.integration.test.1.setfalse", true),
				"mutation.webhook.admission.k8s.io/round_1_index_0": mutationAnnotationValue("admission.integration.test-4", "admission.integration.test.0.settrue", true),
				"mutation.webhook.admission.k8s.io/round_1_index_1": mutationAnnotationValue("admission.integration.test-4", "admission.integration.test.1.setfalse", true),
			},
		},
		{ // in-tree (mutation), webhook A is SKIPPED due to objectSelector not matching, webhook B (mutation), reinvoke in-tree (no-mutation), webhook A is SKIPPED even though the labels match now, because it's not called in the first round. No reinvocation of webhook B required
			name:                 "no reinvocation of webhook B when in-tree or prior webhook mutations",
			initialPriorityClass: "low-priority", // trigger initial in-tree mutation
			webhooks: []testWebhook{
				{path: "/conditionaladdlabel", policy: &reinvokeIfNeeded, objectSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "true"}}},
				{path: "/addlabel", policy: &reinvokeIfNeeded},
			},
			expectLabels:      map[string]string{"x": "true", "a": "true"},
			expectInvocations: map[string]int{"/addlabel": 1, "/conditionaladdlabel": 0},
			expectAuditPatchAnnotations: map[string]string{
				"patch.webhook.admission.k8s.io/round_0_index_1": patchAnnotationValue("admission.integration.test-5", "admission.integration.test.1.addlabel", `[{"op": "add", "path": "/metadata/labels/a", "value": "true"}]`),
			},
			expectAuditMutationAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_1": mutationAnnotationValue("admission.integration.test-5", "admission.integration.test.1.addlabel", true),
			},
		},
		{
			name: "invalid priority class set by webhook should result in error from in-tree priority plugin",
			webhooks: []testWebhook{
				// Priority plugin is ordered to run before mutating webhooks
				{path: "/setinvalidpriority", policy: &reinvokeIfNeeded},
			},
			expectError:       true,
			errorContains:     "no PriorityClass with name invalid was found",
			expectInvocations: map[string]int{"/setinvalidpriority": 1},
		},
		{
			name: "'reinvoke never' policy respected",
			webhooks: []testWebhook{
				{path: "/conditionaladdlabel", policy: &reinvokeNever},
				{path: "/addlabel", policy: &reinvokeNever},
			},
			expectLabels:      map[string]string{"x": "true", "a": "true"},
			expectInvocations: map[string]int{"/conditionaladdlabel": 1, "/addlabel": 1},
			expectAuditPatchAnnotations: map[string]string{
				"patch.webhook.admission.k8s.io/round_0_index_1": patchAnnotationValue("admission.integration.test-7", "admission.integration.test.1.addlabel", `[{"op": "add", "path": "/metadata/labels/a", "value": "true"}]`),
			},
			expectAuditMutationAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue("admission.integration.test-7", "admission.integration.test.0.conditionaladdlabel", false),
				"mutation.webhook.admission.k8s.io/round_0_index_1": mutationAnnotationValue("admission.integration.test-7", "admission.integration.test.1.addlabel", true),
			},
		},
		{
			name: "'reinvoke never' (by default) policy respected",
			webhooks: []testWebhook{
				{path: "/conditionaladdlabel", policy: nil},
				{path: "/addlabel", policy: nil},
			},
			expectLabels:      map[string]string{"x": "true", "a": "true"},
			expectInvocations: map[string]int{"/conditionaladdlabel": 1, "/addlabel": 1},
			expectAuditPatchAnnotations: map[string]string{
				"patch.webhook.admission.k8s.io/round_0_index_1": patchAnnotationValue("admission.integration.test-8", "admission.integration.test.1.addlabel", `[{"op": "add", "path": "/metadata/labels/a", "value": "true"}]`),
			},
			expectAuditMutationAnnotations: map[string]string{
				"mutation.webhook.admission.k8s.io/round_0_index_0": mutationAnnotationValue("admission.integration.test-8", "admission.integration.test.0.conditionaladdlabel", false),
				"mutation.webhook.admission.k8s.io/round_0_index_1": mutationAnnotationValue("admission.integration.test-8", "admission.integration.test.1.addlabel", true),
			},
		},
	}

	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert with error: %+v", err)
	}

	recorder := &invocationRecorder{counts: map[string]int{}}
	webhookServer := httptest.NewUnstartedServer(newReinvokeWebhookHandler(recorder))
	webhookServer.TLS = &tls.Config{

		RootCAs:      roots,
		Certificates: []tls.Certificate{cert},
	}
	webhookServer.StartTLS()
	defer webhookServer.Close()

	// prepare audit policy file
	policyFile, err := ioutil.TempFile("", "audit-policy.yaml")
	if err != nil {
		t.Fatalf("Failed to create audit policy file: %v", err)
	}
	defer os.Remove(policyFile.Name())
	if _, err := policyFile.Write([]byte(auditPolicy)); err != nil {
		t.Fatalf("Failed to write audit policy file: %v", err)
	}
	if err := policyFile.Close(); err != nil {
		t.Fatalf("Failed to close audit policy file: %v", err)
	}

	// prepare audit log file
	logFile, err := ioutil.TempFile("", "audit.log")
	if err != nil {
		t.Fatalf("Failed to create audit log file: %v", err)
	}
	defer os.Remove(logFile.Name())

	s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--disable-admission-plugins=ServiceAccount",
		fmt.Sprintf("--watch-cache=%v", watchCache),
		"--audit-policy-file", policyFile.Name(),
		"--audit-log-version", "audit.k8s.io/v1",
		"--audit-log-mode", "blocking",
		"--audit-log-path", logFile.Name(),
	}, framework.SharedEtcd())
	defer s.TearDownFn()

	// Configure a client with a distinct user name so that it is easy to distinguish requests
	// made by the client from requests made by controllers. We use this to filter out requests
	// before recording them to ensure we don't accidentally mistake requests from controllers
	// as requests made by the client.
	clientConfig := rest.CopyConfig(s.ClientConfig)
	clientConfig.Impersonate.UserName = testReinvocationClientUsername
	clientConfig.Impersonate.Groups = []string{"system:masters", "system:authenticated"}
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for priorityClass, priority := range map[string]int{"low-priority": 1, "high-priority": 10} {
		_, err = client.SchedulingV1().PriorityClasses().Create(&schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: priorityClass}, Value: int32(priority)})
		if err != nil {
			t.Fatal(err)
		}
	}

	for i, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			upCh := recorder.Reset()
			testCaseID := strconv.Itoa(i)
			ns := "reinvoke-" + testCaseID
			nsLabels := map[string]string{"test-case": testCaseID}
			_, err = client.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns, Labels: nsLabels}})
			if err != nil {
				t.Fatal(err)
			}

			// Write markers to a separate namespace to avoid cross-talk
			markerNs := ns + "-markers"
			markerNsLabels := map[string]string{"test-markers": testCaseID}
			_, err = client.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: markerNs, Labels: markerNsLabels}})
			if err != nil {
				t.Fatal(err)
			}

			// Create a maker object to use to check for the webhook configurations to be ready.
			marker, err := client.CoreV1().Pods(markerNs).Create(newReinvocationMarkerFixture(markerNs))
			if err != nil {
				t.Fatal(err)
			}

			fail := admissionv1beta1.Fail
			webhooks := []admissionv1beta1.MutatingWebhook{}
			for j, webhook := range tt.webhooks {
				endpoint := webhookServer.URL + webhook.path
				name := fmt.Sprintf("admission.integration.test.%d.%s", j, strings.TrimPrefix(webhook.path, "/"))
				webhooks = append(webhooks, admissionv1beta1.MutatingWebhook{
					Name: name,
					ClientConfig: admissionv1beta1.WebhookClientConfig{
						URL:      &endpoint,
						CABundle: localhostCert,
					},
					Rules: []admissionv1beta1.RuleWithOperations{{
						Operations: []admissionv1beta1.OperationType{admissionv1beta1.OperationAll},
						Rule:       admissionv1beta1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"pods"}},
					}},
					ObjectSelector:          webhook.objectSelector,
					NamespaceSelector:       &metav1.LabelSelector{MatchLabels: nsLabels},
					FailurePolicy:           &fail,
					ReinvocationPolicy:      webhook.policy,
					AdmissionReviewVersions: []string{"v1beta1"},
				})
			}
			// Register a marker checking webhook with each set of webhook configurations
			markerEndpoint := webhookServer.URL + "/marker"
			webhooks = append(webhooks, admissionv1beta1.MutatingWebhook{
				Name: "admission.integration.test.marker",
				ClientConfig: admissionv1beta1.WebhookClientConfig{
					URL:      &markerEndpoint,
					CABundle: localhostCert,
				},
				Rules: []admissionv1beta1.RuleWithOperations{{
					Operations: []admissionv1beta1.OperationType{admissionv1beta1.OperationAll},
					Rule:       admissionv1beta1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"pods"}},
				}},
				NamespaceSelector:       &metav1.LabelSelector{MatchLabels: markerNsLabels},
				ObjectSelector:          &metav1.LabelSelector{MatchLabels: map[string]string{"marker": "true"}},
				AdmissionReviewVersions: []string{"v1beta1"},
			})

			cfg, err := client.AdmissionregistrationV1beta1().MutatingWebhookConfigurations().Create(&admissionv1beta1.MutatingWebhookConfiguration{
				ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("admission.integration.test-%d", i)},
				Webhooks:   webhooks,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				err := client.AdmissionregistrationV1beta1().MutatingWebhookConfigurations().Delete(cfg.GetName(), &metav1.DeleteOptions{})
				if err != nil {
					t.Fatal(err)
				}
			}()

			// wait until new webhook is called the first time
			if err := wait.PollImmediate(time.Millisecond*5, wait.ForeverTestTimeout, func() (bool, error) {
				_, err = client.CoreV1().Pods(markerNs).Patch(marker.Name, types.JSONPatchType, []byte("[]"))
				select {
				case <-upCh:
					return true, nil
				default:
					t.Logf("Waiting for webhook to become effective, getting marker object: %v", err)
					return false, nil
				}
			}); err != nil {
				t.Fatal(err)
			}

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: ns,
					Name:      "labeled",
					Labels:    map[string]string{"x": "true"},
				},
				Spec: corev1.PodSpec{
					Containers: []v1.Container{{
						Name:  "fake-name",
						Image: "fakeimage",
					}},
				},
			}
			if tt.initialPriorityClass != "" {
				pod.Spec.PriorityClassName = tt.initialPriorityClass
			}
			obj, err := client.CoreV1().Pods(ns).Create(pod)

			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error but got none")
				}
				if tt.errorContains != "" {
					if !strings.Contains(err.Error(), tt.errorContains) {
						t.Errorf("expected an error saying %q, but got: %v", tt.errorContains, err)
					}
				}
				return
			}

			if err != nil {
				t.Fatal(err)
			}

			if tt.expectLabels != nil {
				labels := obj.GetLabels()
				if !reflect.DeepEqual(tt.expectLabels, labels) {
					t.Errorf("expected labels '%v', but got '%v'", tt.expectLabels, labels)
				}
			}

			if tt.expectInvocations != nil {
				for k, v := range tt.expectInvocations {
					if recorder.GetCount(k) != v {
						t.Errorf("expected %d invocations of %s, but got %d", v, k, recorder.GetCount(k))
					}
				}
			}

			stream, err := os.OpenFile(logFile.Name(), os.O_RDWR, 0600)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			defer stream.Close()
			missing, err := utils.CheckAuditLines(stream, expectedAuditEvents(tt.expectAuditMutationAnnotations, tt.expectAuditPatchAnnotations, ns), auditv1.SchemeGroupVersion)
			if err != nil {
				t.Errorf("unexpected error checking audit lines: %v", err)
			}
			if len(missing.MissingEvents) > 0 {
				t.Errorf("failed to get expected events -- missing: %s", missing)
			}
			if err := stream.Truncate(0); err != nil {
				t.Errorf("unexpected error truncate file: %v", err)
			}
			if _, err := stream.Seek(0, 0); err != nil {
				t.Errorf("unexpected error reset offset: %v", err)
			}
		})
	}
}

type invocationRecorder struct {
	mu     sync.Mutex
	upCh   chan struct{}
	upOnce sync.Once
	counts map[string]int
}

// Reset zeros out all counts and returns a channel that is closed when the first admission of the
// marker object is received.
func (i *invocationRecorder) Reset() chan struct{} {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.counts = map[string]int{}
	i.upCh = make(chan struct{})
	i.upOnce = sync.Once{}
	return i.upCh
}

func (i *invocationRecorder) MarkerReceived() {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.upOnce.Do(func() {
		close(i.upCh)
	})
}

func (i *invocationRecorder) GetCount(path string) int {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.counts[path]
}

func (i *invocationRecorder) IncrementCount(path string) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.counts[path]++
}

func newReinvokeWebhookHandler(recorder *invocationRecorder) http.Handler {
	patch := func(w http.ResponseWriter, patch string) {
		w.Header().Set("Content-Type", "application/json")
		pt := v1beta1.PatchTypeJSONPatch
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed:   true,
				PatchType: &pt,
				Patch:     []byte(patch),
			},
		})
	}
	allow := func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: true,
			},
		})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		data, err := ioutil.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), 400)
		}
		review := v1beta1.AdmissionReview{}
		if err := json.Unmarshal(data, &review); err != nil {
			http.Error(w, err.Error(), 400)
		}
		if review.Request.UserInfo.Username != testReinvocationClientUsername {
			// skip requests not originating from this integration test's client
			allow(w)
			return
		}

		if len(review.Request.Object.Raw) == 0 {
			http.Error(w, err.Error(), 400)
		}
		pod := &corev1.Pod{}
		if err := json.Unmarshal(review.Request.Object.Raw, pod); err != nil {
			http.Error(w, err.Error(), 400)
		}

		recorder.IncrementCount(r.URL.Path)

		switch r.URL.Path {
		case "/marker":
			// When resetting between tests, a marker object is patched until this webhook
			// observes it, at which point it is considered ready.
			recorder.MarkerReceived()
			allow(w)
			return
		case "/noop":
			allow(w)
		case "/settrue":
			patch(w, `[{"op": "replace", "path": "/metadata/labels/fight", "value": "true"}]`)
		case "/setfalse":
			patch(w, `[{"op": "replace", "path": "/metadata/labels/fight", "value": "false"}]`)
		case "/addlabel":
			labels := pod.GetLabels()
			if a, ok := labels["a"]; !ok || a != "true" {
				patch(w, `[{"op": "add", "path": "/metadata/labels/a", "value": "true"}]`)
				return
			}
			allow(w)
		case "/conditionaladdlabel": // if 'a' is set, set 'b' to true
			labels := pod.GetLabels()
			if _, ok := labels["a"]; ok {
				patch(w, `[{"op": "add", "path": "/metadata/labels/b", "value": "true"}]`)
				return
			}
			allow(w)
		case "/setpriority": // sets /spec/priorityClassName to high-priority if it is not already set
			if pod.Spec.PriorityClassName != "high-priority" {
				if pod.Spec.Priority != nil {
					patch(w, `[{"op": "add", "path": "/spec/priorityClassName", "value": "high-priority"},{"op": "remove", "path": "/spec/priority"}]`)
				} else {
					patch(w, `[{"op": "add", "path": "/spec/priorityClassName", "value": "high-priority"}]`)
				}
				return
			}
			allow(w)
		case "/setinvalidpriority":
			patch(w, `[{"op": "add", "path": "/spec/priorityClassName", "value": "invalid"}]`)
		default:
			http.NotFound(w, r)
		}
	})
}

func expectedAuditEvents(webhookMutationAnnotations, webhookPatchAnnotations map[string]string, namespace string) []utils.AuditEvent {
	return []utils.AuditEvent{
		{
			Level:                               auditinternal.LevelRequest,
			Stage:                               auditinternal.StageResponseComplete,
			RequestURI:                          fmt.Sprintf("/api/v1/namespaces/%s/pods", namespace),
			Verb:                                "create",
			Code:                                201,
			User:                                "system:apiserver",
			ImpersonatedUser:                    testReinvocationClientUsername,
			ImpersonatedGroups:                  "system:authenticated,system:masters",
			Resource:                            "pods",
			Namespace:                           namespace,
			AuthorizeDecision:                   "allow",
			RequestObject:                       true,
			ResponseObject:                      false,
			AdmissionWebhookMutationAnnotations: webhookMutationAnnotations,
			AdmissionWebhookPatchAnnotations:    webhookPatchAnnotations,
		},
	}
}

func newReinvocationMarkerFixture(namespace string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      "marker",
			Labels: map[string]string{
				"marker": "true",
			},
		},
		Spec: corev1.PodSpec{
			Containers: []v1.Container{{
				Name:  "fake-name",
				Image: "fakeimage",
			}},
		},
	}
}
