/*
Copyright 2020 The Kubernetes Authors.

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
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/admission/v1"
	admissionv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestMutatingWebhookDuplicateOwnerReferences ensures that the API server
// handler correctly deduplicates owner references if a mutating webhook
// patches create/update requests with duplicate owner references.
func TestMutatingWebhookDuplicateOwnerReferences(t *testing.T) {
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert with error: %+v", err)
	}

	webhookServer := httptest.NewUnstartedServer(newDuplicateOwnerReferencesWebhookHandler(t))
	webhookServer.TLS = &tls.Config{
		RootCAs:      roots,
		Certificates: []tls.Certificate{cert},
	}
	webhookServer.StartTLS()
	defer webhookServer.Close()

	s := kubeapiservertesting.StartTestServerOrDie(t,
		kubeapiservertesting.NewDefaultTestServerOptions(), []string{
			"--disable-admission-plugins=ServiceAccount",
		}, framework.SharedEtcd())
	defer s.TearDownFn()

	b := &bytes.Buffer{}
	warningWriter := restclient.NewWarningWriter(b, restclient.WarningWriterOptions{})
	s.ClientConfig.WarningHandler = warningWriter
	client := clientset.NewForConfigOrDie(s.ClientConfig)
	if _, err := client.CoreV1().Pods("default").Create(
		context.TODO(), duplicateOwnerReferencesMarkerFixture, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	fail := admissionv1.Fail
	none := admissionv1.SideEffectClassNone
	mutatingCfg, err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(context.TODO(), &admissionv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "dup-owner-references.admission.integration.test"},
		Webhooks: []admissionv1.MutatingWebhook{{
			Name: "dup-owner-references.admission.integration.test",
			ClientConfig: admissionv1.WebhookClientConfig{
				URL:      &webhookServer.URL,
				CABundle: localhostCert,
			},
			Rules: []admissionv1.RuleWithOperations{{
				Operations: []admissionv1.OperationType{admissionv1.Create, admissionv1.Update},
				Rule:       admissionv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"pods"}},
			}},
			FailurePolicy:           &fail,
			AdmissionReviewVersions: []string{"v1", "v1beta1"},
			SideEffects:             &none,
		}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(context.TODO(), mutatingCfg.GetName(), metav1.DeleteOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}()

	// Make sure dedup happens in patch requests
	var pod *corev1.Pod
	var lastErr string
	// wait until new webhook is called
	expectedWarning := fmt.Sprintf(handlers.DuplicateOwnerReferencesAfterMutatingAdmissionWarningFormat,
		duplicateOwnerReferencesMarkerFixture.OwnerReferences[0].UID)
	if err := wait.PollImmediate(time.Millisecond*5, wait.ForeverTestTimeout, func() (bool, error) {
		pod, err = client.CoreV1().Pods("default").Patch(context.TODO(), duplicateOwnerReferencesMarkerFixture.Name, types.JSONPatchType, []byte("[]"), metav1.PatchOptions{})
		if err != nil {
			return false, err
		}
		if warningWriter.WarningCount() == 0 {
			lastErr = fmt.Sprintf("no warning, owner references: %v", pod.OwnerReferences)
			return false, nil
		}
		if !strings.Contains(b.String(), expectedWarning) {
			lastErr = fmt.Sprintf("unexpected warning, expected: %v, got: %v",
				expectedWarning, b.String())
			return false, nil
		}
		if len(pod.OwnerReferences) != 1 {
			lastErr = fmt.Sprintf("unexpected owner references, expected one entry, got: %v",
				pod.OwnerReferences)
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatalf("failed to wait for apiserver handling webhook mutation: %v, last error: %v", err, lastErr)
	}
	if strings.Contains(b.String(), ".metadata.ownerReferences contains duplicate entries,") {
		t.Errorf("unexpected warning happened before mutating admission")
	}
	if warningWriter.WarningCount() != 1 {
		t.Errorf("expected one warning, got: %v", warningWriter.WarningCount())
	}
	b.Reset()

	// Make sure dedup happens in update requests
	pod, err = client.CoreV1().Pods("default").Update(context.TODO(), pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if warningWriter.WarningCount() != 2 {
		t.Errorf("expected two warnings, got: %v", warningWriter.WarningCount())
	}
	if !strings.Contains(b.String(), expectedWarning) {
		t.Errorf("unexpected warning, expected: %v, got: %v",
			expectedWarning, b.String())
	}
	if strings.Contains(b.String(), ".metadata.ownerReferences contains duplicate entries,") {
		t.Errorf("unexpected warning happened before mutating admission")
	}
	b.Reset()

	if err := client.CoreV1().Pods("default").Delete(context.TODO(), duplicateOwnerReferencesMarkerFixture.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("failed to delete marker pod: %v", err)
	}
	// expect no more warning
	if warningWriter.WarningCount() != 2 {
		t.Errorf("expected two warnings, got: %v", warningWriter.WarningCount())
	}

}

func newDuplicateOwnerReferencesWebhookHandler(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		data, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}
		review := v1.AdmissionReview{}
		if err := json.Unmarshal(data, &review); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}

		if len(review.Request.Object.Raw) == 0 {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		pod := &corev1.Pod{}
		if err := json.Unmarshal(review.Request.Object.Raw, pod); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		review.Response = &v1.AdmissionResponse{
			Allowed: true,
			UID:     review.Request.UID,
			Result:  &metav1.Status{Message: "admitted"},
		}
		if len(pod.OwnerReferences) > 0 {
			review.Response.Patch = []byte(fmt.Sprintf(`[{"op":"add","path":"/metadata/ownerReferences/-","value":{"apiVersion":"v1", "kind": "Node", "name": "fake-node", "uid": "%v"}}]`, pod.OwnerReferences[0].UID))
			jsonPatch := v1.PatchTypeJSONPatch
			review.Response.PatchType = &jsonPatch
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(review); err != nil {
			t.Errorf("Marshal of response failed with error: %v", err)
		}
	})
}

var duplicateOwnerReferencesMarkerFixture = &corev1.Pod{
	ObjectMeta: metav1.ObjectMeta{
		Namespace: "default",
		Name:      "duplicate-owner-references-test-marker",
		OwnerReferences: []metav1.OwnerReference{{
			APIVersion: "v1",
			Kind:       "Node",
			Name:       "fake-node",
			UID:        uuid.NewUUID(),
		}},
	},
	Spec: corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:  "fake-name",
			Image: "fakeimage",
		}},
	},
}
