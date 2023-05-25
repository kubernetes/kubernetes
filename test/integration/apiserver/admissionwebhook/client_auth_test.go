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
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"sync"
	"testing"
	"time"

	utiltesting "k8s.io/client-go/util/testing"

	"k8s.io/api/admission/v1beta1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	testClientAuthClientUsername = "webhook-client-auth-integration-client"
)

// TestWebhookClientAuthWithAggregatorRouting ensures client auth is used for requests to URL backends
func TestWebhookClientAuthWithAggregatorRouting(t *testing.T) {
	testWebhookClientAuth(t, true)
}

// TestWebhookClientAuthWithoutAggregatorRouting ensures client auth is used for requests to URL backends
func TestWebhookClientAuthWithoutAggregatorRouting(t *testing.T) {
	testWebhookClientAuth(t, false)
}

func testWebhookClientAuth(t *testing.T, enableAggregatorRouting bool) {

	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert with error: %+v", err)
	}

	recorder := &clientAuthRecorder{}
	webhookServer := httptest.NewUnstartedServer(newClientAuthWebhookHandler(t, recorder))
	webhookServer.TLS = &tls.Config{

		RootCAs:      roots,
		Certificates: []tls.Certificate{cert},
	}
	webhookServer.StartTLS()
	defer webhookServer.Close()

	webhookServerURL, err := url.Parse(webhookServer.URL)
	if err != nil {
		t.Fatal(err)
	}

	kubeConfigFile, err := os.CreateTemp("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer utiltesting.CloseAndRemove(t, kubeConfigFile)

	if err := os.WriteFile(kubeConfigFile.Name(), []byte(`
apiVersion: v1
kind: Config
users:
- name: "`+webhookServerURL.Host+`"
  user:
    token: "localhost-match-with-port"
- name: "`+webhookServerURL.Hostname()+`"
  user:
    token: "localhost-match-without-port"
- name: "*.localhost"
  user:
    token: "localhost-prefix"
- name: "*"
  user:
    token: "fallback"
`), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}

	admissionConfigFile, err := os.CreateTemp("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer utiltesting.CloseAndRemove(t, admissionConfigFile)

	if err := os.WriteFile(admissionConfigFile.Name(), []byte(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: ValidatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1alpha1
    kind: WebhookAdmission
    kubeConfigFile: "`+kubeConfigFile.Name()+`"
- name: MutatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1alpha1
    kind: WebhookAdmission
    kubeConfigFile: "`+kubeConfigFile.Name()+`"
`), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}

	s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--disable-admission-plugins=ServiceAccount",
		fmt.Sprintf("--enable-aggregator-routing=%v", enableAggregatorRouting),
		"--admission-control-config-file=" + admissionConfigFile.Name(),
	}, framework.SharedEtcd())
	defer s.TearDownFn()

	// Configure a client with a distinct user name so that it is easy to distinguish requests
	// made by the client from requests made by controllers. We use this to filter out requests
	// before recording them to ensure we don't accidentally mistake requests from controllers
	// as requests made by the client.
	clientConfig := rest.CopyConfig(s.ClientConfig)
	clientConfig.Impersonate.UserName = testClientAuthClientUsername
	clientConfig.Impersonate.Groups = []string{"system:masters", "system:authenticated"}
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = client.CoreV1().Pods("default").Create(context.TODO(), clientAuthMarkerFixture, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	upCh := recorder.Reset()
	ns := "load-balance"
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	fail := admissionregistrationv1.Fail
	mutatingCfg, err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(context.TODO(), &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "admission.integration.test"},
		Webhooks: []admissionregistrationv1.MutatingWebhook{{
			Name: "admission.integration.test",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL:      &webhookServer.URL,
				CABundle: localhostCert,
			},
			Rules: []admissionregistrationv1.RuleWithOperations{{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll},
				Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"pods"}},
			}},
			FailurePolicy:           &fail,
			AdmissionReviewVersions: []string{"v1beta1"},
			SideEffects:             &noSideEffects,
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

	// wait until new webhook is called
	if err := wait.PollImmediate(time.Millisecond*5, wait.ForeverTestTimeout, func() (bool, error) {
		_, err = client.CoreV1().Pods("default").Patch(context.TODO(), clientAuthMarkerFixture.Name, types.JSONPatchType, []byte("[]"), metav1.PatchOptions{})
		if t.Failed() {
			return true, nil
		}
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

}

type clientAuthRecorder struct {
	mu     sync.Mutex
	upCh   chan struct{}
	upOnce sync.Once
}

// Reset zeros out all counts and returns a channel that is closed when the first admission of the
// marker object is received.
func (i *clientAuthRecorder) Reset() chan struct{} {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.upCh = make(chan struct{})
	i.upOnce = sync.Once{}
	return i.upCh
}

func (i *clientAuthRecorder) MarkerReceived() {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.upOnce.Do(func() {
		close(i.upCh)
	})
}

func newClientAuthWebhookHandler(t *testing.T, recorder *clientAuthRecorder) http.Handler {
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
		data, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}
		review := v1beta1.AdmissionReview{}
		if err := json.Unmarshal(data, &review); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}
		if review.Request.UserInfo.Username != testClientAuthClientUsername {
			// skip requests not originating from this integration test's client
			allow(w)
			return
		}

		if authz := r.Header.Get("Authorization"); authz != "Bearer localhost-match-with-port" {
			t.Errorf("unexpected authz header: %q", authz)
			http.Error(w, "Invalid auth", http.StatusUnauthorized)
			return
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

		// When resetting between tests, a marker object is patched until this webhook
		// observes it, at which point it is considered ready.
		if pod.Namespace == clientAuthMarkerFixture.Namespace && pod.Name == clientAuthMarkerFixture.Name {
			recorder.MarkerReceived()
			allow(w)
			return
		}
	})
}

var clientAuthMarkerFixture = &corev1.Pod{
	ObjectMeta: metav1.ObjectMeta{
		Namespace: "default",
		Name:      "marker",
	},
	Spec: corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:  "fake-name",
			Image: "fakeimage",
		}},
	},
}
