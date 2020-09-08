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
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"k8s.io/api/admission/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestManifestBasedWebhookWithClientAuth(t *testing.T) {
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

	kubeConfigFile, err := ioutil.TempFile("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(kubeConfigFile.Name())

	if err := ioutil.WriteFile(kubeConfigFile.Name(), []byte(`
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

	manifestFile, err := ioutil.TempFile("", "manifest.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(manifestFile.Name())

	if err := ioutil.WriteFile(manifestFile.Name(), []byte(`
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: admission.integration.test
webhooks:
- clientConfig:
    caBundle: "`+base64.URLEncoding.EncodeToString(localhostCert)+`"
    url: "`+webhookServer.URL+`"
  failurePolicy: Fail
  name: admission.integration.test
  rules:
  - apiGroups:
    - '*'
    apiVersions:
    - 'v1'
    operations:
    - '*'
    resources:
    - 'pods'
  sideEffects: None
  admissionReviewVersions: ["v1beta1"]
`), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}

	admissionConfigFile, err := ioutil.TempFile("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(admissionConfigFile.Name())

	if err := ioutil.WriteFile(admissionConfigFile.Name(), []byte(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: MutatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    kubeConfigFile: "`+kubeConfigFile.Name()+`"
    manifestFile: "`+manifestFile.Name()+`"
`), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}

	s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--disable-admission-plugins=ServiceAccount",
		fmt.Sprintf("--enable-aggregator-routing=%v", true),
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

	upCh := recorder.Reset()
	_, err = client.CoreV1().Pods("default").Create(context.TODO(), clientAuthMarkerFixture, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

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

func TestManifestSwitching(t *testing.T) {
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert with error: %+v", err)
	}

	recorder := &clientAuthRecorder{}
	mutatingWebhook := httptest.NewUnstartedServer(newSimpleNoopHandler(recorder))
	mutatingWebhook.TLS = &tls.Config{
		RootCAs:      roots,
		Certificates: []tls.Certificate{cert},
	}
	mutatingWebhook.StartTLS()
	defer mutatingWebhook.Close()

	validatingWebhook := httptest.NewUnstartedServer(newSimpleCreateDenyHandler())
	validatingWebhook.StartTLS()
	validatingWebhook.TLS = &tls.Config{
		RootCAs:      roots,
		Certificates: []tls.Certificate{cert},
	}
	defer validatingWebhook.Close()

	manifestFile, err := ioutil.TempFile("", "manifest.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(manifestFile.Name())

	writeManifestFile(t, manifestFile, true, mutatingWebhook, "v1")

	kubeConfigFile, err := ioutil.TempFile("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(kubeConfigFile.Name())

	admissionConfigFile, err := ioutil.TempFile("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(admissionConfigFile.Name())

	if err := ioutil.WriteFile(admissionConfigFile.Name(), []byte(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: MutatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    manifestFile: "`+manifestFile.Name()+`"
- name: ValidatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    manifestFile: "`+manifestFile.Name()+`"
`), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}

	s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--disable-admission-plugins=ServiceAccount",
		fmt.Sprintf("--enable-aggregator-routing=%v", true),
		"--admission-control-config-file=" + admissionConfigFile.Name(),
	}, framework.SharedEtcd())
	defer s.TearDownFn()

	clientConfig := rest.CopyConfig(s.ClientConfig)
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	upCh := recorder.Reset()
	_, err = client.CoreV1().Pods("default").Create(context.TODO(), clientAuthMarkerFixture, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

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

	// change config to validation config
	writeManifestFile(t, manifestFile, false, validatingWebhook, "v1")
	time.Sleep(time.Second * 2)

	_, err = client.CoreV1().Pods("default").Create(context.TODO(), clientAuthMarkerFixture, metav1.CreateOptions{})
	if err == nil {
		t.Fatal(err)
	}

	recorder.Reset()
	// and back again
	writeManifestFile(t, manifestFile, true, mutatingWebhook, "v1")

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

func TestManifesMetricsOnFailure(t *testing.T) {
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert with error: %+v", err)
	}

	validatingWebhook := httptest.NewUnstartedServer(newSimpleCreateDenyHandler())
	validatingWebhook.TLS = &tls.Config{
		RootCAs:      roots,
		Certificates: []tls.Certificate{cert},
	}
	validatingWebhook.StartTLS()
	defer validatingWebhook.Close()

	recorder := &clientAuthRecorder{}
	mutatingWebhook := httptest.NewUnstartedServer(newSimpleNoopHandler(recorder))
	mutatingWebhook.TLS = &tls.Config{
		RootCAs:      roots,
		Certificates: []tls.Certificate{cert},
	}
	mutatingWebhook.StartTLS()
	defer mutatingWebhook.Close()

	manifestFile, err := ioutil.TempFile("", "manifest.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(manifestFile.Name())

	writeManifestFile(t, manifestFile, false, validatingWebhook, "v1")

	kubeConfigFile, err := ioutil.TempFile("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(kubeConfigFile.Name())

	admissionConfigFile, err := ioutil.TempFile("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(admissionConfigFile.Name())

	if err := ioutil.WriteFile(admissionConfigFile.Name(), []byte(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: MutatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    manifestFile: "`+manifestFile.Name()+`"
- name: ValidatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    manifestFile: "`+manifestFile.Name()+`"
`), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}

	s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--disable-admission-plugins=ServiceAccount",
		fmt.Sprintf("--enable-aggregator-routing=%v", true),
		"--admission-control-config-file=" + admissionConfigFile.Name(),
	}, framework.SharedEtcd())
	defer s.TearDownFn()

	clientConfig := rest.CopyConfig(s.ClientConfig)
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = client.CoreV1().Pods("default").Create(context.TODO(), clientAuthMarkerFixture, metav1.CreateOptions{})
	if err == nil {
		t.Fatal("expected error but got nil")
	}

	grabber, err := e2emetrics.NewMetricsGrabber(client, nil, false, false, false, true, false)
	if err != nil {
		t.Fatalf("failed to create MetricsGrabber: %v", err)
	}

	expectGaugeValue(t, grabber, "apiserver_admission_webhook_manifest_error", 0)

	// change config to wrong config
	writeManifestFile(t, manifestFile, false, validatingWebhook, "v1beta1")
	time.Sleep(time.Second * 2)
	expectGaugeValue(t, grabber, "apiserver_admission_webhook_manifest_error", 1)
	upCh := recorder.Reset()

	writeManifestFile(t, manifestFile, true, mutatingWebhook, "v1")

	time.Sleep(time.Second * 2)

	_, err = client.CoreV1().Pods("default").Create(context.TODO(), clientAuthMarkerFixture, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

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

	expectGaugeValue(t, grabber, "apiserver_admission_webhook_manifest_error", 0)

	recorder.Reset()

	writeManifestFile(t, manifestFile, true, mutatingWebhook, "v1beta1")
	time.Sleep(time.Second * 2)
	expectGaugeValue(t, grabber, "apiserver_admission_webhook_manifest_error", 1)

	recorder.Reset()
}

func expectGaugeValue(t *testing.T, grabber *e2emetrics.Grabber, name string, want float64) {
	received, err := grabber.Grab()
	if err != nil {
		t.Fatalf("failed to grab metrics: %v", err)
	}

	metrics := (*e2emetrics.ComponentCollection)(&received)
	val := metrics.APIServerMetrics[name][0].Value

	if float64(val) != want {
		t.Fatalf("expected metric value: %v, got %v", want, val)
	}
}

func TestManifestBasedWebhookFailStart(t *testing.T) {
	manifestFile, err := ioutil.TempFile("", "manifest.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(manifestFile.Name())

	// Invalid Webhook Config; v1beta1 apiVersion
	if err := ioutil.WriteFile(manifestFile.Name(), []byte(`
apiVersion: admissionregistration.k8s.io/v1beta1
kind: MutatingWebhookConfiguration
metadata:
  name: admission.integration.test
webhooks:
- clientConfig:
    caBundle: "`+base64.URLEncoding.EncodeToString(localhostCert)+`"
    url: "https://127.0.0.1/"
  failurePolicy: Fail
  name: admission.integration.test
  rules:
  - apiGroups:
    - '*'
    apiVersions:
    - 'v1'
    operations:
    - '*'
    resources:
    - 'pods'
  sideEffects: None
  admissionReviewVersions: ["v1beta1"]
`), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}

	admissionConfigFile, err := ioutil.TempFile("", "admission-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(admissionConfigFile.Name())

	if err := ioutil.WriteFile(admissionConfigFile.Name(), []byte(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: MutatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    manifestFile: "`+manifestFile.Name()+`"
`), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}

	_, err = kubeapiservertesting.StartTestServer(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--disable-admission-plugins=ServiceAccount",
		fmt.Sprintf("--enable-aggregator-routing=%v", true),
		"--admission-control-config-file=" + admissionConfigFile.Name(),
	}, framework.SharedEtcd())
	expectedError := `no kind "MutatingWebhookConfiguration" is registered for version "admissionregistration.k8s.io/v1beta1"`
	if err == nil || !strings.Contains(err.Error(), expectedError) {
		t.Errorf("Expected error to contain '%s' but got %v", expectedError, err)
	}
}

func writeManifestFile(t *testing.T, manifest *os.File, mutating bool, server *httptest.Server, version string) {
	var typ string
	if mutating {
		typ = "MutatingWebhookConfiguration"
	} else {
		typ = "ValidatingWebhookConfiguration"
	}
	if err := ioutil.WriteFile(manifest.Name(), []byte(`
apiVersion: admissionregistration.k8s.io/`+version+`
kind: `+typ+`
metadata:
  name: admission.integration.test
webhooks:
- clientConfig:
    caBundle: "`+base64.URLEncoding.EncodeToString(localhostCert)+`"
    url: "`+server.URL+`"
  failurePolicy: Fail
  name: admission.integration.test
  rules:
  - apiGroups:
    - '*'
    apiVersions:
    - 'v1'
    operations:
    - '*'
    resources:
    - 'pods'
  sideEffects: None
  admissionReviewVersions: ["v1beta1"]
`), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}
}

func newSimpleCreateDenyHandler() http.Handler {
	allow := func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: true,
			},
		})
	}
	disallow := func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: false,
			},
		})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		data, err := ioutil.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}
		review := v1beta1.AdmissionReview{}
		if err := json.Unmarshal(data, &review); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}

		if review.Request.Operation != v1beta1.Create {
			allow(w)
			return
		}
		if len(review.Request.Object.Raw) == 0 {
			http.Error(w, "Empty object", http.StatusBadRequest)
			return
		}
		pod := &corev1.Pod{}
		if err := json.Unmarshal(review.Request.Object.Raw, pod); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		disallow(w)
		return
	})
}

func newSimpleNoopHandler(recorder *clientAuthRecorder) http.Handler {
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
			http.Error(w, err.Error(), http.StatusBadRequest)
		}
		review := v1beta1.AdmissionReview{}
		if err := json.Unmarshal(data, &review); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}

		if len(review.Request.Object.Raw) == 0 {
			http.Error(w, "Empty object", http.StatusBadRequest)
			return
		}
		pod := &corev1.Pod{}
		if err := json.Unmarshal(review.Request.Object.Raw, pod); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		recorder.MarkerReceived()
		allow(w)
		return
	})
}
