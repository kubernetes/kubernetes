/*
Copyright 2023 The Kubernetes Authors.

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
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

var (
	runtimeSchemeGVKTest = runtime.NewScheme()
	codecFactoryGVKTest  = serializer.NewCodecFactory(runtimeSchemeGVKTest)
	deserializerGVKTest  = codecFactoryGVKTest.UniversalDeserializer()
)

type admissionTypeChecker struct {
	mu       sync.Mutex
	upCh     chan struct{}
	upOnce   sync.Once
	requests []*admissionv1.AdmissionRequest
}

func (r *admissionTypeChecker) Reset() chan struct{} {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.upCh = make(chan struct{})
	r.upOnce = sync.Once{}
	r.requests = []*admissionv1.AdmissionRequest{}
	return r.upCh
}

func (r *admissionTypeChecker) TypeCheck(req *admissionv1.AdmissionRequest, version string) *admissionv1.AdmissionResponse {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requests = append(r.requests, req)
	raw := req.Object.Raw
	var into runtime.Object
	if _, gvk, err := deserializerGVKTest.Decode(raw, nil, into); err != nil {
		if gvk.Version != version {
			return &admissionv1.AdmissionResponse{
				UID:     req.UID,
				Allowed: false,
			}
		}
	}

	return &admissionv1.AdmissionResponse{
		UID:     req.UID,
		Allowed: true,
	}
}

func (r *admissionTypeChecker) MarkerReceived() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.upOnce.Do(func() {
		close(r.upCh)
	})
}

func newAdmissionTypeCheckerHandler(recorder *admissionTypeChecker) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		data, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), 400)
		}
		review := admissionv1.AdmissionReview{}
		if err := json.Unmarshal(data, &review); err != nil {
			http.Error(w, err.Error(), 400)
		}

		switch r.URL.Path {
		case "/marker":
			recorder.MarkerReceived()
			return
		case "/v1":
			review.Response = recorder.TypeCheck(review.Request, "v1")
		case "/v2":
			review.Response = recorder.TypeCheck(review.Request, "v2")
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(review); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}

	})
}

// Test_MutatingWebhookConvertsGVKWithMatchPolicyEquivalent tests if a equivalent resource is properly converted between mutating webhooks
func Test_MutatingWebhookConvertsGVKWithMatchPolicyEquivalent(t *testing.T) {

	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert with error: %+v", err)
	}

	typeChecker := &admissionTypeChecker{}

	webhookServer := httptest.NewUnstartedServer(newAdmissionTypeCheckerHandler(typeChecker))
	webhookServer.TLS = &tls.Config{
		RootCAs:      roots,
		Certificates: []tls.Certificate{cert},
	}
	webhookServer.StartTLS()
	defer webhookServer.Close()

	upCh := typeChecker.Reset()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--disable-admission-plugins=ServiceAccount",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(server.ClientConfig), false, versionedCustomResourceDefinition())
	if err != nil {
		t.Fatal(err)
	}

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	// Write markers to a separate namespace to avoid cross-talk
	markerNs := "marker"
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: markerNs}}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Create a marker object to use to check for the webhook configurations to be ready.
	marker, err := client.CoreV1().Pods(markerNs).Create(context.TODO(), newMarkerPodGVKConversion(markerNs), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	equivalent := admissionregistrationv1.Equivalent
	ignore := admissionregistrationv1.Ignore

	v1Endpoint := webhookServer.URL + "/v1"
	markerEndpoint := webhookServer.URL + "/marker"
	v2Endpoint := webhookServer.URL + "/v2"
	mutatingWebhook := &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "admission.integration.test",
		},
		Webhooks: []admissionregistrationv1.MutatingWebhook{
			{
				Name: "admission.integration.test.v2",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{"awesome.example.com"},
						APIVersions: []string{"v2"},
						Resources:   []string{"*/*"},
					},
				}},
				MatchPolicy: &equivalent,
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					URL:      &v2Endpoint,
					CABundle: localhostCert,
				},
				FailurePolicy:           &ignore,
				SideEffects:             &noSideEffects,
				AdmissionReviewVersions: []string{"v1"},
				MatchConditions: []admissionregistrationv1.MatchCondition{
					{
						Name:       "test-v2",
						Expression: "object.apiVersion == 'awesome.example.com/v2'",
					},
				},
			},
			{
				Name: "admission.integration.test",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{"awesome.example.com"},
						APIVersions: []string{"v1"},
						Resources:   []string{"*/*"},
					},
				}},
				MatchPolicy: &equivalent,
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					URL:      &v1Endpoint,
					CABundle: localhostCert,
				},
				SideEffects:             &noSideEffects,
				AdmissionReviewVersions: []string{"v1"},
				MatchConditions: []admissionregistrationv1.MatchCondition{
					{
						Name:       "test-v1",
						Expression: "object.apiVersion == 'awesome.example.com/v1'",
					},
				},
			},
			{
				Name: "admission.integration.test.marker",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.OperationAll},
					Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"pods"}},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					URL:      &markerEndpoint,
					CABundle: localhostCert,
				},
				NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{
					corev1.LabelMetadataName: "marker",
				}},
				ObjectSelector:          &metav1.LabelSelector{MatchLabels: map[string]string{"marker": "true"}},
				SideEffects:             &noSideEffects,
				AdmissionReviewVersions: []string{"v1"},
			},
		},
	}

	mutatingCfg, err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(context.TODO(), mutatingWebhook, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(context.TODO(), mutatingCfg.GetName(), metav1.DeleteOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}()

	// wait until new webhook is called the first time
	if err := wait.PollImmediate(time.Millisecond*5, wait.ForeverTestTimeout, func() (bool, error) {
		_, err = client.CoreV1().Pods(markerNs).Patch(context.TODO(), marker.Name, types.JSONPatchType, []byte("[]"), metav1.PatchOptions{})
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
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	v1Resource := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "awesome.example.com" + "/" + "v1",
			"kind":       "Panda",
			"metadata": map[string]interface{}{
				"name": "v1-bears",
			},
		},
	}

	v2Resource := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "awesome.example.com" + "/" + "v2",
			"kind":       "Panda",
			"metadata": map[string]interface{}{
				"name": "v2-bears",
			},
		},
	}

	_, err = dynamicClient.Resource(schema.GroupVersionResource{Group: "awesome.example.com", Version: "v1", Resource: "pandas"}).Create(context.TODO(), v1Resource, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("error1 %v", err.Error())
	}

	_, err = dynamicClient.Resource(schema.GroupVersionResource{Group: "awesome.example.com", Version: "v2", Resource: "pandas"}).Create(context.TODO(), v2Resource, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("error2 %v", err.Error())
	}

	if len(typeChecker.requests) != 4 {
		t.Errorf("expected 4 request got %v", len(typeChecker.requests))
	}
}

func newMarkerPodGVKConversion(namespace string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      "marker",
			Labels: map[string]string{
				"marker": "true",
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:  "fake-name",
				Image: "fakeimage",
			}},
		},
	}
}

// Copied from etcd.GetCustomResourceDefinitionData
func versionedCustomResourceDefinition() *apiextensionsv1.CustomResourceDefinition {
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pandas.awesome.example.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "awesome.example.com",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema:  fixtures.AllowAllSchema(),
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
						Scale: &apiextensionsv1.CustomResourceSubresourceScale{
							SpecReplicasPath:   ".spec.replicas",
							StatusReplicasPath: ".status.replicas",
							LabelSelectorPath:  func() *string { path := ".status.selector"; return &path }(),
						},
					},
				},
				{
					Name:    "v2",
					Served:  true,
					Storage: false,
					Schema:  fixtures.AllowAllSchema(),
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
						Scale: &apiextensionsv1.CustomResourceSubresourceScale{
							SpecReplicasPath:   ".spec.replicas",
							StatusReplicasPath: ".status.replicas",
							LabelSelectorPath:  func() *string { path := ".status.selector"; return &path }(),
						},
					},
				},
			},
			Scope: apiextensionsv1.ClusterScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "pandas",
				Kind:   "Panda",
			},
		},
	}
}
