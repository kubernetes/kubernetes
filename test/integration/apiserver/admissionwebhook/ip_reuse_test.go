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

package admissionwebhook

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

type dynamicServiceResolver struct {
	lock      sync.Mutex
	targetURL string
}

func (r *dynamicServiceResolver) SetTarget(url string) {
	r.lock.Lock()
	defer r.lock.Unlock()
	r.targetURL = url
}

func (r *dynamicServiceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	r.lock.Lock()
	defer r.lock.Unlock()
	return url.Parse(r.targetURL)
}

// TestWebhookConnectionPoolIPReuse asserts that the API Server's webhook client
// isolates its HTTP connection pool by the resolved backend IP address.
// If a service's resolved endpoint changes to a new IP, requests must not
// be routed to the old connection in the pool, preventing IP-reuse security bypasses.
func TestWebhookConnectionPoolIPReuse(t *testing.T) {
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert: %v", err)
	}

	// Start Server A
	var serverACalls int32
	serverAListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	serverA := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&serverACalls, 1)
			body, err := io.ReadAll(r.Body)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			var review struct {
				Request struct {
					UID string `json:"uid"`
				} `json:"request"`
			}
			if err := json.Unmarshal(body, &review); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(&admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "admission.k8s.io/v1",
					Kind:       "AdmissionReview",
				},
				Response: &admissionv1.AdmissionResponse{
					UID:     types.UID(review.Request.UID),
					Allowed: true,
				},
			})
		}),
		TLSConfig: &tls.Config{
			Certificates: []tls.Certificate{cert},
		},
	}
	go func() {
		_ = serverA.ServeTLS(serverAListener, "", "")
	}()
	defer func() { _ = serverA.Close() }()

	// Start Server B
	var serverBCalls int32
	serverBListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	serverB := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&serverBCalls, 1)
			body, err := io.ReadAll(r.Body)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			var review struct {
				Request struct {
					UID string `json:"uid"`
				} `json:"request"`
			}
			if err := json.Unmarshal(body, &review); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(&admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "admission.k8s.io/v1",
					Kind:       "AdmissionReview",
				},
				Response: &admissionv1.AdmissionResponse{
					UID:     types.UID(review.Request.UID),
					Allowed: true,
				},
			})
		}),
		TLSConfig: &tls.Config{
			Certificates: []tls.Certificate{cert},
		},
	}
	go func() {
		_ = serverB.ServeTLS(serverBListener, "", "")
	}()
	defer func() { _ = serverB.Close() }()

	urlA := "https://" + serverAListener.Addr().String()
	urlB := "https://" + serverBListener.Addr().String()

	// Set up dynamic service resolver starting with Server A
	resolver := &dynamicServiceResolver{targetURL: urlA}
	t.Cleanup(app.SetServiceResolverForTests(resolver))

	// Start Test API Server
	s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--disable-admission-plugins=ServiceAccount",
	}, framework.SharedEtcd())
	defer s.TearDownFn()

	clientConfig := rest.CopyConfig(s.ClientConfig)
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatalf("Unexpected error creating client: %v", err)
	}

	// Register Validating Webhook Configuration using a ServiceReference
	fail := admissionregistrationv1.Fail
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone
	webhookConfig, err := client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(context.TODO(), &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "ip-reuse.integration.test"},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name: "ip-reuse.integration.test",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				CABundle: localhostCert,
				Service: &admissionregistrationv1.ServiceReference{
					Namespace: "test",
					Name:      "webhook",
					Port:      ptr.To[int32](443),
				},
			},
			Rules: []admissionregistrationv1.RuleWithOperations{{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
				Rule: admissionregistrationv1.Rule{
					APIGroups:   []string{""},
					APIVersions: []string{"v1"},
					Resources:   []string{"pods"},
				},
			}},
			FailurePolicy:           &fail,
			SideEffects:             &sideEffectsNone,
			AdmissionReviewVersions: []string{"v1"},
		}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create validating webhook config: %v", err)
	}
	defer func() {
		_ = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(context.TODO(), webhookConfig.Name, metav1.DeleteOptions{})
	}()

	// Request 1: Resolves to Server A. Trigger hook.
	podFixture := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod-1", Namespace: "default"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "test", Image: "nginx"}},
		},
	}

	// Wait for webhook to become ready/active
	err = wait.PollUntilContextTimeout(context.TODO(), 100*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err = client.CoreV1().Pods("default").Create(ctx, podFixture, metav1.CreateOptions{})
		if err == nil {
			_ = client.CoreV1().Pods("default").Delete(ctx, podFixture.Name, metav1.DeleteOptions{})
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Webhook failed to become active: %v", err)
	}

	// Reset call counts after warm-up
	atomic.StoreInt32(&serverACalls, 0)
	atomic.StoreInt32(&serverBCalls, 0)

	// Execute Request 1 (Must go to Server A)
	pod1 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-a", Namespace: "default"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "test", Image: "nginx"}},
		},
	}
	_, err = client.CoreV1().Pods("default").Create(context.TODO(), pod1, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("First pod creation failed: %v", err)
	}
	_ = client.CoreV1().Pods("default").Delete(context.TODO(), pod1.Name, metav1.DeleteOptions{})

	if atomic.LoadInt32(&serverACalls) != 1 {
		t.Fatalf("Expected 1 call to Server A, got %d", serverACalls)
	}

	// Update Endpoint Resolution to Server B (simulating Pod upgrade/IP change)
	resolver.SetTarget(urlB)

	// Execute Request 2 (Must go to Server B)
	pod2 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-b", Namespace: "default"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "test", Image: "nginx"}},
		},
	}
	_, err = client.CoreV1().Pods("default").Create(context.TODO(), pod2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Second pod creation failed: %v", err)
	}
	_ = client.CoreV1().Pods("default").Delete(context.TODO(), pod2.Name, metav1.DeleteOptions{})

	// Assertions
	// If the connection pool was NOT isolated by IP, the second request would reuse
	// the idle connection to Server A.
	// With the fix, it must correctly connect to Server B.
	if atomic.LoadInt32(&serverBCalls) != 1 {
		t.Errorf("Expected 1 call to Server B (New Pod), got %d. The request was incorrectly routed/stuck!", serverBCalls)
	}
	if atomic.LoadInt32(&serverACalls) != 1 {
		t.Errorf("Expected exactly 1 call to Server A (Old Pod), got %d. Stale connection reuse detected!", serverACalls)
	}
}
