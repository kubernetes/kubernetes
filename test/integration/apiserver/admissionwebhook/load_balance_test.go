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
	"net"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/api/admission/v1beta1"
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

const (
	testLoadBalanceClientUsername = "webhook-balance-integration-client"
)

type staticURLServiceResolver string

func (u staticURLServiceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return url.Parse(string(u))
}

// TestWebhookLoadBalance ensures that the admission webhook opens multiple connections to backends to satisfy concurrent requests
func TestWebhookLoadBalance(t *testing.T) {
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert with error: %+v", err)
	}

	tests := []struct {
		name     string
		http2    bool
		expected int64
	}{
		{
			name:     "10 connections when using http1",
			http2:    false,
			expected: 10,
		},
		{
			name:     "1 connections when using http2",
			http2:    true,
			expected: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			localListener, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				if localListener, err = net.Listen("tcp6", "[::1]:0"); err != nil {
					t.Fatal(err)
				}
			}
			trackingListener := &connectionTrackingListener{delegate: localListener}

			recorder := &connectionRecorder{}
			handler := newLoadBalanceWebhookHandler(recorder)
			httpServer := &http.Server{
				Handler: handler,
				TLSConfig: &tls.Config{
					RootCAs:      roots,
					Certificates: []tls.Certificate{cert},
				},
			}
			go func() {
				_ = httpServer.ServeTLS(trackingListener, "", "")
			}()
			defer func() {
				_ = httpServer.Close()
			}()

			webhookURL := "https://" + localListener.Addr().String()
			t.Cleanup(app.SetServiceResolverForTests(staticURLServiceResolver(webhookURL)))

			s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
				"--disable-admission-plugins=ServiceAccount",
			}, framework.SharedEtcd())
			defer s.TearDownFn()

			// Configure a client with a distinct user name so that it is easy to distinguish requests
			// made by the client from requests made by controllers. We use this to filter out requests
			// before recording them to ensure we don't accidentally mistake requests from controllers
			// as requests made by the client.
			clientConfig := rest.CopyConfig(s.ClientConfig)
			clientConfig.QPS = 100
			clientConfig.Burst = 200
			clientConfig.Impersonate.UserName = testLoadBalanceClientUsername
			clientConfig.Impersonate.Groups = []string{"system:masters", "system:authenticated"}
			client, err := clientset.NewForConfig(clientConfig)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			_, err = client.CoreV1().Pods("default").Create(context.TODO(), loadBalanceMarkerFixture, metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			upCh := recorder.Reset()
			ns := "load-balance"
			_, err = client.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}, metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			webhooksClientConfig := admissionregistrationv1.WebhookClientConfig{
				CABundle: localhostCert,
			}
			if tc.http2 {
				webhooksClientConfig.URL = &webhookURL
			} else {
				webhooksClientConfig.Service = &admissionregistrationv1.ServiceReference{
					Namespace: "test",
					Name:      "webhook",
				}
			}
			fail := admissionregistrationv1.Fail
			mutatingCfg, err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(context.TODO(), &admissionregistrationv1.MutatingWebhookConfiguration{
				ObjectMeta: metav1.ObjectMeta{Name: "admission.integration.test"},
				Webhooks: []admissionregistrationv1.MutatingWebhook{{
					Name:         "admission.integration.test",
					ClientConfig: webhooksClientConfig,
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

			// wait until new webhook is called the first time
			if err := wait.PollUntilContextTimeout(context.TODO(), time.Millisecond*5, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
				_, err = client.CoreV1().Pods("default").Patch(ctx, loadBalanceMarkerFixture.Name, types.JSONPatchType, []byte("[]"), metav1.PatchOptions{})
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

			pod := func() *corev1.Pod {
				return &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:    ns,
						GenerateName: "loadbalance-",
					},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "fake-name",
							Image: "fakeimage",
						}},
					},
				}
			}

			// Submit 10 parallel requests
			wg := &sync.WaitGroup{}
			for range 10 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					_, err := client.CoreV1().Pods(ns).Create(context.TODO(), pod(), metav1.CreateOptions{})
					if err != nil {
						t.Error(err)
					}
				}()
			}
			wg.Wait()

			actual := trackingListener.connections.Load()
			if tc.http2 && actual != tc.expected {
				t.Errorf("expected %d connections, got %d", tc.expected, actual)
			}
			if !tc.http2 && actual < tc.expected {
				t.Errorf("expected at least %d connections, got %d", tc.expected, actual)
			}
			trackingListener.Reset()

			// Submit 10 more parallel requests
			wg = &sync.WaitGroup{}
			for range 10 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					_, err := client.CoreV1().Pods(ns).Create(context.TODO(), pod(), metav1.CreateOptions{})
					if err != nil {
						t.Error(err)
					}
				}()
			}
			wg.Wait()

			if actual := trackingListener.connections.Load(); actual > 0 {
				t.Errorf("expected no additional connections (reusing kept-alive connections), got %d", actual)
			}
		})
	}

}

type connectionRecorder struct {
	mu     sync.Mutex
	upCh   chan struct{}
	upOnce sync.Once
}

// Reset zeros out all counts and returns a channel that is closed when the first admission of the
// marker object is received.
func (i *connectionRecorder) Reset() chan struct{} {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.upCh = make(chan struct{})
	i.upOnce = sync.Once{}
	return i.upCh
}

func (i *connectionRecorder) MarkerReceived() {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.upOnce.Do(func() {
		close(i.upCh)
	})
}

func newLoadBalanceWebhookHandler(recorder *connectionRecorder) http.Handler {
	allow := func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1beta1.AdmissionReview{
			Response: &v1beta1.AdmissionResponse{
				Allowed: true,
			},
		})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Println(r.Proto)
		defer r.Body.Close()
		data, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), 400)
		}
		review := v1beta1.AdmissionReview{}
		if err := json.Unmarshal(data, &review); err != nil {
			http.Error(w, err.Error(), 400)
		}
		if review.Request.UserInfo.Username != testLoadBalanceClientUsername {
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

		// When resetting between tests, a marker object is patched until this webhook
		// observes it, at which point it is considered ready.
		if pod.Namespace == loadBalanceMarkerFixture.Namespace && pod.Name == loadBalanceMarkerFixture.Name {
			recorder.MarkerReceived()
			allow(w)
			return
		}

		// simulate a loaded backend
		time.Sleep(2 * time.Second)
		allow(w)
	})
}

var loadBalanceMarkerFixture = &corev1.Pod{
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

type connectionTrackingListener struct {
	connections atomic.Int64
	delegate    net.Listener
}

func (c *connectionTrackingListener) Reset() {
	c.connections.Store(0)
}

func (c *connectionTrackingListener) Accept() (net.Conn, error) {
	conn, err := c.delegate.Accept()
	if err == nil {
		c.connections.Add(1)
	}
	return conn, err
}
func (c *connectionTrackingListener) Close() error {
	return c.delegate.Close()
}
func (c *connectionTrackingListener) Addr() net.Addr {
	return c.delegate.Addr()
}

// TestWebhookLoadBalanceAcrossEndpoints ensures that the webhook client distributes sequential requests
// across different endpoint IPs behind a service when using a ServiceReference.
// Prior to the host-rewrite pool isolation fix, all sequential requests would reuse the first established
// TCP connection from the single shared pool, completely starving other endpoints.
func TestWebhookLoadBalanceAcrossEndpoints(t *testing.T) {
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(localhostCert) {
		t.Fatal("Failed to append Cert from PEM")
	}
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to build cert with error: %+v", err)
	}

	// 1. Start Server A
	var serverACalls int64
	serverAListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	serverA := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt64(&serverACalls, 1)
			defer func() { _ = r.Body.Close() }()
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

	// 2. Start Server B
	var serverBCalls int64
	serverBListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	serverB := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt64(&serverBCalls, 1)
			defer func() { _ = r.Body.Close() }()
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

	// Parse URLs to extract target hosts (IP:Port)
	urlA, _ := url.Parse("https://" + serverAListener.Addr().String())
	urlB, _ := url.Parse("https://" + serverBListener.Addr().String())

	// Set up a round-robin service resolver that returns alternate endpoints for each resolve call
	var resolveIndex int64
	resolver := funcServiceResolver(func(namespace, name string, port int32) (*url.URL, error) {
		idx := atomic.AddInt64(&resolveIndex, 1)
		if idx%2 == 0 {
			return urlA, nil
		}
		return urlB, nil
	})

	t.Cleanup(app.SetServiceResolverForTests(resolver))

	// 3. Start Test API Server
	s := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), []string{
		"--disable-admission-plugins=ServiceAccount",
	}, framework.SharedEtcd())
	defer s.TearDownFn()

	clientConfig := rest.CopyConfig(s.ClientConfig)
	client, err := clientset.NewForConfig(clientConfig)
	if err != nil {
		t.Fatalf("Unexpected error creating client: %v", err)
	}

	// 4. Register a Validating Webhook with ServiceReference
	fail := admissionregistrationv1.Fail
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone
	webhookConfig, err := client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(context.TODO(), &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "load-balance-endpoints.integration.test"},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name: "load-balance-endpoints.integration.test",
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

	// 5. Warm up the webhook until it is ready/active
	podFixture := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "warmup-pod", Namespace: "default"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "test", Image: "nginx"}},
		},
	}
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

	// Reset counts
	atomic.StoreInt64(&serverACalls, 0)
	atomic.StoreInt64(&serverBCalls, 0)

	// 6. Send 100 sequential requests to trigger validating webhook
	for i := range 100 {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("test-pod-%d", i), Namespace: "default"},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "test", Image: "nginx"}},
			},
		}
		_, err = client.CoreV1().Pods("default").Create(context.TODO(), pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Request %d failed: %v", i, err)
		}
		_ = client.CoreV1().Pods("default").Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
	}

	// 7. Assertions
	callsA := atomic.LoadInt64(&serverACalls)
	callsB := atomic.LoadInt64(&serverBCalls)

	t.Logf("Sequential requests distribution: ServerA = %d calls, ServerB = %d calls", callsA, callsB)

	// Both servers MUST receive a balanced portion of calls under the pool isolation fix.
	// We assert a minimum 70-30 distribution split to verify dynamic load-balancing.
	if callsA < 30 || callsB < 30 {
		t.Errorf("Expected balanced load-balancing across both servers (minimum 70-30 split), but got ServerA = %d calls, ServerB = %d calls. Stale connection pool reuse or starvation detected!", callsA, callsB)
	}
}

type funcServiceResolver func(namespace, name string, port int32) (*url.URL, error)

func (f funcServiceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return f(namespace, name, port)
}
