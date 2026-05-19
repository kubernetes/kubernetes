/*
Copyright 2025 The Kubernetes Authors.

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

package httpproxy

import (
	"context"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnettesting "k8s.io/apimachinery/pkg/util/net/testing"
	"k8s.io/client-go/kubernetes"
	kastesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/fakedns"
)

func TestEgressToWebhookWithProxy(t *testing.T) {
	// Go's http.ProxyFromEnvironment bypasses the proxy
	// for localhost/127.0.0.1. To test this, we must make the HTTP client
	// resolve a non-local hostname to 127.0.0.1.
	const (
		webhookHostname = "webhook.example.com"
		proxyHostname   = "proxy.example.com"
	)

	hosts := map[string]string{
		webhookHostname: "127.0.0.1",
		proxyHostname:   "127.0.0.1",
	}

	// Fake DNS server
	dnsServer, err := fakedns.NewServer(hosts)
	if err != nil {
		t.Fatalf("failed to create fake DNS server: %v", err)
	}
	dnsServer.Hijack(t)

	webhookHit := make(chan struct{}, 2)
	webhookServer := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Log("Webhook received request")
		w.Header().Set("Content-Type", "application/json")
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Logf("Failed to read webhook request body: %v", err)
			http.Error(w, "failed to read body", http.StatusBadRequest)
			return
		}
		var review admissionv1.AdmissionReview
		if err := json.Unmarshal(body, &review); err != nil {
			t.Logf("Failed to unmarshal admission review: %v", err)
			http.Error(w, "failed to unmarshal", http.StatusBadRequest)
			return
		}

		resp, err := json.Marshal(admissionv1.AdmissionReview{
			TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
			Response: &admissionv1.AdmissionResponse{UID: review.Request.UID, Allowed: true},
		})
		if err != nil {
			t.Logf("Failed to marshal admission response: %v", err)
			http.Error(w, "failed to marshal response", http.StatusInternalServerError)
			return
		}
		_, _ = w.Write(resp)
		select {
		case webhookHit <- struct{}{}:
		default:
		}
	}))
	webhookServer.EnableHTTP2 = true
	webhookServer.StartTLS()
	defer webhookServer.Close() //nolint:errcheck

	// Proxy server
	proxyHit := make(chan struct{}, 2)
	proxyHandler := utilnettesting.NewHTTPProxyHandler(t, func(r *http.Request) bool {
		t.Logf("Proxy received request for: %s %s", r.Method, r.URL.String())
		select {
		case proxyHit <- struct{}{}:
		default:
		}
		return true
	})
	defer proxyHandler.Wait()

	proxyServer := httptest.NewUnstartedServer(proxyHandler)
	proxyServer.Start()
	defer proxyServer.Close() //nolint:errcheck

	proxyServerURL, err := url.Parse(proxyServer.URL)
	if err != nil {
		t.Fatalf("failed to parse proxy server URL: %v", err)
	}
	proxyURL := fmt.Sprintf("http://%s:%s", proxyHostname, proxyServerURL.Port())

	etcd := framework.SharedEtcd()

	// Construct the webhook URL using our fake hostname and the real port.
	webhookServerURL, err := url.Parse(webhookServer.URL)
	if err != nil {
		t.Fatalf("failed to parse webhook server URL: %v", err)
	}
	webhookURL := fmt.Sprintf("https://%s:%s", webhookHostname, webhookServerURL.Port())
	caCertDER := webhookServer.TLS.Certificates[0].Certificate[0]
	caCertPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: caCertDER,
	})

	sideEffectsNone := admissionregistrationv1.SideEffectClassNone
	failPolicy := admissionregistrationv1.Fail
	webhookConfig := &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "test-webhook-proxy"},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name: "test-proxy.example.com",
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					URL:      &webhookURL,
					CABundle: caCertPEM,
				},
				Rules: []admissionregistrationv1.RuleWithOperations{
					{
						Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
						Rule: admissionregistrationv1.Rule{
							APIGroups:   []string{""},
							APIVersions: []string{"v1"},
							Resources:   []string{"pods"},
						},
					},
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1"},
				TimeoutSeconds:          &[]int32{30}[0],
				FailurePolicy:           &failPolicy,
			},
		},
	}

	// Test with HTTP_PROXY
	t.Log("Testing with HTTP_PROXY")
	t.Setenv("HTTP_PROXY", proxyURL)
	t.Setenv("HTTPS_PROXY", proxyURL)
	t.Setenv("NO_PROXY", "") // Ensure NO_PROXY is cleared
	serverA := kastesting.StartTestServerOrDie(t, kastesting.NewDefaultTestServerOptions(), []string{"--disable-admission-plugins=ServiceAccount"}, etcd)

	clientA, err := kubernetes.NewForConfig(serverA.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create kubernetes client: %v", err)
	}

	_, err = clientA.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(context.Background(), webhookConfig, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create webhook config: %v", err)
	}
	// It can take a moment for the webhook to be consistently available.
	time.Sleep(2 * time.Second)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "test", Image: "test"}},
		},
	}
	_, err = clientA.CoreV1().Pods("default").Create(context.Background(), pod, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("failed to create pod: %v", err)
	}

	select {
	case <-proxyHit:
		t.Log("Proxy was hit as expected")
	case <-time.After(5 * time.Second):
		t.Fatal("Proxy was not hit")
	}
	select {
	case <-webhookHit:
		t.Log("Webhook was hit as expected")
	case <-time.After(5 * time.Second):
		t.Fatal("Webhook was not hit")
	}

	serverA.TearDownFn()

	// Clear channels for the next run
	for len(proxyHit) > 0 {
		<-proxyHit
	}
	for len(webhookHit) > 0 {
		<-webhookHit
	}

	// Part 2: Test with NO_PROXY
	t.Log("Testing with NO_PROXY")
	t.Setenv("HTTP_PROXY", proxyURL)
	t.Setenv("HTTPS_PROXY", proxyURL)
	// Use the fake hostname in NO_PROXY
	t.Setenv("NO_PROXY", strings.Join([]string{webhookHostname, "127.0.0.1", "localhost"}, ","))

	serverB := kastesting.StartTestServerOrDie(t, kastesting.NewDefaultTestServerOptions(), []string{"--disable-admission-plugins=ServiceAccount"}, etcd)
	defer serverB.TearDownFn()

	clientB, err := kubernetes.NewForConfig(serverB.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create kubernetes client: %v", err)
	}

	pod2 := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod-2"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "test", Image: "test"}},
		},
	}
	_, err = clientB.CoreV1().Pods("default").Create(context.Background(), pod2, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create pod: %v", err)
	}

	select {
	case <-webhookHit:
		t.Log("Webhook was hit as expected")
	case <-time.After(5 * time.Second):
		t.Fatal("Webhook was not hit")
	}

	select {
	case <-proxyHit:
		t.Fatal("Proxy was hit, but should have been bypassed by NO_PROXY")
	case <-time.After(2 * time.Second):
		t.Log("Proxy was not hit, as expected")
	}
	// It needs to break the proxy connection or it will panic at cleanup
	webhookServer.Close()
}
