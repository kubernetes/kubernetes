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

package egressselector

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	kubeapiserverapptesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

const (
	webhookName           = "timeout.egress-selector.integration.test"
	webhookNamespace      = "egress-selector-webhook-timeout"
	webhookServiceName    = "egress-selector-webhook"
	webhookLabel          = "egress-selector-webhook-timeout"
	webhookTimeoutSeconds = 5
)

// TestWebhookDialTimeout verifies that a webhook call routed through an HTTP
// CONNECT egress proxy is bounded by the webhook's TimeoutSeconds. The proxy
// used here never answers the CONNECT request, so the webhook call must time out
// and the apiserver must close its connection to the proxy instead of leaking it.
func TestWebhookDialTimeout(t *testing.T) {
	udsName, connectRequestDone := runStallingHTTPConnectProxy(t)

	server := kubeapiserverapptesting.StartTestServerOrDie(
		t,
		kubeapiserverapptesting.NewDefaultTestServerOptions(),
		[]string{fmt.Sprintf("--egress-selector-config-file=%s", writeEgressSelectorConfig(t, udsName))},
		framework.SharedEtcd(),
	)
	t.Cleanup(server.TearDownFn)

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}
	registerStallingWebhook(t, client)

	waitForWebhookTimeout(t, client)

	select {
	case <-connectRequestDone:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("HTTP CONNECT request context was not canceled after the webhook timeout")
	}
}

// writeEgressSelectorConfig writes an egress selector configuration that
// routes all cluster traffic (which includes webhook calls) through an HTTP
// CONNECT proxy listening on the given unix domain socket, and returns the
// config file path.
func writeEgressSelectorConfig(t *testing.T, udsName string) string {
	t.Helper()

	config := fmt.Sprintf(`
apiVersion: apiserver.k8s.io/v1beta1
kind: EgressSelectorConfiguration
egressSelections:
- name: cluster
  connection:
    proxyProtocol: HTTPConnect
    transport:
      uds:
        udsName: %s
`, udsName)

	path := filepath.Join(t.TempDir(), "egress-selector.yaml")
	if err := os.WriteFile(path, []byte(config), 0644); err != nil {
		t.Fatalf("failed to write egress selector config: %v", err)
	}
	return path
}

// registerStallingWebhook creates a validating webhook scoped to a dedicated
// namespace and backed by a service reachable through the egress proxy.
func registerStallingWebhook(t *testing.T, client kubernetes.Interface) {
	t.Helper()

	if _, err := client.CoreV1().Namespaces().Create(t.Context(), &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: webhookNamespace},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create webhook namespace: %v", err)
	}

	if _, err := client.CoreV1().Services(webhookNamespace).Create(t.Context(), &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: webhookServiceName},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{{Port: 443}},
		},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create webhook service: %v", err)
	}

	if _, err := client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(
		t.Context(),
		&admissionregistrationv1.ValidatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: webhookLabel},
			Webhooks: []admissionregistrationv1.ValidatingWebhook{{
				Name: webhookName,
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: webhookNamespace,
						Name:      webhookServiceName,
						Path:      new("/validate"),
						Port:      ptr.To[int32](443),
					},
				},
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{""},
						APIVersions: []string{"v1"},
						Resources:   []string{"configmaps"},
					},
				}},
				FailurePolicy:  ptr.To(admissionregistrationv1.Fail),
				SideEffects:    ptr.To(admissionregistrationv1.SideEffectClassNone),
				TimeoutSeconds: ptr.To[int32](webhookTimeoutSeconds),
				NamespaceSelector: &metav1.LabelSelector{MatchLabels: map[string]string{
					corev1.LabelMetadataName: webhookNamespace,
				}},
				AdmissionReviewVersions: []string{"v1"},
			}},
		},
		metav1.CreateOptions{},
	); err != nil {
		t.Fatalf("failed to create validating webhook configuration: %v", err)
	}
}

// waitForWebhookTimeout creates ConfigMaps until the webhook configuration
// becomes active and rejects one with a timeout error.
func waitForWebhookTimeout(t *testing.T, client kubernetes.Interface) {
	t.Helper()

	var lastErr error
	err := wait.PollUntilContextTimeout(t.Context(), 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		_, lastErr = client.CoreV1().ConfigMaps(webhookNamespace).Create(ctx, &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: webhookLabel + "-",
			},
		}, metav1.CreateOptions{})
		if lastErr == nil {
			return false, nil
		}
		if !strings.Contains(lastErr.Error(), webhookName) {
			return false, lastErr
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("failed waiting for the webhook to reject a ConfigMap (last error: %v): %v", lastErr, err)
	}
	webhookURL := fmt.Sprintf("https://%s.%s.svc:443/validate?timeout=%ds", webhookServiceName, webhookNamespace, webhookTimeoutSeconds)
	gotErr := lastErr.Error()
	if strings.Contains(gotErr, webhookURL) &&
		(strings.HasSuffix(gotErr, context.DeadlineExceeded.Error()) ||
			strings.HasSuffix(gotErr, "(Client.Timeout exceeded while awaiting headers)")) {
		return
	}
	t.Fatalf("unexpected webhook timeout error: %q", gotErr)
}

// runStallingHTTPConnectProxy starts an HTTP CONNECT proxy on a unix domain
// socket that never answers CONNECT requests. It returns the socket path and
// a channel that receives an event whenever a stalled CONNECT request is
// canceled.
func runStallingHTTPConnectProxy(t *testing.T) (string, <-chan struct{}) {
	t.Helper()

	udsName := filepath.Join(t.TempDir(), "proxy.sock")
	listener, err := net.Listen("unix", udsName)
	if err != nil {
		t.Fatalf("failed to listen on UDS: %v", err)
	}

	connectRequestDone := make(chan struct{}, 16)
	server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodConnect {
			http.Error(w, "CONNECT required", http.StatusMethodNotAllowed)
			return
		}
		<-r.Context().Done()
		t.Logf("HTTP CONNECT request canceled: %v", r.Context().Err())
		connectRequestDone <- struct{}{}
	})}
	serveErrCh := make(chan error, 1)
	go func() {
		serveErrCh <- server.Serve(listener)
	}()

	t.Cleanup(func() {
		if err := server.Close(); err != nil {
			t.Logf("failed to close HTTP CONNECT proxy: %v", err)
		}
		if err := <-serveErrCh; err != nil && !errors.Is(err, http.ErrServerClosed) {
			t.Logf("HTTP CONNECT proxy failed: %v", err)
		}
	})

	return udsName, connectRequestDone
}
