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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	admissionreviewv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	// orderAnnotation accumulates the markers of the mutating webhooks that have been
	// invoked, in invocation order.
	orderAnnotation = "mutation-order"
	// orderNamespaceLabel scopes the order-recording webhooks to the test namespace so
	// they cannot interfere with the rest of the cluster.
	orderNamespaceLabel = "mutating-webhook-order-test"
)

// Markers recorded by each webhook. The configuration names they belong to sort as
// a-order-test < b-order-test < b-order-test.static.k8s.io < c-order-test.
const (
	markerAPIA    = "a-api"
	markerAPIB    = "b-api"
	markerStaticB = "b-static"
	markerAPIC    = "c-api"
)

// TestStaticManifestMutatingWebhookOrder verifies the observed invocation order of
// interleaved manifest-based and API-based mutating webhooks for each staticOrder value.
func TestStaticManifestMutatingWebhookOrder(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ManifestBasedAdmissionControlConfig, true)

	webhookServer, caCert := createOrderRecordingWebhookServer(t)
	defer webhookServer.Close()

	// restBaselineOrder is the order observed when b-order-test is still registered
	// through the API. Under ByName, migrating it to b-order-test.static.k8s.io must
	// keep the same relative position.
	restBaselineOrder := []string{markerAPIA, markerAPIB, markerAPIC}

	testcases := []struct {
		name        string
		staticOrder string
		// restB registers b-order-test through the API instead of loading
		// b-order-test.static.k8s.io from the manifests directory.
		restB     bool
		wantOrder []string
	}{
		{
			name:      "REST baseline",
			restB:     true,
			wantOrder: restBaselineOrder,
		},
		{
			name:        "unset",
			staticOrder: "",
			wantOrder:   []string{markerStaticB, markerAPIA, markerAPIC},
		},
		{
			name:        "StaticFirst",
			staticOrder: "StaticFirst",
			wantOrder:   []string{markerStaticB, markerAPIA, markerAPIC},
		},
		{
			name:        "StaticLast",
			staticOrder: "StaticLast",
			wantOrder:   []string{markerAPIA, markerAPIC, markerStaticB},
		},
		{
			name:        "ByName",
			staticOrder: "ByName",
			// The migrated configuration takes the slot b-order-test had before.
			wantOrder: replaceMarker(restBaselineOrder, markerAPIB, markerStaticB),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()

			manifestsDir := t.TempDir()
			apiConfigs := map[string]string{"a-order-test": markerAPIA, "c-order-test": markerAPIC}
			if tc.restB {
				apiConfigs["b-order-test"] = markerAPIB
			} else {
				writeManifestToFile(t, manifestsDir, "mutating.yaml",
					orderMutatingWebhookConfiguration(t, "b-order-test.static.k8s.io", webhookServer.URL+"/"+markerStaticB, caCert))
			}

			server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
				"--admission-control-config-file=" + createOrderAdmissionConfig(t, manifestsDir, tc.staticOrder),
			}, framework.SharedEtcd())
			defer server.TearDownFn()

			client, err := clientset.NewForConfig(rest.CopyConfig(server.ClientConfig))
			if err != nil {
				t.Fatalf("Failed to create client: %v", err)
			}

			ns := "webhook-order-" + strings.ReplaceAll(strings.ToLower(tc.name), " ", "-")
			if _, err := client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:   ns,
					Labels: map[string]string{orderNamespaceLabel: "true"},
				},
			}, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create namespace %s: %v", ns, err)
			}

			// Shared etcd outlives this API server, so the namespace and the webhook
			// configurations must not leak into the next subtest. These defers run
			// before the server teardown defer.
			defer func() {
				if err := client.CoreV1().Namespaces().Delete(context.Background(), ns, metav1.DeleteOptions{}); err != nil {
					t.Errorf("Failed to delete namespace %s: %v", ns, err)
				}
			}()

			var apiConfigNames []string
			defer func() {
				for _, name := range apiConfigNames {
					if err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(context.Background(), name, metav1.DeleteOptions{}); err != nil {
						t.Errorf("Failed to delete MutatingWebhookConfiguration %s: %v", name, err)
					}
				}
			}()
			for name, marker := range apiConfigs {
				cfg := orderMutatingWebhookConfiguration(t, name, webhookServer.URL+"/"+marker, caCert)
				if _, err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(ctx, cfg, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create MutatingWebhookConfiguration %s: %v", name, err)
				}
				apiConfigNames = append(apiConfigNames, name)
			}

			// Poll until every webhook is dispatched: the API-based configurations only
			// take effect once the admission informer has observed them.
			var gotOrder []string
			probe := 0
			err = wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 60*time.Second, true, func(ctx context.Context) (bool, error) {
				probe++
				order, probeErr := observeMutationOrder(ctx, client, ns, probe)
				if probeErr != nil {
					t.Logf("Probe %d failed, retrying: %v", probe, probeErr)
					return false, nil
				}
				gotOrder = order
				return len(gotOrder) == len(tc.wantOrder), nil
			})
			if err != nil {
				t.Fatalf("Timed out waiting for all %d webhooks to be dispatched, last observed order %v: %v", len(tc.wantOrder), gotOrder, err)
			}

			if !slices.Equal(gotOrder, tc.wantOrder) {
				t.Errorf("staticOrder %q: got mutation order %v, want %v", tc.staticOrder, gotOrder, tc.wantOrder)
			}
		})
	}
}

// replaceMarker returns a copy of order with every occurrence of from replaced by to.
func replaceMarker(order []string, from, to string) []string {
	replaced := slices.Clone(order)
	for i, marker := range replaced {
		if marker == from {
			replaced[i] = to
		}
	}
	return replaced
}

// observeMutationOrder creates a ConfigMap and returns the markers recorded by the
// mutating webhooks, in invocation order.
func observeMutationOrder(ctx context.Context, client clientset.Interface, ns string, probe int) ([]string, error) {
	cm, err := client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:        fmt.Sprintf("order-probe-%d", probe),
			Namespace:   ns,
			Annotations: map[string]string{orderAnnotation: ""},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}
	recorded := cm.Annotations[orderAnnotation]
	if len(recorded) == 0 {
		return nil, nil
	}
	return strings.Split(recorded, ","), nil
}

// orderMutatingWebhookConfiguration builds a MutatingWebhookConfiguration whose single
// webhook records its invocation against ConfigMap creates in the test namespace.
func orderMutatingWebhookConfiguration(t *testing.T, name, url string, caCert []byte) *admissionregistrationv1.MutatingWebhookConfiguration {
	t.Helper()
	return &admissionregistrationv1.MutatingWebhookConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admissionregistration.k8s.io/v1",
			Kind:       "MutatingWebhookConfiguration",
		},
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Webhooks: []admissionregistrationv1.MutatingWebhook{{
			Name: "order.webhook.io",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL:      &url,
				CABundle: caCert,
			},
			Rules: []admissionregistrationv1.RuleWithOperations{{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
				Rule:       admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"configmaps"}},
			}},
			AdmissionReviewVersions: []string{"v1"},
			SideEffects:             new(admissionregistrationv1.SideEffectClassNone),
			NamespaceSelector: &metav1.LabelSelector{
				MatchLabels: map[string]string{orderNamespaceLabel: "true"},
			},
			ObjectSelector: &metav1.LabelSelector{},
			FailurePolicy:  new(admissionregistrationv1.Fail),
		}},
	}
}

// createOrderAdmissionConfig writes an admission configuration file enabling
// manifest-based mutating webhooks with the given merge order. An empty staticOrder
// omits the field entirely.
func createOrderAdmissionConfig(t *testing.T, manifestsDir, staticOrder string) string {
	t.Helper()
	admissionConfig := fmt.Sprintf(`
apiVersion: apiserver.k8s.io/v1alpha1
kind: AdmissionConfiguration
plugins:
- name: MutatingAdmissionWebhook
  configuration:
    apiVersion: apiserver.config.k8s.io/v1
    kind: WebhookAdmissionConfiguration
    staticManifestsDir: %q
`, manifestsDir)
	if len(staticOrder) > 0 {
		admissionConfig += fmt.Sprintf("    staticOrder: %s\n", staticOrder)
	}

	configFile, err := os.CreateTemp("", "admission-order-config-*.yaml")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	t.Cleanup(func() { _ = os.Remove(configFile.Name()) })
	if _, err := configFile.WriteString(admissionConfig); err != nil {
		t.Fatalf("Failed to write config: %v", err)
	}
	if err := configFile.Close(); err != nil {
		t.Fatalf("Failed to close config file: %v", err)
	}
	return configFile.Name()
}

// createOrderRecordingWebhookServer creates an HTTPS webhook server that appends the
// marker in the request path to the order annotation of the incoming ConfigMap. Because
// mutating webhooks are dispatched sequentially against the output of the previous one,
// the resulting annotation records the invocation order.
func createOrderRecordingWebhookServer(t *testing.T) (*httptest.Server, []byte) {
	t.Helper()
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("Failed to create key pair: %v", err)
	}

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := readStaticBody(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var review admissionreviewv1.AdmissionReview
		if err := json.Unmarshal(body, &review); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if review.Request == nil {
			http.Error(w, "missing admission request", http.StatusBadRequest)
			return
		}
		var cm corev1.ConfigMap
		if err := json.Unmarshal(review.Request.Object.Raw, &cm); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		recorded := cm.Annotations[orderAnnotation]
		if len(recorded) > 0 {
			recorded += ","
		}
		recorded += strings.TrimPrefix(r.URL.Path, "/")

		patch, err := json.Marshal([]map[string]string{{
			"op":    "replace",
			"path":  "/metadata/annotations/" + orderAnnotation,
			"value": recorded,
		}})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		patchType := admissionreviewv1.PatchTypeJSONPatch
		respBytes, err := json.Marshal(&admissionreviewv1.AdmissionReview{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "admission.k8s.io/v1",
				Kind:       "AdmissionReview",
			},
			Response: &admissionreviewv1.AdmissionResponse{
				UID:       review.Request.UID,
				Allowed:   true,
				Patch:     patch,
				PatchType: &patchType,
			},
		})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(respBytes)
	})

	server := httptest.NewUnstartedServer(handler)
	server.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
	server.StartTLS()

	return server, localhostCert
}
