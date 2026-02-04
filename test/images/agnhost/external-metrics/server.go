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

package externalmetrics

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/spf13/cobra"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/options"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

// CmdExternalMetricsServer is used by agnhost Cobra.
var CmdExternalMetricsServer = &cobra.Command{
	Use:   "external-metrics",
	Short: "Starts an external metrics server for testing",
	Long:  "Starts an HTTPS server that implements the external metrics API for testing HPA with external metrics",
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

var provider *metricProvider

var (
	port        int
	serviceName string
	serviceNs   string
)

func init() {
	CmdExternalMetricsServer.Flags().IntVar(&port, "port", 6443, "Port number.")
	CmdExternalMetricsServer.Flags().StringVar(&serviceName, "service-name", "external-metrics-server", "Name of the external metrics service.")
	CmdExternalMetricsServer.Flags().StringVar(&serviceNs, "service-namespace", "default", "Namespace of the external metrics service.")
}

func main(cmd *cobra.Command, args []string) {

	// Initialize the metric provider
	provider = NewConfigurableProvider()
	// Create some default metrics
	provider.createMetric("queue_messages_ready", nil, 100, false)
	provider.createMetric("http_requests_total", nil, 500, false)
	secureServing := options.NewSecureServingOptions()
	secureServing.BindPort = port
	secureServing.ServerCert.CertDirectory = "/tmp/cert"
	secureServing.ServerCert.PairName = "apiserver"
	// Generate self-signed TLS certificates if none exist. This allows the server to run with HTTPS
	// without requiring manually provisioned certificates. The certs are valid for "localhost" and
	// the loopback IP 127.0.0.1. The second parameter (nil) means no additional alternate names.
	if err := secureServing.MaybeDefaultWithSelfSignedCerts(
		"localhost",
		nil,
		[]net.IP{netutils.ParseIPSloppy("127.0.0.1")},
	); err != nil {
		klog.Fatalf("Error creating self-signed certs: %v", err)
	}

	var servingInfo *genericapiserver.SecureServingInfo
	if err := secureServing.ApplyTo(&servingInfo); err != nil {
		klog.Fatalf("Error applying secure serving: %v", err)
	}

	if servingInfo == nil {
		klog.Fatal("SecureServingInfo is nil")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/apis/external.metrics.k8s.io/", handleMetrics)
	mux.HandleFunc("/apis/external.metrics.k8s.io", handleMetrics)
	mux.HandleFunc("/healthz", healthz)
	mux.HandleFunc("/readyz", healthz)
	mux.HandleFunc("/fail/", failMetric)
	mux.HandleFunc("/set/", setMetric)
	mux.HandleFunc("/create/", createMetric)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	klog.InfoS("Starting server on", "address", servingInfo.Listener.Addr().String())

	stoppedCh, listenerStoppedCh, err := servingInfo.Serve(mux, 30*time.Second, ctx.Done())
	if err != nil {
		klog.Fatalf("Error starting server: %v", err)
	}

	<-listenerStoppedCh
	<-stoppedCh

}

func healthz(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write([]byte("ok")); err != nil {
		klog.ErrorS(err, "failed to write healthz response")
	}
}

func handleMetrics(w http.ResponseWriter, r *http.Request) {
	klog.InfoS("", "method", r.Method, "path", r.URL.Path)
	w.Header().Set("Content-Type", "application/json")

	path := strings.TrimPrefix(r.URL.Path, "/apis/external.metrics.k8s.io")
	path = strings.TrimPrefix(path, "/")

	if path == "" {
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"kind":       "APIGroup",
			"apiVersion": "v1",
			"name":       "external.metrics.k8s.io",
			"versions": []map[string]string{
				{"groupVersion": "external.metrics.k8s.io/v1beta1", "version": "v1beta1"},
			},
			"preferredVersion": map[string]string{
				"groupVersion": "external.metrics.k8s.io/v1beta1",
				"version":      "v1beta1",
			},
		}); err != nil {
			klog.ErrorS(err, "failed to encode APIGroup response")
		}
		return
	}

	if path == "v1beta1" || path == "v1beta1/" {
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"kind":         "APIResourceList",
			"apiVersion":   "v1",
			"groupVersion": "external.metrics.k8s.io/v1beta1",
			"resources": []map[string]interface{}{
				{"name": "*", "namespaced": true, "kind": "ExternalMetricValueList", "verbs": []string{"get"}},
			},
		}); err != nil {
			klog.ErrorS(err, "failed to encode APIResourceList response")
		}
		return
	}

	parts := strings.Split(path, "/")
	if len(parts) >= 4 && parts[1] == "namespaces" {
		metricName := parts[3]

		// Parse label selector from query parameter
		labelSelector := r.URL.Query().Get("labelSelector")
		labels := parseLabels(labelSelector)

		// Get all metrics matching the name and label selector
		metrics := provider.getMetrics(metricName, labels)

		if len(metrics) == 0 {
			errorMessage := fmt.Sprintf("no metric name called %s", metricName)
			if len(labels) > 0 {
				errorMessage = fmt.Sprintf("no metric name called %s with labels %v", metricName, labels)
			}
			http.Error(w, errorMessage, http.StatusNotFound)
			return
		}

		// Build response items
		var items []map[string]interface{}
		for _, metric := range metrics {
			if metric.shouldFail {
				errorMessage := fmt.Sprintf("metric %s is configured to fail", metricName)
				http.Error(w, errorMessage, http.StatusInternalServerError)
				return
			}

			items = append(items, map[string]interface{}{
				"metricName":   metric.metricName,
				"metricLabels": metric.labels,
				"timestamp":    time.Now().UTC().Format(time.RFC3339), // TODO: maybe we need to make this configurable?
				"value":        fmt.Sprintf("%d", metric.value),
			})
		}

		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"kind":       "ExternalMetricValueList",
			"apiVersion": "external.metrics.k8s.io/v1beta1",
			"metadata":   map[string]interface{}{},
			"items":      items,
		}); err != nil {
			klog.ErrorS(err, "failed to encode ExternalMetricValueList response")
		}
		return
	}

	http.NotFound(w, r)
}

// failMetric handles requests to mark a metric as failing or not failing.
// Supports label selectors via the "labels" query parameter (format: key1=value1,key2=value2).
func failMetric(w http.ResponseWriter, r *http.Request) {
	klog.InfoS("", "method", r.Method, "path", r.URL.Path)
	w.Header().Set("Content-Type", "application/json")

	path := strings.TrimPrefix(r.URL.Path, "/fail/")
	parts := strings.Split(path, "/")
	if len(parts) == 0 || parts[0] == "" {
		http.Error(w, "metric name required", http.StatusBadRequest)
		return
	}

	metricName := parts[0]
	failParam := r.URL.Query().Get("fail")

	if failParam != "true" && failParam != "false" {
		http.Error(w, "fail param should be true or false", http.StatusBadRequest)
		return
	}

	labelSelector := r.URL.Query().Get("labels")
	labels := parseLabels(labelSelector)
	metricKey := buildMetricKey(metricName, labels)

	if err := provider.setMetricFailure(metricKey, failParam); err != nil {
		klog.ErrorS(err, "failed to set metric failure", "metric", metricName, "labels", labels)
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "success",
		"metric":  metricName,
		"labels":  labels,
		"failing": failParam,
	}); err != nil {
		klog.ErrorS(err, "failed to encode failMetric response")
	}
}

func setMetric(w http.ResponseWriter, r *http.Request) {
	klog.InfoS("", "method", r.Method, "path", r.URL.Path)
	w.Header().Set("Content-Type", "application/json")

	path := strings.TrimPrefix(r.URL.Path, "/set/")
	parts := strings.Split(path, "/")
	if len(parts) == 0 || parts[0] == "" {
		http.Error(w, "metric name required", http.StatusBadRequest)
		return
	}

	metricName := parts[0]
	valueParam := r.URL.Query().Get("value")
	if valueParam == "" {
		http.Error(w, "value param required", http.StatusBadRequest)
		return
	}

	// Parse labels from query parameter
	labelSelector := r.URL.Query().Get("labels")
	labels := parseLabels(labelSelector)
	metricKey := buildMetricKey(metricName, labels)

	var value int
	if _, err := fmt.Sscanf(valueParam, "%d", &value); err != nil {
		http.Error(w, "value param must be an integer", http.StatusBadRequest)
		return
	}

	if err := provider.setMetricValue(metricKey, value); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "success",
		"metric": metricName,
		"value":  value,
	}); err != nil {
		klog.ErrorS(err, "failed to encode setMetric response")
	}
}

func createMetric(w http.ResponseWriter, r *http.Request) {
	klog.InfoS("", "method", r.Method, "path", r.URL.Path)
	w.Header().Set("Content-Type", "application/json")

	path := strings.TrimPrefix(r.URL.Path, "/create/")
	parts := strings.Split(path, "/")
	if len(parts) == 0 || parts[0] == "" {
		http.Error(w, "metric name required", http.StatusBadRequest)
		return
	}

	metricName := parts[0]
	valueParam := r.URL.Query().Get("value")
	if valueParam == "" {
		http.Error(w, "value param required", http.StatusBadRequest)
		return
	}

	var value int
	if _, err := fmt.Sscanf(valueParam, "%d", &value); err != nil {
		http.Error(w, "value param must be an integer", http.StatusBadRequest)
		return
	}

	failParam := r.URL.Query().Get("fail")
	shouldFail := failParam == "true"

	// Parse labels from query parameter
	labelSelector := r.URL.Query().Get("labels")
	labels := parseLabels(labelSelector)

	provider.createMetric(metricName, labels, value, shouldFail)

	w.WriteHeader(http.StatusCreated)
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "created",
		"metric":  metricName,
		"labels":  labels,
		"value":   value,
		"failing": shouldFail,
	}); err != nil {
		klog.ErrorS(err, "failed to encode createMetric response")
	}
}
