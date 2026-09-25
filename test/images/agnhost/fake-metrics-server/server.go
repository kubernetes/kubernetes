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

package fakemetricsserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/spf13/cobra"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/options"
	"k8s.io/klog/v2"
	metricsv1 "k8s.io/metrics/pkg/apis/metrics/v1"
	metricsv1beta1 "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	netutils "k8s.io/utils/net"
)

// CmdFakeMetricsServer is used by agnhost Cobra.
var CmdFakeMetricsServer = &cobra.Command{
	Use:   "fake-metrics-server",
	Short: "Starts a fake metrics server for testing",
	Long:  "Starts an HTTPS server that implements the metrics API for testing HPA with metrics",
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

type metricsServer struct {
	provider *metricProvider
}

var (
	port int
)

func init() {
	CmdFakeMetricsServer.Flags().IntVar(&port, "port", 6443, "Port number.")
}

func main(cmd *cobra.Command, args []string) {
	// Initialize the metric provider
	server := &metricsServer{
		provider: newMetricProvider(),
	}
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
	mux.HandleFunc("/apis/metrics.k8s.io/", server.handleMetrics)
	mux.HandleFunc("/apis/metrics.k8s.io", server.handleMetrics)
	mux.HandleFunc("/healthz", server.healthz)
	mux.HandleFunc("/readyz", server.healthz)
	mux.HandleFunc("/configure", server.configureMetrics)

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

func (s *metricsServer) healthz(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write([]byte("ok")); err != nil {
		klog.ErrorS(err, "failed to write healthz response")
	}
}

func (s *metricsServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "metrics API requires GET", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	path := strings.TrimPrefix(r.URL.Path, "/apis/metrics.k8s.io")
	path = strings.Trim(path, "/")

	switch path {
	case "":
		s.writeAPIGroup(w)
		return
	case "v1":
		s.writeAPIResourceList(w, metricsv1.SchemeGroupVersion)
		return
	case "v1beta1":
		s.writeAPIResourceList(w, metricsv1beta1.SchemeGroupVersion)
		return
	}

	parts := strings.Split(path, "/")
	if len(parts) != 4 ||
		parts[1] != "namespaces" ||
		parts[3] != "pods" {
		http.NotFound(w, r)
		return
	}

	namespace := parts[2]

	selector, err := labels.Parse(r.URL.Query().Get("labelSelector"))
	if err != nil {
		http.Error(w, fmt.Sprintf("invalid label selector: %v", err), http.StatusBadRequest)
		return
	}

	switch parts[0] {
	case "v1":
		metrics := s.provider.listV1(namespace, selector)
		refreshV1Timestamps(&metrics)
		if err := json.NewEncoder(w).Encode(metrics); err != nil {
			klog.ErrorS(err, "failed to encode v1 PodMetricsList")
		}
	case "v1beta1":
		metrics, err := s.provider.listV1beta1(namespace, selector)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to list v1beta1 PodMetrics: %v", err), http.StatusInternalServerError)
			return
		}
		refreshV1beta1Timestamps(&metrics)
		if err := json.NewEncoder(w).Encode(metrics); err != nil {
			klog.ErrorS(err, "failed to encode v1beta1 PodMetricsList")
		}
	default:
		http.NotFound(w, r)
	}
}

func (s *metricsServer) configureMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "configure metrics requires POST", http.StatusMethodNotAllowed)
		return
	}
	var metrics metricsv1.PodMetricsList
	if err := json.NewDecoder(r.Body).Decode(&metrics); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode v1 PodMetricsList: %v", err), http.StatusBadRequest)
		return
	}
	if err := validatePodMetricsList(metrics.TypeMeta, metricsv1.SchemeGroupVersion.String()); err != nil {
		http.Error(w, fmt.Sprintf("invalid v1 PodMetricsList: %v", err), http.StatusBadRequest)
		return
	}
	s.provider.replace(metrics.Items)

	w.WriteHeader(http.StatusOK)
}

func validatePodMetricsList(typeMeta metav1.TypeMeta, expectedVersion string) error {
	if typeMeta.APIVersion != expectedVersion {
		return fmt.Errorf(
			"expected apiVersion %q, got %q",
			expectedVersion,
			typeMeta.APIVersion,
		)
	}
	if typeMeta.Kind != "PodMetricsList" {
		return fmt.Errorf(
			"expected kind %q, got %q",
			"PodMetricsList",
			typeMeta.Kind,
		)
	}
	return nil
}

func (s *metricsServer) writeAPIResourceList(w http.ResponseWriter, groupVersion schema.GroupVersion) {
	resourceList := metav1.APIResourceList{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "APIResourceList",
		},
		GroupVersion: groupVersion.String(),
		APIResources: []metav1.APIResource{{
			Name:       "pods",
			Namespaced: true,
			Kind:       "PodMetrics",
			Verbs:      metav1.Verbs{"list"},
		}},
	}

	if err := json.NewEncoder(w).Encode(resourceList); err != nil {
		klog.ErrorS(err, "failed to encode APIResourceList")
	}
}

func (s *metricsServer) writeAPIGroup(w http.ResponseWriter) {
	group := metav1.APIGroup{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "APIGroup",
		},
		Name: metricsv1.GroupName,
		Versions: []metav1.GroupVersionForDiscovery{
			{
				GroupVersion: metricsv1.SchemeGroupVersion.String(),
				Version:      metricsv1.SchemeGroupVersion.Version,
			},
			{
				GroupVersion: metricsv1beta1.SchemeGroupVersion.String(),
				Version:      metricsv1beta1.SchemeGroupVersion.Version,
			},
		},
		PreferredVersion: metav1.GroupVersionForDiscovery{
			GroupVersion: metricsv1.SchemeGroupVersion.String(),
			Version:      metricsv1.SchemeGroupVersion.Version,
		},
	}

	if err := json.NewEncoder(w).Encode(group); err != nil {
		klog.ErrorS(err, "failed to encode metrics API group")
	}
}

func refreshV1Timestamps(metrics *metricsv1.PodMetricsList) {
	now := metav1.Now()
	for i := range metrics.Items {
		metrics.Items[i].Timestamp = now
	}
}

func refreshV1beta1Timestamps(metrics *metricsv1beta1.PodMetricsList) {
	now := metav1.Now()
	for i := range metrics.Items {
		metrics.Items[i].Timestamp = now
	}
}
