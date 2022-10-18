/*
Copyright 2021 The Kubernetes Authors.

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

package tracing

import (
	"context"
	"fmt"
	"net"
	"os"
	"strings"
	"testing"
	"time"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	traceservice "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/grpc"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	client "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/tracing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestAPIServerTracing(t *testing.T) {
	// Listen for traces from the API Server before starting it, so the
	// API Server will successfully connect right away during the test.
	listener, err := net.Listen("tcp", "localhost:")
	if err != nil {
		t.Fatal(err)
	}
	// Write the configuration for tracing to a file
	tracingConfigFile, err := os.CreateTemp("", "tracing-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tracingConfigFile.Name())

	if err := os.WriteFile(tracingConfigFile.Name(), []byte(fmt.Sprintf(`
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: TracingConfiguration
endpoint: %s`, listener.Addr().String())), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}
	testAPIServerTracing(t,
		listener,
		[]string{"--tracing-config-file=" + tracingConfigFile.Name()},
	)
}

func TestAPIServerTracingWithEgressSelector(t *testing.T) {
	// Listen for traces from the API Server before starting it, so the
	// API Server will successfully connect right away during the test.
	listener, err := net.Listen("tcp", "localhost:")
	if err != nil {
		t.Fatal(err)
	}
	// Use an egress selector which doesn't have a controlplane config to ensure
	// tracing works in that context. Write the egress selector configuration to a file.
	egressSelectorConfigFile, err := os.CreateTemp("", "egress_selector_configuration.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(egressSelectorConfigFile.Name())

	if err := os.WriteFile(egressSelectorConfigFile.Name(), []byte(`
apiVersion: apiserver.config.k8s.io/v1beta1
kind: EgressSelectorConfiguration
egressSelections:
- name: cluster
  connection:
    proxyProtocol: Direct
    transport:`), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}

	// Write the configuration for tracing to a file
	tracingConfigFile, err := os.CreateTemp("", "tracing-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tracingConfigFile.Name())

	if err := os.WriteFile(tracingConfigFile.Name(), []byte(fmt.Sprintf(`
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: TracingConfiguration
endpoint: %s`, listener.Addr().String())), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}
	testAPIServerTracing(t,
		listener,
		[]string{
			"--tracing-config-file=" + tracingConfigFile.Name(),
			"--egress-selector-config-file=" + egressSelectorConfigFile.Name(),
		},
	)
}

func testAPIServerTracing(t *testing.T, listener net.Listener, apiserverArgs []string) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIServerTracing, true)()

	traceFound := make(chan struct{})
	defer close(traceFound)
	srv := grpc.NewServer()
	traceservice.RegisterTraceServiceServer(srv, &traceServer{
		traceFound: traceFound,
		filterFunc: containsNodeListSpan})

	go srv.Serve(listener)
	defer srv.Stop()

	// Start the API Server with our tracing configuration
	testServer := kubeapiservertesting.StartTestServerOrDie(t,
		kubeapiservertesting.NewDefaultTestServerOptions(),
		apiserverArgs,
		framework.SharedEtcd(),
	)
	defer testServer.TearDownFn()
	clientConfig := testServer.ClientConfig

	// Create a client that creates sampled traces.
	tp := trace.TracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))
	clientConfig.Wrap(tracing.WrapperFor(tp))
	clientSet, err := client.NewForConfig(clientConfig)
	if err != nil {
		t.Fatal(err)
	}

	// Make a request with the instrumented client
	_, err = clientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Wait for a span to be recorded from our request
	select {
	case <-traceFound:
		return
	case <-time.After(30 * time.Second):
		t.Fatal("Timed out waiting for trace")
	}
}

func containsNodeListSpan(req *traceservice.ExportTraceServiceRequest) bool {
	for _, resourceSpans := range req.GetResourceSpans() {
		for _, instrumentationSpans := range resourceSpans.GetScopeSpans() {
			for _, span := range instrumentationSpans.GetSpans() {
				if span.Name != "HTTP GET" {
					continue
				}
				for _, attr := range span.GetAttributes() {
					if attr.GetKey() == "http.url" {
						value := attr.GetValue().GetStringValue()
						if strings.HasSuffix(value, "/api/v1/nodes") {
							// We found our request!
							return true
						}
					}
				}
			}
		}
	}
	return false
}

// traceServer implements TracesServiceServer
type traceServer struct {
	traceFound chan struct{}
	filterFunc func(req *traceservice.ExportTraceServiceRequest) bool
	traceservice.UnimplementedTraceServiceServer
}

func (t *traceServer) Export(ctx context.Context, req *traceservice.ExportTraceServiceRequest) (*traceservice.ExportTraceServiceResponse, error) {
	var emptyValue = traceservice.ExportTraceServiceResponse{}
	if t.filterFunc(req) {
		t.traceFound <- struct{}{}
	}
	return &emptyValue, nil
}
