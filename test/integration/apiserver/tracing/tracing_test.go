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
	"io/ioutil"
	"net"
	"os"
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
	"k8s.io/component-base/traces"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestAPIServerTracing(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIServerTracing, true)()

	// Listen for traces from the API Server before starting it, so the
	// API Server will successfully connect right away during the test.
	listener, err := net.Listen("tcp", "localhost:")
	if err != nil {
		t.Fatal(err)
	}

	traceFound := make(chan struct{})
	defer close(traceFound)
	srv := grpc.NewServer()
	traceservice.RegisterTraceServiceServer(srv, &traceServer{
		traceFound: traceFound,
		filterFunc: containsNodeListSpan,
	})

	go srv.Serve(listener)
	defer srv.Stop()

	// Write the configuration for tracing to a file
	tracingConfigFile, err := ioutil.TempFile("", "tracing-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tracingConfigFile.Name())

	if err := ioutil.WriteFile(tracingConfigFile.Name(), []byte(fmt.Sprintf(`
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: TracingConfiguration
endpoint: %s`, listener.Addr().String())), os.FileMode(0o755)); err != nil {
		t.Fatal(err)
	}

	// Start the API Server with our tracing configuration
	stopCh := make(chan struct{})
	defer close(stopCh)
	testServer := kubeapiservertesting.StartTestServerOrDie(t,
		kubeapiservertesting.NewDefaultTestServerOptions(),
		[]string{"--tracing-config-file=" + tracingConfigFile.Name()},
		framework.SharedEtcd(),
	)
	clientConfig := testServer.ClientConfig

	// Create a client that creates sampled traces.
	tp := trace.TracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.AlwaysSample())))
	clientConfig.Wrap(traces.WrapperFor(&tp))
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
		for _, instrumentationSpans := range resourceSpans.GetInstrumentationLibrarySpans() {
			for _, span := range instrumentationSpans.GetSpans() {
				if span.Name != "KubernetesAPI" {
					continue
				}
				for _, attr := range span.GetAttributes() {
					if attr.GetKey() == "http.target" && attr.GetValue().GetStringValue() == "/api/v1/nodes" {
						// We found our request!
						return true
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
	emptyValue := traceservice.ExportTraceServiceResponse{}
	if t.filterFunc(req) {
		t.traceFound <- struct{}{}
	}
	return &emptyValue, nil
}
