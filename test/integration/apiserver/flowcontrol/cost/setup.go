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

package cost

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	genericapiserver "k8s.io/apiserver/pkg/server"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/test/integration/framework"
	netutils "k8s.io/utils/net"
)

func setupAPIServer(tb testing.TB) (clientset.Interface, http.Handler) {
	tb.Helper()

	var handler http.Handler

	client, _, tearDownFn := framework.StartTestServer(tb.Context(), tb, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.AdvertiseAddress = netutils.ParseIPSloppy("10.0.1.1")
			opts.Authorization.Modes = []string{"AlwaysAllow"}
		},
		ModifyServerConfig: func(cfg *controlplane.Config) {
			generic := cfg.ControlPlane.Generic
			generic.LoopbackClientConfig.QPS = -1
			bearer := "Bearer " + generic.LoopbackClientConfig.BearerToken

			prevBuildChain := generic.BuildHandlerChainFunc
			generic.BuildHandlerChainFunc = func(apiHandler http.Handler, c *genericapiserver.Config) http.Handler {
				chain := prevBuildChain(apiHandler, c)
				handler = http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
					req.Header.Set("Authorization", bearer)
					chain.ServeHTTP(w, req)
				})
				return chain
			}
		},
	})
	tb.Cleanup(tearDownFn)
	return client, handler
}

// responseSink is a non-buffering http.ResponseWriter so measurement only
// captures kube-apiserver handler allocations instead of httptest.ResponseRecorder's
// growing bytes.Buffer.
type responseSink struct {
	header       http.Header
	code         int
	bytesWritten int
}

func newResponseSink() *responseSink {
	return &responseSink{header: make(http.Header), code: http.StatusOK}
}

func (s *responseSink) Header() http.Header { return s.header }

func (s *responseSink) WriteHeader(code int) { s.code = code }

func (s *responseSink) Write(b []byte) (int, error) {
	s.bytesWritten += len(b)
	return len(b), nil
}

// Flush and CloseNotify satisfy responsewriter.CloseNotifierFlusher so
// apiserver's responsewriter.WrapForHTTP1Or2 preserves http.Flusher for WATCH.
func (s *responseSink) Flush() {}

func (s *responseSink) CloseNotify() <-chan bool { return nil }

func runHTTP(tb testing.TB, server http.Handler, method, path, contentType, accept string, body []byte) int {
	tb.Helper()
	var bodyReader *bytes.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	} else {
		bodyReader = bytes.NewReader(nil)
	}
	req := httptest.NewRequest(method, path, bodyReader).WithContext(context.Background())
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}

	sink := newResponseSink()
	server.ServeHTTP(sink, req)
	if sink.code < 200 || sink.code >= 300 {
		tb.Fatalf("%s %s (%s) returned HTTP %d", method, path, accept, sink.code)
	}
	return sink.bytesWritten
}
