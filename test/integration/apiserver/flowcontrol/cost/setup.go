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
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/ktesting"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controlplane"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
	netutils "k8s.io/utils/net"
)

func setupAPIServer(tCtx ktesting.TContext) (ktesting.TContext, http.Handler, *informerWaiter) {
	tCtx.Helper()

	var (
		handler     http.Handler
		podInformer cache.SharedIndexInformer
	)

	client, _, tearDownFn := framework.StartTestServer(tCtx, tCtx, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.AdvertiseAddress = netutils.ParseIPSloppy("10.0.1.1")
			opts.Authorization.Modes = []string{"AlwaysAllow"}
			// Prevent periodic 5m auto-compaction from expiring seeded ResourceVersions
			// (LIST rv=Exact) or injecting mid-benchmark allocation/latency spikes.
			opts.Etcd.StorageConfig.CompactionInterval = 0
		},
		ModifyServerConfig: func(cfg *controlplane.Config) {
			generic := cfg.ControlPlane.Generic
			generic.LoopbackClientConfig.QPS = -1
			bearer := "Bearer " + generic.LoopbackClientConfig.BearerToken
			podInformer = cfg.ControlPlane.VersionedInformers.Core().V1().Pods().Informer()

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
	tCtx.Cleanup(tearDownFn)
	if !cache.WaitForCacheSync(tCtx.Done(), podInformer.HasSynced) {
		tCtx.Fatalf("timed out waiting for pod informer cache sync")
	}
	informer := newInformerWaiter(tCtx, podInformer)
	tCtx = tCtx.WithClients(nil, nil, client, nil)
	return tCtx, handler, informer
}

type informerWaiter struct {
	informer cache.SharedIndexInformer
	events   atomic.Int64
	notify   chan struct{}
}

func newInformerWaiter(tb testutils.TB, informer cache.SharedIndexInformer) *informerWaiter {
	tb.Helper()
	t := &informerWaiter{
		informer: informer,
		notify:   make(chan struct{}, 1),
	}
	_, err := informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    t.record,
		UpdateFunc: func(_, newObj any) { t.record(newObj) },
		DeleteFunc: t.record,
	})
	if err != nil {
		tb.Fatalf("failed to register pod informer event handler: %v", err)
	}
	return t
}

func (t *informerWaiter) record(obj any) {
	if d, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		obj = d.Obj
	}
	if _, ok := obj.(*v1.Pod); ok {
		t.events.Add(1)
		select {
		case t.notify <- struct{}{}:
		default:
		}
	}
}

func (t *informerWaiter) resetEvents() {
	t.events.Store(0)
	select {
	case <-t.notify:
	default:
	}
}

func (t *informerWaiter) wait(ctx context.Context, tb testutils.TB, target int64) {
	for t.events.Load() < target {
		select {
		case <-t.notify:
		case <-ctx.Done():
			tb.Fatalf("context canceled waiting for informer events (%d/%d): %v", t.events.Load(), target, ctx.Err())
		}
	}
}

// responseSink is a non-buffering http.ResponseWriter so measurement only
// captures kube-apiserver handler allocations instead of httptest.ResponseRecorder's
// growing bytes.Buffer.
type responseSink struct {
	header            http.Header
	code              int
	bytesWritten      int
	writes            int
	cancelAfterWrites int
	cancelRequest     context.CancelFunc
	errBody           []byte
}

func newResponseSink(cancelAfterWrites int, cancelRequest context.CancelFunc) *responseSink {
	return &responseSink{
		header:            make(http.Header),
		code:              http.StatusOK,
		cancelAfterWrites: cancelAfterWrites,
		cancelRequest:     cancelRequest,
	}
}

func (s *responseSink) Header() http.Header { return s.header }

func (s *responseSink) WriteHeader(code int) { s.code = code }

func (s *responseSink) Write(b []byte) (int, error) {
	s.bytesWritten += len(b)
	s.writes++
	if s.code >= 300 {
		s.errBody = append(s.errBody, b...)
	}
	return len(b), nil
}

// Flush and CloseNotify satisfy responsewriter.CloseNotifierFlusher so
// apiserver's responsewriter.WrapForHTTP1Or2 preserves http.Flusher for WATCH.
func (s *responseSink) Flush() {
	if s.code >= 300 || s.cancelRequest == nil {
		return
	}
	if s.writes >= s.cancelAfterWrites {
		s.cancelRequest()
		s.cancelRequest = nil
	}
}

func (s *responseSink) CloseNotify() <-chan bool { return nil }

func runHTTP(ctx context.Context, tb testutils.TB, server http.Handler, method, path, contentType, accept string, body []byte, sink *responseSink) int {
	tb.Helper()
	req := httptest.NewRequest(method, path, bytes.NewReader(body)).WithContext(ctx)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}

	server.ServeHTTP(sink, req)
	if sink.code < 200 || sink.code >= 300 {
		tb.Fatalf("%s %s (%s) returned HTTP %d: %s", method, path, accept, sink.code, sink.errBody)
	}
	if sink.cancelAfterWrites > 0 && sink.writes != sink.cancelAfterWrites {
		tb.Fatalf("%s %s (%s) delivered %d writes, want %d", method, path, accept, sink.writes, sink.cancelAfterWrites)
	}
	return sink.bytesWritten
}
