/*
Copyright 2022 The Kubernetes Authors.

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

package client

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/metrics"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

type resultMetricCall struct {
	code   string
	method string
	host   string
}

type resultMetrics struct {
	mu    sync.Mutex
	calls []resultMetricCall
	codes map[string]int
}

func (r *resultMetrics) Increment(ctx context.Context, code string, method string, host string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, resultMetricCall{code: code, method: method, host: host})
	if r.codes == nil {
		r.codes = map[string]int{}
	}

	if _, ok := r.codes[code]; !ok {
		r.codes[code] = 1
	} else {
		r.codes[code]++
	}

}

func (r *resultMetrics) reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = []resultMetricCall{}
	r.codes = map[string]int{}
}

func Test_Client_Get_Network_Errors(t *testing.T) {
	actualMetrics := &resultMetrics{}
	metrics.RequestResult = actualMetrics

	// run an apiserver
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins", "ServiceAccount"}, framework.SharedEtcd())
	defer result.TearDownFn()

	apiURL, err := url.Parse(result.ClientConfig.Host)
	if err != nil {
		t.Errorf("unexpected error %v", err)
	}

	tr, err := rest.TransportFor(result.ClientConfig)
	if err != nil {
		t.Errorf("unexpected error %v", err)
	}

	stopCh := make(chan struct{})
	defer close(stopCh)

	// create a layer 7 (http) loadbalancer and a clientset using it
	httpLB := newHTTPLB(apiURL, tr)
	httpLB.serve(stopCh)

	cfg := rest.CopyConfig(result.ClientConfig)
	cfg.Host = httpLB.url()
	cfg.TLSClientConfig = rest.TLSClientConfig{
		Insecure: true,
	}
	clientHTTPLB := clientset.NewForConfigOrDie(cfg)

	// create a layer 4 (tcp) loadbalancer
	tcpLB := newLB(t, apiURL.Host)
	tcpLB.serve(stopCh)
	cfg = rest.CopyConfig(result.ClientConfig)
	cfg.Host = tcpLB.url()
	clientTCPLB := clientset.NewForConfigOrDie(cfg)

	tests := []struct {
		name                 string
		lb                   loadBalancer
		client               *clientset.Clientset
		expectedConnections  int
		expectedDuration     time.Duration
		expectedError        string
		expectedMetricResult string
	}{
		{
			name:                 "retries for TCP loadbalancer",
			lb:                   tcpLB,
			client:               clientTCPLB,
			expectedConnections:  11,
			expectedDuration:     10 * time.Second,
			expectedError:        "EOF",
			expectedMetricResult: "<error>",
		},
		{
			name:                 "retries for HTTP loadbalancer",
			lb:                   httpLB,
			client:               clientHTTPLB,
			expectedConnections:  11,
			expectedDuration:     10 * time.Second,
			expectedError:        "INJECTED ERROR",
			expectedMetricResult: "500",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// check proxy works fine
			_, err = test.client.CoreV1().Services("default").Get(context.TODO(), "kubernetes", metav1.GetOptions{})
			if err != nil {
				t.Errorf("unexpected error received: %v", err)
			}

			// reset connections so we start clean
			utilnet.CloseIdleConnectionsFor(test.client.RESTClient().(*rest.RESTClient).Client.Transport)
			test.lb.closeConnections()

			// clean metrics
			test.lb.resetCounter()
			actualMetrics.reset()

			// start rejecting
			test.lb.setReject(true)
			defer test.lb.setReject(false)

			now := time.Now()
			_, err = test.client.CoreV1().Services("default").Get(context.TODO(), "kubernetes", metav1.GetOptions{})
			if err == nil {
				t.Errorf("expected error")
			}
			if err != nil && !strings.Contains(err.Error(), test.expectedError) {
				t.Errorf("received error %v expected to contain: %s", err, test.expectedError)
			}

			elapsed := time.Since(now)
			if elapsed < test.expectedDuration {
				t.Errorf("expected call blocked for ~ 10 seconds (10 retries per second), current duration: %v", elapsed)
			}

			if test.lb.getCounter() != test.expectedConnections {
				t.Errorf("expected %d connections, got %d", test.expectedConnections, test.lb.getCounter())
			}

			v, ok := actualMetrics.codes[test.expectedMetricResult]
			if !ok {
				t.Errorf("expected metrics with code: %s", test.expectedMetricResult)
			}
			if v != test.expectedConnections {
				t.Errorf("expected %d metrics with code: %s, got %d", test.expectedConnections, test.expectedMetricResult, v)
			}
		})
	}

}

type loadBalancer interface {
	serve(stopCh chan struct{})
	setReject(v bool)
	getReject() bool
	getCounter() int
	resetCounter()
	closeConnections()
	url() string
}

var _ loadBalancer = (*httpLB)(nil)
var _ loadBalancer = (*tcpLB)(nil)

type httpLB struct {
	mu       sync.Mutex
	reject   bool
	server   *httptest.Server
	requests int32
}

func newHTTPLB(serverURL *url.URL, tr http.RoundTripper) *httpLB {
	h := &httpLB{}
	proxyHandler := httputil.NewSingleHostReverseProxy(serverURL)
	proxyHandler.Transport = tr
	h.server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if h.getReject() {
			atomic.AddInt32(&h.requests, 1)
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte("INJECTED ERROR"))
			return
		}
		proxyHandler.ServeHTTP(w, r)
	}))
	h.server.EnableHTTP2 = true
	return h
}

func (h *httpLB) serve(stopCh chan struct{}) {
	h.server.StartTLS()
	go func() {
		<-stopCh
		h.server.Close()
	}()
}

func (h *httpLB) setReject(v bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.reject = v
}

func (h *httpLB) getReject() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.reject
}

func (h *httpLB) getCounter() int {
	return int(atomic.LoadInt32(&h.requests))
}

func (h *httpLB) resetCounter() {
	atomic.StoreInt32(&h.requests, 0)
}

func (h *httpLB) closeConnections() {
	h.server.CloseClientConnections()
}

func (h *httpLB) url() string {
	return h.server.URL
}

type tcpLB struct {
	t         *testing.T
	mu        sync.Mutex
	ln        net.Listener
	serverURL string
	dials     int32
	reject    bool
	connMap   sync.Map
}

func (lb *tcpLB) handleConnection(in net.Conn) {
	lb.connMap.Store(in, "")
	defer lb.connMap.Delete(in)
	out, err := net.Dial("tcp", lb.serverURL)
	if err != nil {
		lb.t.Log(err)
		return
	}
	defer out.Close()

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		io.Copy(out, in)
	}()
	go func() {
		defer wg.Done()
		io.Copy(in, out)
	}()
	wg.Wait()
}

func (lb *tcpLB) serve(stopCh chan struct{}) {
	go func() {
		<-stopCh
		lb.closeConnections()
		lb.ln.Close()
	}()
	go func() {
		for {
			conn, err := lb.ln.Accept()
			lb.t.Logf("New connection")
			select {
			case <-stopCh:
				return
			default:
			}
			if err != nil {
				lb.t.Fatalf("failed to accept: %v", err)
			}
			atomic.AddInt32(&lb.dials, 1)
			if lb.getReject() {
				lb.t.Logf("Closing connection")
				conn.Close()
			} else {
				go lb.handleConnection(conn)
			}
		}
	}()
}

func (lb *tcpLB) setReject(v bool) {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	lb.reject = v
}

func (lb *tcpLB) getReject() bool {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	return lb.reject
}

func (lb *tcpLB) closeConnections() {
	lb.connMap.Range(func(key, value interface{}) bool {
		if conn, ok := key.(net.Conn); ok {
			lb.t.Logf("Closing connection")
			conn.Close()
		}
		return true
	})
}

func (lb *tcpLB) getCounter() int {
	return int(atomic.LoadInt32(&lb.dials))
}
func (lb *tcpLB) resetCounter() {
	atomic.StoreInt32(&lb.dials, 0)
}

func (lb *tcpLB) url() string {
	return "https://" + lb.ln.Addr().String()
}

func newLB(t *testing.T, serverURL string) *tcpLB {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to bind: %v", err)
	}
	lb := tcpLB{
		serverURL: serverURL,
		ln:        ln,
		t:         t,
		connMap:   sync.Map{},
	}
	return &lb
}
