/*
Copyright 2016 The Kubernetes Authors.

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

package healthcheck

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-base/metrics/testutil"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/apimachinery/pkg/util/sets"

	basemetrics "k8s.io/component-base/metrics"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	testingclock "k8s.io/utils/clock/testing"
)

type fakeListener struct {
	openPorts sets.Set[string]
}

func newFakeListener() *fakeListener {
	return &fakeListener{
		openPorts: sets.Set[string]{},
	}
}

func (fake *fakeListener) hasPort(addr string) bool {
	return fake.openPorts.Has(addr)
}

func (fake *fakeListener) Listen(addr string) (net.Listener, error) {
	fake.openPorts.Insert(addr)
	return &fakeNetListener{
		parent: fake,
		addr:   addr,
	}, nil
}

type fakeNetListener struct {
	parent *fakeListener
	addr   string
}

type fakeAddr struct {
}

func (fa fakeAddr) Network() string {
	return "tcp"
}
func (fa fakeAddr) String() string {
	return "<test>"
}
func (fake *fakeNetListener) Accept() (net.Conn, error) {
	// Not implemented
	return nil, nil
}

func (fake *fakeNetListener) Close() error {
	fake.parent.openPorts.Delete(fake.addr)
	return nil
}

func (fake *fakeNetListener) Addr() net.Addr {
	// Not implemented
	return fakeAddr{}
}

type fakeHTTPServerFactory struct{}

func newFakeHTTPServerFactory() *fakeHTTPServerFactory {
	return &fakeHTTPServerFactory{}
}

func (fake *fakeHTTPServerFactory) New(addr string, handler http.Handler) httpServer {
	return &fakeHTTPServer{
		addr:    addr,
		handler: handler,
	}
}

type fakeHTTPServer struct {
	addr    string
	handler http.Handler
}

func (fake *fakeHTTPServer) Serve(listener net.Listener) error {
	return nil // Cause the goroutine to return
}

func (fake *fakeHTTPServer) Close() error {
	return nil
}

func mknsn(ns, name string) types.NamespacedName {
	return types.NamespacedName{
		Namespace: ns,
		Name:      name,
	}
}

type hcPayload struct {
	Service struct {
		Namespace string
		Name      string
	}
	LocalEndpoints      int
	ServiceProxyHealthy bool
}

type healthzPayload struct {
	LastUpdated string
	CurrentTime string
	NodeHealthy bool
}

type fakeProxierHealthChecker struct {
	healthy bool
}

func (fake fakeProxierHealthChecker) IsHealthy() bool {
	return fake.healthy
}

func TestServer(t *testing.T) {
	listener := newFakeListener()
	httpFactory := newFakeHTTPServerFactory()
	nodePortAddresses := proxyutil.NewNodePortAddresses(v1.IPv4Protocol, []string{})
	proxyChecker := &fakeProxierHealthChecker{true}

	hcsi := newServiceHealthServer("hostname", nil, listener, httpFactory, nodePortAddresses, proxyChecker)
	hcs := hcsi.(*server)
	if len(hcs.services) != 0 {
		t.Errorf("expected 0 services, got %d", len(hcs.services))
	}

	// sync nothing
	hcs.SyncServices(nil)
	if len(hcs.services) != 0 {
		t.Errorf("expected 0 services, got %d", len(hcs.services))
	}
	hcs.SyncEndpoints(nil)
	if len(hcs.services) != 0 {
		t.Errorf("expected 0 services, got %d", len(hcs.services))
	}

	// sync unknown endpoints, should be dropped
	hcs.SyncEndpoints(map[types.NamespacedName]int{mknsn("a", "b"): 93})
	if len(hcs.services) != 0 {
		t.Errorf("expected 0 services, got %d", len(hcs.services))
	}

	// sync a real service
	nsn := mknsn("a", "b")
	hcs.SyncServices(map[types.NamespacedName]uint16{nsn: 9376})
	if len(hcs.services) != 1 {
		t.Errorf("expected 1 service, got %d", len(hcs.services))
	}
	if hcs.services[nsn].endpoints != 0 {
		t.Errorf("expected 0 endpoints, got %d", hcs.services[nsn].endpoints)
	}
	if len(listener.openPorts) != 1 {
		t.Errorf("expected 1 open port, got %d\n%s", len(listener.openPorts), dump.Pretty(listener.openPorts))
	}
	if !listener.hasPort("0.0.0.0:9376") {
		t.Errorf("expected port :9376 to be open\n%s", dump.Pretty(listener.openPorts))
	}
	// test the handler
	testHandler(hcs, nsn, http.StatusServiceUnavailable, 0, t)

	// sync an endpoint
	hcs.SyncEndpoints(map[types.NamespacedName]int{nsn: 18})
	if len(hcs.services) != 1 {
		t.Errorf("expected 1 service, got %d", len(hcs.services))
	}
	if hcs.services[nsn].endpoints != 18 {
		t.Errorf("expected 18 endpoints, got %d", hcs.services[nsn].endpoints)
	}
	// test the handler
	testHandler(hcs, nsn, http.StatusOK, 18, t)

	// sync zero endpoints
	hcs.SyncEndpoints(map[types.NamespacedName]int{nsn: 0})
	if len(hcs.services) != 1 {
		t.Errorf("expected 1 service, got %d", len(hcs.services))
	}
	if hcs.services[nsn].endpoints != 0 {
		t.Errorf("expected 0 endpoints, got %d", hcs.services[nsn].endpoints)
	}
	// test the handler
	testHandler(hcs, nsn, http.StatusServiceUnavailable, 0, t)

	// put the endpoint back
	hcs.SyncEndpoints(map[types.NamespacedName]int{nsn: 11})
	if len(hcs.services) != 1 {
		t.Errorf("expected 1 service, got %d", len(hcs.services))
	}
	if hcs.services[nsn].endpoints != 11 {
		t.Errorf("expected 18 endpoints, got %d", hcs.services[nsn].endpoints)
	}
	// sync nil endpoints
	hcs.SyncEndpoints(nil)
	if len(hcs.services) != 1 {
		t.Errorf("expected 1 service, got %d", len(hcs.services))
	}
	if hcs.services[nsn].endpoints != 0 {
		t.Errorf("expected 0 endpoints, got %d", hcs.services[nsn].endpoints)
	}
	// test the handler
	testHandler(hcs, nsn, http.StatusServiceUnavailable, 0, t)

	// put the endpoint back
	hcs.SyncEndpoints(map[types.NamespacedName]int{nsn: 18})
	if len(hcs.services) != 1 {
		t.Errorf("expected 1 service, got %d", len(hcs.services))
	}
	if hcs.services[nsn].endpoints != 18 {
		t.Errorf("expected 18 endpoints, got %d", hcs.services[nsn].endpoints)
	}
	// delete the service
	hcs.SyncServices(nil)
	if len(hcs.services) != 0 {
		t.Errorf("expected 0 services, got %d", len(hcs.services))
	}

	// sync multiple services
	nsn1 := mknsn("a", "b")
	nsn2 := mknsn("c", "d")
	nsn3 := mknsn("e", "f")
	nsn4 := mknsn("g", "h")
	hcs.SyncServices(map[types.NamespacedName]uint16{
		nsn1: 9376,
		nsn2: 12909,
		nsn3: 11113,
	})
	if len(hcs.services) != 3 {
		t.Errorf("expected 3 service, got %d", len(hcs.services))
	}
	if hcs.services[nsn1].endpoints != 0 {
		t.Errorf("expected 0 endpoints, got %d", hcs.services[nsn1].endpoints)
	}
	if hcs.services[nsn2].endpoints != 0 {
		t.Errorf("expected 0 endpoints, got %d", hcs.services[nsn2].endpoints)
	}
	if hcs.services[nsn3].endpoints != 0 {
		t.Errorf("expected 0 endpoints, got %d", hcs.services[nsn3].endpoints)
	}
	if len(listener.openPorts) != 3 {
		t.Errorf("expected 3 open ports, got %d\n%s", len(listener.openPorts), dump.Pretty(listener.openPorts))
	}
	// test the handlers
	testHandler(hcs, nsn1, http.StatusServiceUnavailable, 0, t)
	testHandler(hcs, nsn2, http.StatusServiceUnavailable, 0, t)
	testHandler(hcs, nsn3, http.StatusServiceUnavailable, 0, t)

	// sync endpoints
	hcs.SyncEndpoints(map[types.NamespacedName]int{
		nsn1: 9,
		nsn2: 3,
		nsn3: 7,
	})
	if len(hcs.services) != 3 {
		t.Errorf("expected 3 services, got %d", len(hcs.services))
	}
	if hcs.services[nsn1].endpoints != 9 {
		t.Errorf("expected 9 endpoints, got %d", hcs.services[nsn1].endpoints)
	}
	if hcs.services[nsn2].endpoints != 3 {
		t.Errorf("expected 3 endpoints, got %d", hcs.services[nsn2].endpoints)
	}
	if hcs.services[nsn3].endpoints != 7 {
		t.Errorf("expected 7 endpoints, got %d", hcs.services[nsn3].endpoints)
	}
	// test the handlers
	testHandler(hcs, nsn1, http.StatusOK, 9, t)
	testHandler(hcs, nsn2, http.StatusOK, 3, t)
	testHandler(hcs, nsn3, http.StatusOK, 7, t)

	// sync new services
	hcs.SyncServices(map[types.NamespacedName]uint16{
		//nsn1: 9376, // remove it
		nsn2: 12909, // leave it
		nsn3: 11114, // change it
		nsn4: 11878, // add it
	})
	if len(hcs.services) != 3 {
		t.Errorf("expected 3 service, got %d", len(hcs.services))
	}
	if hcs.services[nsn2].endpoints != 3 {
		t.Errorf("expected 3 endpoints, got %d", hcs.services[nsn2].endpoints)
	}
	if hcs.services[nsn3].endpoints != 0 {
		t.Errorf("expected 0 endpoints, got %d", hcs.services[nsn3].endpoints)
	}
	if hcs.services[nsn4].endpoints != 0 {
		t.Errorf("expected 0 endpoints, got %d", hcs.services[nsn4].endpoints)
	}
	// test the handlers
	testHandler(hcs, nsn2, http.StatusOK, 3, t)
	testHandler(hcs, nsn3, http.StatusServiceUnavailable, 0, t)
	testHandler(hcs, nsn4, http.StatusServiceUnavailable, 0, t)

	// sync endpoints
	hcs.SyncEndpoints(map[types.NamespacedName]int{
		nsn1: 9,
		nsn2: 3,
		nsn3: 7,
		nsn4: 6,
	})
	if len(hcs.services) != 3 {
		t.Errorf("expected 3 services, got %d", len(hcs.services))
	}
	if hcs.services[nsn2].endpoints != 3 {
		t.Errorf("expected 3 endpoints, got %d", hcs.services[nsn2].endpoints)
	}
	if hcs.services[nsn3].endpoints != 7 {
		t.Errorf("expected 7 endpoints, got %d", hcs.services[nsn3].endpoints)
	}
	if hcs.services[nsn4].endpoints != 6 {
		t.Errorf("expected 6 endpoints, got %d", hcs.services[nsn4].endpoints)
	}
	// test the handlers
	testHandler(hcs, nsn2, http.StatusOK, 3, t)
	testHandler(hcs, nsn3, http.StatusOK, 7, t)
	testHandler(hcs, nsn4, http.StatusOK, 6, t)

	// sync endpoints, missing nsn2
	hcs.SyncEndpoints(map[types.NamespacedName]int{
		nsn3: 7,
		nsn4: 6,
	})
	if len(hcs.services) != 3 {
		t.Errorf("expected 3 services, got %d", len(hcs.services))
	}
	if hcs.services[nsn2].endpoints != 0 {
		t.Errorf("expected 0 endpoints, got %d", hcs.services[nsn2].endpoints)
	}
	if hcs.services[nsn3].endpoints != 7 {
		t.Errorf("expected 7 endpoints, got %d", hcs.services[nsn3].endpoints)
	}
	if hcs.services[nsn4].endpoints != 6 {
		t.Errorf("expected 6 endpoints, got %d", hcs.services[nsn4].endpoints)
	}
	// test the handlers
	testHandler(hcs, nsn2, http.StatusServiceUnavailable, 0, t)
	testHandler(hcs, nsn3, http.StatusOK, 7, t)
	testHandler(hcs, nsn4, http.StatusOK, 6, t)

	// fake a temporary unhealthy proxy
	proxyChecker.healthy = false
	testHandlerWithHealth(hcs, nsn2, http.StatusServiceUnavailable, 0, false, t)
	testHandlerWithHealth(hcs, nsn3, http.StatusServiceUnavailable, 7, false, t)
	testHandlerWithHealth(hcs, nsn4, http.StatusServiceUnavailable, 6, false, t)

	// fake a healthy proxy
	proxyChecker.healthy = true
	testHandlerWithHealth(hcs, nsn2, http.StatusServiceUnavailable, 0, true, t)
	testHandlerWithHealth(hcs, nsn3, http.StatusOK, 7, true, t)
	testHandlerWithHealth(hcs, nsn4, http.StatusOK, 6, true, t)
}

func testHandler(hcs *server, nsn types.NamespacedName, status int, endpoints int, t *testing.T) {
	tHandler(hcs, nsn, status, endpoints, true, t)
}

func testHandlerWithHealth(hcs *server, nsn types.NamespacedName, status int, endpoints int, kubeProxyHealthy bool, t *testing.T) {
	tHandler(hcs, nsn, status, endpoints, kubeProxyHealthy, t)
}

func tHandler(hcs *server, nsn types.NamespacedName, status int, endpoints int, kubeProxyHealthy bool, t *testing.T) {
	instance := hcs.services[nsn]
	for _, h := range instance.httpServers {
		handler := h.(*fakeHTTPServer).handler

		req, err := http.NewRequest("GET", "/healthz", nil)
		if err != nil {
			t.Fatal(err)
		}
		resp := httptest.NewRecorder()

		handler.ServeHTTP(resp, req)

		if resp.Code != status {
			t.Errorf("expected status code %v, got %v", status, resp.Code)
		}
		var payload hcPayload
		if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
			t.Fatal(err)
		}
		if payload.Service.Name != nsn.Name || payload.Service.Namespace != nsn.Namespace {
			t.Errorf("expected payload name %q, got %v", nsn.String(), payload.Service)
		}
		if payload.LocalEndpoints != endpoints {
			t.Errorf("expected %d endpoints, got %d", endpoints, payload.LocalEndpoints)
		}
		if payload.ServiceProxyHealthy != kubeProxyHealthy {
			t.Errorf("expected %v kubeProxyHealthy, got %v", kubeProxyHealthy, payload.ServiceProxyHealthy)
		}
		if !cmp.Equal(resp.Header()["Content-Type"], []string{"application/json"}) {
			t.Errorf("expected 'Content-Type: application/json' respose header, got: %v", resp.Header()["Content-Type"])
		}
		if !cmp.Equal(resp.Header()["X-Content-Type-Options"], []string{"nosniff"}) {
			t.Errorf("expected 'X-Content-Type-Options: nosniff' respose header, got: %v", resp.Header()["X-Content-Type-Options"])
		}
		if !cmp.Equal(resp.Header()["X-Load-Balancing-Endpoint-Weight"], []string{strconv.Itoa(endpoints)}) {
			t.Errorf("expected 'X-Load-Balancing-Endpoint-Weight: %d' respose header, got: %v", endpoints, resp.Header()["X-Load-Balancing-Endpoint-Weight"])
		}
	}
}

type nodeTweak func(n *v1.Node)

func makeNode(tweaks ...nodeTweak) *v1.Node {
	n := &v1.Node{}
	for _, tw := range tweaks {
		tw(n)
	}
	return n
}

func tweakDeleted() nodeTweak {
	return func(n *v1.Node) {
		n.DeletionTimestamp = &metav1.Time{
			Time: time.Now(),
		}
	}
}

func tweakTainted(key string) nodeTweak {
	return func(n *v1.Node) {
		n.Spec.Taints = append(n.Spec.Taints, v1.Taint{Key: key})
	}
}

type serverTest struct {
	server      httpServer
	url         url
	tracking200 int
	tracking503 int
}

func TestHealthzServer(t *testing.T) {
	metrics.RegisterMetrics("")
	listener := newFakeListener()
	httpFactory := newFakeHTTPServerFactory()
	fakeClock := testingclock.NewFakeClock(time.Now())

	hs := newProxierHealthServer(listener, httpFactory, fakeClock, "127.0.0.1:10256", 10*time.Second)
	server := hs.httpFactory.New(hs.addr, healthzHandler{hs: hs})

	hsTest := &serverTest{
		server:      server,
		url:         healthzURL,
		tracking200: 0,
		tracking503: 0,
	}

	testProxierHealthUpdater(hs, hsTest, fakeClock, t)

	// Should return 200 "OK" if we've synced a node, tainted in any other way
	hs.SyncNode(makeNode(tweakTainted("other")))
	testHTTPHandler(hsTest, http.StatusOK, t)

	// Should return 503 "ServiceUnavailable" if we've synced a ToBeDeletedTaint node
	hs.SyncNode(makeNode(tweakTainted(ToBeDeletedTaint)))
	testHTTPHandler(hsTest, http.StatusServiceUnavailable, t)

	// Should return 200 "OK" if we've synced a node, tainted in any other way
	hs.SyncNode(makeNode(tweakTainted("other")))
	testHTTPHandler(hsTest, http.StatusOK, t)

	// Should return 503 "ServiceUnavailable" if we've synced a deleted node
	hs.SyncNode(makeNode(tweakDeleted()))
	testHTTPHandler(hsTest, http.StatusServiceUnavailable, t)
}

func TestLivezServer(t *testing.T) {
	metrics.RegisterMetrics("")
	listener := newFakeListener()
	httpFactory := newFakeHTTPServerFactory()
	fakeClock := testingclock.NewFakeClock(time.Now())

	hs := newProxierHealthServer(listener, httpFactory, fakeClock, "127.0.0.1:10256", 10*time.Second)
	server := hs.httpFactory.New(hs.addr, livezHandler{hs: hs})

	hsTest := &serverTest{
		server:      server,
		url:         livezURL,
		tracking200: 0,
		tracking503: 0,
	}

	testProxierHealthUpdater(hs, hsTest, fakeClock, t)

	// Should return 200 "OK" irrespective of node syncs
	hs.SyncNode(makeNode(tweakTainted("other")))
	testHTTPHandler(hsTest, http.StatusOK, t)

	// Should return 200 "OK" irrespective of node syncs
	hs.SyncNode(makeNode(tweakTainted(ToBeDeletedTaint)))
	testHTTPHandler(hsTest, http.StatusOK, t)

	// Should return 200 "OK" irrespective of node syncs
	hs.SyncNode(makeNode(tweakTainted("other")))
	testHTTPHandler(hsTest, http.StatusOK, t)

	// Should return 200 "OK" irrespective of node syncs
	hs.SyncNode(makeNode(tweakDeleted()))
	testHTTPHandler(hsTest, http.StatusOK, t)
}

type url string

var (
	healthzURL url = "/healthz"
	livezURL   url = "/livez"
)

func testProxierHealthUpdater(hs *ProxierHealthServer, hsTest *serverTest, fakeClock *testingclock.FakeClock, t *testing.T) {
	// Should return 200 "OK" by default.
	testHTTPHandler(hsTest, http.StatusOK, t)

	// Should return 200 "OK" after first update for both IPv4 and IPv6 proxiers.
	hs.Updated(v1.IPv4Protocol)
	hs.Updated(v1.IPv6Protocol)
	testHTTPHandler(hsTest, http.StatusOK, t)

	// Should continue to return 200 "OK" as long as no further updates are queued for any proxier.
	fakeClock.Step(25 * time.Second)
	testHTTPHandler(hsTest, http.StatusOK, t)

	// Should return 503 "ServiceUnavailable" if IPv4 proxier exceed max update-processing time.
	hs.QueuedUpdate(v1.IPv4Protocol)
	fakeClock.Step(25 * time.Second)
	testHTTPHandler(hsTest, http.StatusServiceUnavailable, t)

	// Should return 200 "OK" after processing update for both IPv4 and IPv6 proxiers.
	hs.Updated(v1.IPv4Protocol)
	hs.Updated(v1.IPv6Protocol)
	fakeClock.Step(5 * time.Second)
	testHTTPHandler(hsTest, http.StatusOK, t)

	// Should return 503 "ServiceUnavailable" if IPv6 proxier exceed max update-processing time.
	hs.QueuedUpdate(v1.IPv6Protocol)
	fakeClock.Step(25 * time.Second)
	testHTTPHandler(hsTest, http.StatusServiceUnavailable, t)

	// Should return 200 "OK" after processing update for both IPv4 and IPv6 proxiers.
	hs.Updated(v1.IPv4Protocol)
	hs.Updated(v1.IPv6Protocol)
	fakeClock.Step(5 * time.Second)
	testHTTPHandler(hsTest, http.StatusOK, t)

	// Should return 503 "ServiceUnavailable" if both IPv4 and IPv6 proxiers exceed max update-processing time.
	hs.QueuedUpdate(v1.IPv4Protocol)
	hs.QueuedUpdate(v1.IPv6Protocol)
	fakeClock.Step(25 * time.Second)
	testHTTPHandler(hsTest, http.StatusServiceUnavailable, t)

	// Should return 200 "OK" after processing update for both IPv4 and IPv6 proxiers.
	hs.Updated(v1.IPv4Protocol)
	hs.Updated(v1.IPv6Protocol)
	fakeClock.Step(5 * time.Second)
	testHTTPHandler(hsTest, http.StatusOK, t)

	// If IPv6 proxier is late for an update but IPv4 proxier is not then updating IPv4 proxier should have no effect.
	hs.QueuedUpdate(v1.IPv6Protocol)
	fakeClock.Step(25 * time.Second)
	testHTTPHandler(hsTest, http.StatusServiceUnavailable, t)

	hs.Updated(v1.IPv4Protocol)
	testHTTPHandler(hsTest, http.StatusServiceUnavailable, t)

	hs.Updated(v1.IPv6Protocol)
	testHTTPHandler(hsTest, http.StatusOK, t)

	// If both IPv4 and IPv6 proxiers are late for an update, we shouldn't report 200 "OK" until after both of them update.
	hs.QueuedUpdate(v1.IPv4Protocol)
	hs.QueuedUpdate(v1.IPv6Protocol)
	fakeClock.Step(25 * time.Second)
	testHTTPHandler(hsTest, http.StatusServiceUnavailable, t)

	hs.Updated(v1.IPv4Protocol)
	testHTTPHandler(hsTest, http.StatusServiceUnavailable, t)

	hs.Updated(v1.IPv6Protocol)
	testHTTPHandler(hsTest, http.StatusOK, t)
}

func testHTTPHandler(hsTest *serverTest, status int, t *testing.T) {
	handler := hsTest.server.(*fakeHTTPServer).handler
	req, err := http.NewRequest("GET", string(hsTest.url), nil)
	if err != nil {
		t.Fatal(err)
	}
	resp := httptest.NewRecorder()

	handler.ServeHTTP(resp, req)

	if resp.Code != status {
		t.Errorf("expected status code %v, got %v", status, resp.Code)
	}
	var payload healthzPayload
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}

	if status == http.StatusOK {
		hsTest.tracking200++
	}
	if status == http.StatusServiceUnavailable {
		hsTest.tracking503++
	}
	if hsTest.url == healthzURL {
		testMetricEquals(metrics.ProxyHealthzTotal.WithLabelValues("200"), float64(hsTest.tracking200), t)
		testMetricEquals(metrics.ProxyHealthzTotal.WithLabelValues("503"), float64(hsTest.tracking503), t)
	}
	if hsTest.url == livezURL {
		testMetricEquals(metrics.ProxyLivezTotal.WithLabelValues("200"), float64(hsTest.tracking200), t)
		testMetricEquals(metrics.ProxyLivezTotal.WithLabelValues("503"), float64(hsTest.tracking503), t)
	}
}

func testMetricEquals(metric basemetrics.CounterMetric, expected float64, t *testing.T) {
	t.Helper()
	val, err := testutil.GetCounterMetricValue(metric)
	if err != nil {
		t.Errorf("unable to retrieve value for metric, err: %v", err)
	}
	if val != expected {
		t.Errorf("expected: %v, found: %v", expected, val)
	}
}

func TestServerWithSelectiveListeningAddress(t *testing.T) {
	listener := newFakeListener()
	httpFactory := newFakeHTTPServerFactory()
	proxyChecker := &fakeProxierHealthChecker{true}

	// limiting addresses to loop back. We don't want any cleverness here around getting IP for
	// machine nor testing ipv6 || ipv4. using loop back guarantees the test will work on any machine
	nodePortAddresses := proxyutil.NewNodePortAddresses(v1.IPv4Protocol, []string{"127.0.0.0/8"})

	hcsi := newServiceHealthServer("hostname", nil, listener, httpFactory, nodePortAddresses, proxyChecker)
	hcs := hcsi.(*server)
	if len(hcs.services) != 0 {
		t.Errorf("expected 0 services, got %d", len(hcs.services))
	}

	// sync nothing
	hcs.SyncServices(nil)
	if len(hcs.services) != 0 {
		t.Errorf("expected 0 services, got %d", len(hcs.services))
	}
	hcs.SyncEndpoints(nil)
	if len(hcs.services) != 0 {
		t.Errorf("expected 0 services, got %d", len(hcs.services))
	}

	// sync unknown endpoints, should be dropped
	hcs.SyncEndpoints(map[types.NamespacedName]int{mknsn("a", "b"): 93})
	if len(hcs.services) != 0 {
		t.Errorf("expected 0 services, got %d", len(hcs.services))
	}

	// sync a real service
	nsn := mknsn("a", "b")
	hcs.SyncServices(map[types.NamespacedName]uint16{nsn: 9376})
	if len(hcs.services) != 1 {
		t.Errorf("expected 1 service, got %d", len(hcs.services))
	}
	if hcs.services[nsn].endpoints != 0 {
		t.Errorf("expected 0 endpoints, got %d", hcs.services[nsn].endpoints)
	}
	if len(listener.openPorts) != 1 {
		t.Errorf("expected 1 open port, got %d\n%s", len(listener.openPorts), dump.Pretty(listener.openPorts))
	}
	if !listener.hasPort("127.0.0.1:9376") {
		t.Errorf("expected port :9376 to be open\n%s", dump.Pretty(listener.openPorts))
	}
	// test the handler
	testHandler(hcs, nsn, http.StatusServiceUnavailable, 0, t)
}
