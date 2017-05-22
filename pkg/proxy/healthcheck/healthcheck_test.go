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
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/davecgh/go-spew/spew"
)

type fakeListener struct {
	openPorts sets.String
}

func newFakeListener() *fakeListener {
	return &fakeListener{
		openPorts: sets.String{},
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
	return nil
}

type fakeHTTPServerFactory struct{}

func newFakeHTTPServerFactory() *fakeHTTPServerFactory {
	return &fakeHTTPServerFactory{}
}

func (fake *fakeHTTPServerFactory) New(addr string, handler http.Handler) HTTPServer {
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
	LocalEndpoints int
}

type healthzPayload struct {
	LastUpdated string
	CurrentTime string
}

func TestServer(t *testing.T) {
	listener := newFakeListener()
	httpFactory := newFakeHTTPServerFactory()

	hcsi := NewServer("hostname", nil, listener, httpFactory)
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
		t.Errorf("expected 1 open port, got %d\n%s", len(listener.openPorts), spew.Sdump(listener.openPorts))
	}
	if !listener.hasPort(":9376") {
		t.Errorf("expected port :9376 to be open\n%s", spew.Sdump(listener.openPorts))
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
		t.Errorf("expected 3 open ports, got %d\n%s", len(listener.openPorts), spew.Sdump(listener.openPorts))
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
}

func testHandler(hcs *server, nsn types.NamespacedName, status int, endpoints int, t *testing.T) {
	handler := hcs.services[nsn].server.(*fakeHTTPServer).handler
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
}

func TestHealthzServer(t *testing.T) {
	listener := newFakeListener()
	httpFactory := newFakeHTTPServerFactory()
	fakeClock := clock.NewFakeClock(time.Now())

	hs := newHealthzServer(listener, httpFactory, fakeClock, "127.0.0.1:10256", 10*time.Second)
	server := hs.httpFactory.New(hs.addr, healthzHandler{hs: hs})

	// Should return 200 "OK" by default.
	testHealthzHandler(server, http.StatusOK, t)

	// Should return 503 "ServiceUnavailable" if exceed max no respond duration.
	hs.UpdateTimestamp()
	fakeClock.Step(25 * time.Second)
	testHealthzHandler(server, http.StatusServiceUnavailable, t)

	// Should return 200 "OK" if timestamp is valid.
	hs.UpdateTimestamp()
	fakeClock.Step(5 * time.Second)
	testHealthzHandler(server, http.StatusOK, t)
}

func testHealthzHandler(server HTTPServer, status int, t *testing.T) {
	handler := server.(*fakeHTTPServer).handler
	req, err := http.NewRequest("GET", "/healthz", nil)
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
}
