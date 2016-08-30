package agent

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/serf/coordinate"
)

func TestHealthChecksInState(t *testing.T) {
	httpTest(t, func(srv *HTTPServer) {
		req, err := http.NewRequest("GET", "/v1/health/state/warning?dc=dc1", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		testutil.WaitForResult(func() (bool, error) {
			resp := httptest.NewRecorder()
			obj, err := srv.HealthChecksInState(resp, req)
			if err != nil {
				return false, err
			}
			if err := checkIndex(resp); err != nil {
				return false, err
			}

			// Should be a non-nil empty list
			nodes := obj.(structs.HealthChecks)
			if nodes == nil || len(nodes) != 0 {
				return false, fmt.Errorf("bad: %v", obj)
			}
			return true, nil
		}, func(err error) { t.Fatalf("err: %v", err) })
	})

	httpTest(t, func(srv *HTTPServer) {
		req, err := http.NewRequest("GET", "/v1/health/state/passing?dc=dc1", nil)
		if err != nil {
			t.Fatalf("err: %v", err)
		}

		testutil.WaitForResult(func() (bool, error) {
			resp := httptest.NewRecorder()
			obj, err := srv.HealthChecksInState(resp, req)
			if err != nil {
				return false, err
			}
			if err := checkIndex(resp); err != nil {
				return false, err
			}

			// Should be 1 health check for the server
			nodes := obj.(structs.HealthChecks)
			if len(nodes) != 1 {
				return false, fmt.Errorf("bad: %v", obj)
			}
			return true, nil
		}, func(err error) { t.Fatalf("err: %v", err) })
	})
}

func TestHealthChecksInState_DistanceSort(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "bar",
		Address:    "127.0.0.1",
		Check: &structs.HealthCheck{
			Node:   "bar",
			Name:   "node check",
			Status: structs.HealthCritical,
		},
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	args.Node, args.Check.Node = "foo", "foo"
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	req, err := http.NewRequest("GET", "/v1/health/state/critical?dc=dc1&near=foo", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.HealthChecksInState(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)
	nodes := obj.(structs.HealthChecks)
	if len(nodes) != 2 {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].Node != "bar" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[1].Node != "foo" {
		t.Fatalf("bad: %v", nodes)
	}

	// Send an update for the node and wait for it to get applied.
	arg := structs.CoordinateUpdateRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Coord:      coordinate.NewCoordinate(coordinate.DefaultConfig()),
	}
	if err := srv.agent.RPC("Coordinate.Update", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	time.Sleep(200 * time.Millisecond)

	// Query again and now foo should have moved to the front of the line.
	resp = httptest.NewRecorder()
	obj, err = srv.HealthChecksInState(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)
	nodes = obj.(structs.HealthChecks)
	if len(nodes) != 2 {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[0].Node != "foo" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[1].Node != "bar" {
		t.Fatalf("bad: %v", nodes)
	}
}

func TestHealthNodeChecks(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	req, err := http.NewRequest("GET", "/v1/health/node/nope?dc=dc1", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.HealthNodeChecks(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)

	// Should be a non-nil empty list
	nodes := obj.(structs.HealthChecks)
	if nodes == nil || len(nodes) != 0 {
		t.Fatalf("bad: %v", obj)
	}

	req, err = http.NewRequest("GET",
		fmt.Sprintf("/v1/health/node/%s?dc=dc1", srv.agent.config.NodeName), nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp = httptest.NewRecorder()
	obj, err = srv.HealthNodeChecks(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)

	// Should be 1 health check for the server
	nodes = obj.(structs.HealthChecks)
	if len(nodes) != 1 {
		t.Fatalf("bad: %v", obj)
	}
}

func TestHealthServiceChecks(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	req, err := http.NewRequest("GET", "/v1/health/checks/consul?dc=dc1", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.HealthServiceChecks(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)

	// Should be a non-nil empty list
	nodes := obj.(structs.HealthChecks)
	if nodes == nil || len(nodes) != 0 {
		t.Fatalf("bad: %v", obj)
	}

	// Create a service check
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       srv.agent.config.NodeName,
		Address:    "127.0.0.1",
		Check: &structs.HealthCheck{
			Node:      srv.agent.config.NodeName,
			Name:      "consul check",
			ServiceID: "consul",
		},
	}

	var out struct{}
	if err = srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	req, err = http.NewRequest("GET", "/v1/health/checks/consul?dc=dc1", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp = httptest.NewRecorder()
	obj, err = srv.HealthServiceChecks(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)

	// Should be 1 health check for consul
	nodes = obj.(structs.HealthChecks)
	if len(nodes) != 1 {
		t.Fatalf("bad: %v", obj)
	}
}

func TestHealthServiceChecks_DistanceSort(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Create a service check
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "bar",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			ID:      "test",
			Service: "test",
		},
		Check: &structs.HealthCheck{
			Node:      "bar",
			Name:      "test check",
			ServiceID: "test",
		},
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	args.Node, args.Check.Node = "foo", "foo"
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	req, err := http.NewRequest("GET", "/v1/health/checks/test?dc=dc1&near=foo", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.HealthServiceChecks(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)
	nodes := obj.(structs.HealthChecks)
	if len(nodes) != 2 {
		t.Fatalf("bad: %v", obj)
	}
	if nodes[0].Node != "bar" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[1].Node != "foo" {
		t.Fatalf("bad: %v", nodes)
	}

	// Send an update for the node and wait for it to get applied.
	arg := structs.CoordinateUpdateRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Coord:      coordinate.NewCoordinate(coordinate.DefaultConfig()),
	}
	if err := srv.agent.RPC("Coordinate.Update", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	time.Sleep(200 * time.Millisecond)

	// Query again and now foo should have moved to the front of the line.
	resp = httptest.NewRecorder()
	obj, err = srv.HealthServiceChecks(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)
	nodes = obj.(structs.HealthChecks)
	if len(nodes) != 2 {
		t.Fatalf("bad: %v", obj)
	}
	if nodes[0].Node != "foo" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[1].Node != "bar" {
		t.Fatalf("bad: %v", nodes)
	}
}

func TestHealthServiceNodes(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	req, err := http.NewRequest("GET", "/v1/health/service/consul?dc=dc1", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.HealthServiceNodes(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	assertIndex(t, resp)

	// Should be 1 health check for consul
	nodes := obj.(structs.CheckServiceNodes)
	if len(nodes) != 1 {
		t.Fatalf("bad: %v", obj)
	}

	req, err = http.NewRequest("GET", "/v1/health/service/nope?dc=dc1", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp = httptest.NewRecorder()
	obj, err = srv.HealthServiceNodes(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	assertIndex(t, resp)

	// Should be a non-nil empty list
	nodes = obj.(structs.CheckServiceNodes)
	if nodes == nil || len(nodes) != 0 {
		t.Fatalf("bad: %v", obj)
	}

	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "bar",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			ID:      "test",
			Service: "test",
		},
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	req, err = http.NewRequest("GET", "/v1/health/service/test?dc=dc1", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp = httptest.NewRecorder()
	obj, err = srv.HealthServiceNodes(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	assertIndex(t, resp)

	// Should be a non-nil empty list for checks
	nodes = obj.(structs.CheckServiceNodes)
	if len(nodes) != 1 || nodes[0].Checks == nil || len(nodes[0].Checks) != 0 {
		t.Fatalf("bad: %v", obj)
	}
}

func TestHealthServiceNodes_DistanceSort(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Create a service check
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       "bar",
		Address:    "127.0.0.1",
		Service: &structs.NodeService{
			ID:      "test",
			Service: "test",
		},
		Check: &structs.HealthCheck{
			Node:      "bar",
			Name:      "test check",
			ServiceID: "test",
		},
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	args.Node, args.Check.Node = "foo", "foo"
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	req, err := http.NewRequest("GET", "/v1/health/service/test?dc=dc1&near=foo", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.HealthServiceNodes(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)
	nodes := obj.(structs.CheckServiceNodes)
	if len(nodes) != 2 {
		t.Fatalf("bad: %v", obj)
	}
	if nodes[0].Node.Node != "bar" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[1].Node.Node != "foo" {
		t.Fatalf("bad: %v", nodes)
	}

	// Send an update for the node and wait for it to get applied.
	arg := structs.CoordinateUpdateRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Coord:      coordinate.NewCoordinate(coordinate.DefaultConfig()),
	}
	if err := srv.agent.RPC("Coordinate.Update", &arg, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	time.Sleep(200 * time.Millisecond)

	// Query again and now foo should have moved to the front of the line.
	resp = httptest.NewRecorder()
	obj, err = srv.HealthServiceNodes(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	assertIndex(t, resp)
	nodes = obj.(structs.CheckServiceNodes)
	if len(nodes) != 2 {
		t.Fatalf("bad: %v", obj)
	}
	if nodes[0].Node.Node != "foo" {
		t.Fatalf("bad: %v", nodes)
	}
	if nodes[1].Node.Node != "bar" {
		t.Fatalf("bad: %v", nodes)
	}
}

func TestHealthServiceNodes_PassingFilter(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Create a failing service check
	args := &structs.RegisterRequest{
		Datacenter: "dc1",
		Node:       srv.agent.config.NodeName,
		Address:    "127.0.0.1",
		Check: &structs.HealthCheck{
			Node:      srv.agent.config.NodeName,
			Name:      "consul check",
			ServiceID: "consul",
			Status:    structs.HealthCritical,
		},
	}

	var out struct{}
	if err := srv.agent.RPC("Catalog.Register", args, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	req, err := http.NewRequest("GET", "/v1/health/service/consul?passing", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.HealthServiceNodes(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	assertIndex(t, resp)

	// Should be 0 health check for consul
	nodes := obj.(structs.CheckServiceNodes)
	if len(nodes) != 0 {
		t.Fatalf("bad: %v", obj)
	}
}

func TestFilterNonPassing(t *testing.T) {
	nodes := structs.CheckServiceNodes{
		structs.CheckServiceNode{
			Checks: structs.HealthChecks{
				&structs.HealthCheck{
					Status: structs.HealthCritical,
				},
				&structs.HealthCheck{
					Status: structs.HealthCritical,
				},
			},
		},
		structs.CheckServiceNode{
			Checks: structs.HealthChecks{
				&structs.HealthCheck{
					Status: structs.HealthCritical,
				},
				&structs.HealthCheck{
					Status: structs.HealthCritical,
				},
			},
		},
		structs.CheckServiceNode{
			Checks: structs.HealthChecks{
				&structs.HealthCheck{
					Status: structs.HealthPassing,
				},
			},
		},
	}
	out := filterNonPassing(nodes)
	if len(out) != 1 && reflect.DeepEqual(out[0], nodes[2]) {
		t.Fatalf("bad: %v", out)
	}
}
