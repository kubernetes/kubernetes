package agent

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/serf/coordinate"
)

func TestCoordinate_Datacenters(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	req, err := http.NewRequest("GET", "/v1/coordinate/datacenters", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.CoordinateDatacenters(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	maps := obj.([]structs.DatacenterMap)
	if len(maps) != 1 ||
		maps[0].Datacenter != "dc1" ||
		len(maps[0].Coordinates) != 1 ||
		maps[0].Coordinates[0].Node != srv.agent.config.NodeName {
		t.Fatalf("bad: %v", maps)
	}
}

func TestCoordinate_Nodes(t *testing.T) {
	dir, srv := makeHTTPServer(t)
	defer os.RemoveAll(dir)
	defer srv.Shutdown()
	defer srv.agent.Shutdown()

	testutil.WaitForLeader(t, srv.agent.RPC, "dc1")

	// Make sure an empty list is non-nil.
	req, err := http.NewRequest("GET", "/v1/coordinate/nodes?dc=dc1", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp := httptest.NewRecorder()
	obj, err := srv.CoordinateNodes(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	coordinates := obj.(structs.Coordinates)
	if coordinates == nil || len(coordinates) != 0 {
		t.Fatalf("bad: %v", coordinates)
	}

	// Register the nodes.
	nodes := []string{"foo", "bar"}
	for _, node := range nodes {
		req := structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       node,
			Address:    "127.0.0.1",
		}
		var reply struct{}
		if err := srv.agent.RPC("Catalog.Register", &req, &reply); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Send some coordinates for a few nodes, waiting a little while for the
	// batch update to run.
	arg1 := structs.CoordinateUpdateRequest{
		Datacenter: "dc1",
		Node:       "foo",
		Coord:      coordinate.NewCoordinate(coordinate.DefaultConfig()),
	}
	var out struct{}
	if err := srv.agent.RPC("Coordinate.Update", &arg1, &out); err != nil {
		t.Fatalf("err: %v", err)
	}

	arg2 := structs.CoordinateUpdateRequest{
		Datacenter: "dc1",
		Node:       "bar",
		Coord:      coordinate.NewCoordinate(coordinate.DefaultConfig()),
	}
	if err := srv.agent.RPC("Coordinate.Update", &arg2, &out); err != nil {
		t.Fatalf("err: %v", err)
	}
	time.Sleep(200 * time.Millisecond)

	// Query back and check the nodes are present and sorted correctly.
	req, err = http.NewRequest("GET", "/v1/coordinate/nodes?dc=dc1", nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	resp = httptest.NewRecorder()
	obj, err = srv.CoordinateNodes(resp, req)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	coordinates = obj.(structs.Coordinates)
	if len(coordinates) != 2 ||
		coordinates[0].Node != "bar" ||
		coordinates[1].Node != "foo" {
		t.Fatalf("bad: %v", coordinates)
	}
}
