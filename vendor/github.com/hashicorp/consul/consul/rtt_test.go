package consul

import (
	"fmt"
	"math"
	"net/rpc"
	"os"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/testutil"
	"github.com/hashicorp/net-rpc-msgpackrpc"
	"github.com/hashicorp/serf/coordinate"
)

// generateCoordinate creates a new coordinate with the given distance from the
// origin.
func generateCoordinate(rtt time.Duration) *coordinate.Coordinate {
	coord := coordinate.NewCoordinate(coordinate.DefaultConfig())
	coord.Vec[0] = rtt.Seconds()
	coord.Height = 0
	return coord
}

// verifyNodeSort makes sure the order of the nodes in the slice is the same as
// the expected order, expressed as a comma-separated string.
func verifyNodeSort(t *testing.T, nodes structs.Nodes, expected string) {
	vec := make([]string, len(nodes))
	for i, node := range nodes {
		vec[i] = node.Node
	}
	actual := strings.Join(vec, ",")
	if actual != expected {
		t.Fatalf("bad sort: %s != %s", actual, expected)
	}
}

// verifyServiceNodeSort makes sure the order of the nodes in the slice is the
// same as the expected order, expressed as a comma-separated string.
func verifyServiceNodeSort(t *testing.T, nodes structs.ServiceNodes, expected string) {
	vec := make([]string, len(nodes))
	for i, node := range nodes {
		vec[i] = node.Node
	}
	actual := strings.Join(vec, ",")
	if actual != expected {
		t.Fatalf("bad sort: %s != %s", actual, expected)
	}
}

// verifyHealthCheckSort makes sure the order of the nodes in the slice is the
// same as the expected order, expressed as a comma-separated string.
func verifyHealthCheckSort(t *testing.T, checks structs.HealthChecks, expected string) {
	vec := make([]string, len(checks))
	for i, check := range checks {
		vec[i] = check.Node
	}
	actual := strings.Join(vec, ",")
	if actual != expected {
		t.Fatalf("bad sort: %s != %s", actual, expected)
	}
}

// verifyCheckServiceNodeSort makes sure the order of the nodes in the slice is
// the same as the expected order, expressed as a comma-separated string.
func verifyCheckServiceNodeSort(t *testing.T, nodes structs.CheckServiceNodes, expected string) {
	vec := make([]string, len(nodes))
	for i, node := range nodes {
		vec[i] = node.Node.Node
	}
	actual := strings.Join(vec, ",")
	if actual != expected {
		t.Fatalf("bad sort: %s != %s", actual, expected)
	}
}

// seedCoordinates uses the client to set up a set of nodes with a specific
// set of distances from the origin. We also include the server so that we
// can wait for the coordinates to get committed to the Raft log.
//
// Here's the layout of the nodes:
//
//       node3 node2 node5                         node4       node1
//   |     |     |     |     |     |     |     |     |     |     |
//   0     1     2     3     4     5     6     7     8     9     10  (ms)
//
func seedCoordinates(t *testing.T, codec rpc.ClientCodec, server *Server) {
	// Register some nodes.
	for i := 0; i < 5; i++ {
		req := structs.RegisterRequest{
			Datacenter: "dc1",
			Node:       fmt.Sprintf("node%d", i+1),
			Address:    "127.0.0.1",
		}
		var reply struct{}
		if err := msgpackrpc.CallWithCodec(codec, "Catalog.Register", &req, &reply); err != nil {
			t.Fatalf("err: %v", err)
		}
	}

	// Seed the fixed setup of the nodes.
	updates := []structs.CoordinateUpdateRequest{
		structs.CoordinateUpdateRequest{
			Datacenter: "dc1",
			Node:       "node1",
			Coord:      generateCoordinate(10 * time.Millisecond),
		},
		structs.CoordinateUpdateRequest{
			Datacenter: "dc1",
			Node:       "node2",
			Coord:      generateCoordinate(2 * time.Millisecond),
		},
		structs.CoordinateUpdateRequest{
			Datacenter: "dc1",
			Node:       "node3",
			Coord:      generateCoordinate(1 * time.Millisecond),
		},
		structs.CoordinateUpdateRequest{
			Datacenter: "dc1",
			Node:       "node4",
			Coord:      generateCoordinate(8 * time.Millisecond),
		},
		structs.CoordinateUpdateRequest{
			Datacenter: "dc1",
			Node:       "node5",
			Coord:      generateCoordinate(3 * time.Millisecond),
		},
	}

	// Apply the updates and wait a while for the batch to get committed to
	// the Raft log.
	for _, update := range updates {
		var out struct{}
		if err := msgpackrpc.CallWithCodec(codec, "Coordinate.Update", &update, &out); err != nil {
			t.Fatalf("err: %v", err)
		}
	}
	time.Sleep(2 * server.config.CoordinateUpdatePeriod)
}

func TestRTT_sortNodesByDistanceFrom(t *testing.T) {
	dir, server := testServer(t)
	defer os.RemoveAll(dir)
	defer server.Shutdown()

	codec := rpcClient(t, server)
	defer codec.Close()
	testutil.WaitForLeader(t, server.RPC, "dc1")

	seedCoordinates(t, codec, server)
	nodes := structs.Nodes{
		&structs.Node{Node: "apple"},
		&structs.Node{Node: "node1"},
		&structs.Node{Node: "node2"},
		&structs.Node{Node: "node3"},
		&structs.Node{Node: "node4"},
		&structs.Node{Node: "node5"},
	}

	// The zero value for the source should not trigger any sorting.
	var source structs.QuerySource
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyNodeSort(t, nodes, "apple,node1,node2,node3,node4,node5")

	// Same for a source in some other DC.
	source.Node = "node1"
	source.Datacenter = "dc2"
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyNodeSort(t, nodes, "apple,node1,node2,node3,node4,node5")

	// Same for a source node in our DC that we have no coordinate for.
	source.Node = "apple"
	source.Datacenter = "dc1"
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyNodeSort(t, nodes, "apple,node1,node2,node3,node4,node5")

	// Set source to legit values relative to node1 but disable coordinates.
	source.Node = "node1"
	source.Datacenter = "dc1"
	server.config.DisableCoordinates = true
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyNodeSort(t, nodes, "apple,node1,node2,node3,node4,node5")

	// Now enable coordinates and sort relative to node1, note that apple
	// doesn't have any seeded coordinate info so it should end up at the
	// end, despite its lexical hegemony.
	server.config.DisableCoordinates = false
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyNodeSort(t, nodes, "node1,node4,node5,node2,node3,apple")
}

func TestRTT_sortNodesByDistanceFrom_Nodes(t *testing.T) {
	dir, server := testServer(t)
	defer os.RemoveAll(dir)
	defer server.Shutdown()

	codec := rpcClient(t, server)
	defer codec.Close()
	testutil.WaitForLeader(t, server.RPC, "dc1")

	seedCoordinates(t, codec, server)
	nodes := structs.Nodes{
		&structs.Node{Node: "apple"},
		&structs.Node{Node: "node1"},
		&structs.Node{Node: "node2"},
		&structs.Node{Node: "node3"},
		&structs.Node{Node: "node4"},
		&structs.Node{Node: "node5"},
	}

	// Now sort relative to node1, note that apple doesn't have any
	// seeded coordinate info so it should end up at the end, despite
	// its lexical hegemony.
	var source structs.QuerySource
	source.Node = "node1"
	source.Datacenter = "dc1"
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyNodeSort(t, nodes, "node1,node4,node5,node2,node3,apple")

	// Try another sort from node2. Note that node5 and node3 are the
	// same distance away so the stable sort should preserve the order
	// they were in from the previous sort.
	source.Node = "node2"
	source.Datacenter = "dc1"
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyNodeSort(t, nodes, "node2,node5,node3,node4,node1,apple")

	// Let's exercise the stable sort explicitly to make sure we didn't
	// just get lucky.
	nodes[1], nodes[2] = nodes[2], nodes[1]
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyNodeSort(t, nodes, "node2,node3,node5,node4,node1,apple")
}

func TestRTT_sortNodesByDistanceFrom_ServiceNodes(t *testing.T) {
	dir, server := testServer(t)
	defer os.RemoveAll(dir)
	defer server.Shutdown()

	codec := rpcClient(t, server)
	defer codec.Close()
	testutil.WaitForLeader(t, server.RPC, "dc1")

	seedCoordinates(t, codec, server)
	nodes := structs.ServiceNodes{
		&structs.ServiceNode{Node: "apple"},
		&structs.ServiceNode{Node: "node1"},
		&structs.ServiceNode{Node: "node2"},
		&structs.ServiceNode{Node: "node3"},
		&structs.ServiceNode{Node: "node4"},
		&structs.ServiceNode{Node: "node5"},
	}

	// Now sort relative to node1, note that apple doesn't have any
	// seeded coordinate info so it should end up at the end, despite
	// its lexical hegemony.
	var source structs.QuerySource
	source.Node = "node1"
	source.Datacenter = "dc1"
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyServiceNodeSort(t, nodes, "node1,node4,node5,node2,node3,apple")

	// Try another sort from node2. Note that node5 and node3 are the
	// same distance away so the stable sort should preserve the order
	// they were in from the previous sort.
	source.Node = "node2"
	source.Datacenter = "dc1"
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyServiceNodeSort(t, nodes, "node2,node5,node3,node4,node1,apple")

	// Let's exercise the stable sort explicitly to make sure we didn't
	// just get lucky.
	nodes[1], nodes[2] = nodes[2], nodes[1]
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyServiceNodeSort(t, nodes, "node2,node3,node5,node4,node1,apple")
}

func TestRTT_sortNodesByDistanceFrom_HealthChecks(t *testing.T) {
	dir, server := testServer(t)
	defer os.RemoveAll(dir)
	defer server.Shutdown()

	codec := rpcClient(t, server)
	defer codec.Close()
	testutil.WaitForLeader(t, server.RPC, "dc1")

	seedCoordinates(t, codec, server)
	checks := structs.HealthChecks{
		&structs.HealthCheck{Node: "apple"},
		&structs.HealthCheck{Node: "node1"},
		&structs.HealthCheck{Node: "node2"},
		&structs.HealthCheck{Node: "node3"},
		&structs.HealthCheck{Node: "node4"},
		&structs.HealthCheck{Node: "node5"},
	}

	// Now sort relative to node1, note that apple doesn't have any
	// seeded coordinate info so it should end up at the end, despite
	// its lexical hegemony.
	var source structs.QuerySource
	source.Node = "node1"
	source.Datacenter = "dc1"
	if err := server.sortNodesByDistanceFrom(source, checks); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyHealthCheckSort(t, checks, "node1,node4,node5,node2,node3,apple")

	// Try another sort from node2. Note that node5 and node3 are the
	// same distance away so the stable sort should preserve the order
	// they were in from the previous sort.
	source.Node = "node2"
	source.Datacenter = "dc1"
	if err := server.sortNodesByDistanceFrom(source, checks); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyHealthCheckSort(t, checks, "node2,node5,node3,node4,node1,apple")

	// Let's exercise the stable sort explicitly to make sure we didn't
	// just get lucky.
	checks[1], checks[2] = checks[2], checks[1]
	if err := server.sortNodesByDistanceFrom(source, checks); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyHealthCheckSort(t, checks, "node2,node3,node5,node4,node1,apple")
}

func TestRTT_sortNodesByDistanceFrom_CheckServiceNodes(t *testing.T) {
	dir, server := testServer(t)
	defer os.RemoveAll(dir)
	defer server.Shutdown()

	codec := rpcClient(t, server)
	defer codec.Close()
	testutil.WaitForLeader(t, server.RPC, "dc1")

	seedCoordinates(t, codec, server)
	nodes := structs.CheckServiceNodes{
		structs.CheckServiceNode{Node: &structs.Node{Node: "apple"}},
		structs.CheckServiceNode{Node: &structs.Node{Node: "node1"}},
		structs.CheckServiceNode{Node: &structs.Node{Node: "node2"}},
		structs.CheckServiceNode{Node: &structs.Node{Node: "node3"}},
		structs.CheckServiceNode{Node: &structs.Node{Node: "node4"}},
		structs.CheckServiceNode{Node: &structs.Node{Node: "node5"}},
	}

	// Now sort relative to node1, note that apple doesn't have any
	// seeded coordinate info so it should end up at the end, despite
	// its lexical hegemony.
	var source structs.QuerySource
	source.Node = "node1"
	source.Datacenter = "dc1"
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyCheckServiceNodeSort(t, nodes, "node1,node4,node5,node2,node3,apple")

	// Try another sort from node2. Note that node5 and node3 are the
	// same distance away so the stable sort should preserve the order
	// they were in from the previous sort.
	source.Node = "node2"
	source.Datacenter = "dc1"
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyCheckServiceNodeSort(t, nodes, "node2,node5,node3,node4,node1,apple")

	// Let's exercise the stable sort explicitly to make sure we didn't
	// just get lucky.
	nodes[1], nodes[2] = nodes[2], nodes[1]
	if err := server.sortNodesByDistanceFrom(source, nodes); err != nil {
		t.Fatalf("err: %v", err)
	}
	verifyCheckServiceNodeSort(t, nodes, "node2,node3,node5,node4,node1,apple")
}

// mockNodeMap is keyed by node name and the values are the coordinates of the
// node.
type mockNodeMap map[string]*coordinate.Coordinate

// mockServer is used to provide a serfer interface for unit tests. The key is
// DC, which selects a map from node name to coordinate for that node.
type mockServer map[string]mockNodeMap

// newMockServer is used to generate a serfer interface that presents a known DC
// topology for unit tests. The server is in dc0.
//
// Here's the layout of the nodes:
//
//            /----   dc1         ----\         /-  dc2  -\ /-  dc0  -\
//             node2 node1       node3             node1       node1
//   |     |     |     |     |     |     |     |     |     |     |
//   0     1     2     3     4     5     6     7     8     9     10  (ms)
//
// We also include a node4 in dc1 with no known coordinate, as well as a
// mysterious dcX with no nodes with known coordinates.
//
func newMockServer() *mockServer {
	s := make(mockServer)
	s["dc0"] = mockNodeMap{
		"dc0.node1": generateCoordinate(10 * time.Millisecond),
	}
	s["dc1"] = mockNodeMap{
		"dc1.node1": generateCoordinate(3 * time.Millisecond),
		"dc1.node2": generateCoordinate(2 * time.Millisecond),
		"dc1.node3": generateCoordinate(5 * time.Millisecond),
		"dc1.node4": nil, // no known coordinate
	}
	s["dc2"] = mockNodeMap{
		"dc2.node1": generateCoordinate(8 * time.Millisecond),
	}
	s["dcX"] = mockNodeMap{
		"dcX.node1": nil, // no known coordinate
	}
	return &s
}

// See serfer.
func (s *mockServer) GetDatacenter() string {
	return "dc0"
}

// See serfer.
func (s *mockServer) GetCoordinate() (*coordinate.Coordinate, error) {
	return (*s)["dc0"]["dc0.node1"], nil
}

// See serfer.
func (s *mockServer) GetCachedCoordinate(node string) (*coordinate.Coordinate, bool) {
	for _, nodes := range *s {
		for n, coord := range nodes {
			if n == node && coord != nil {
				return coord, true
			}
		}
	}
	return nil, false
}

// See serfer.
func (s *mockServer) GetNodesForDatacenter(dc string) []string {
	nodes := make([]string, 0)
	if n, ok := (*s)[dc]; ok {
		for name := range n {
			nodes = append(nodes, name)
		}
	}
	sort.Strings(nodes)
	return nodes
}

func TestRTT_getDatacenterDistance(t *testing.T) {
	s := newMockServer()

	// The serfer's own DC is always 0 ms away.
	if dist, err := getDatacenterDistance(s, "dc0"); err != nil || dist != 0.0 {
		t.Fatalf("bad: %v err: %v", dist, err)
	}

	// Check a DC with no coordinates, which should give positive infinity.
	if dist, err := getDatacenterDistance(s, "dcX"); err != nil || dist != math.Inf(1.0) {
		t.Fatalf("bad: %v err: %v", dist, err)
	}

	// Similar for a totally unknown DC.
	if dist, err := getDatacenterDistance(s, "acdc"); err != nil || dist != math.Inf(1.0) {
		t.Fatalf("bad: %v err: %v", dist, err)
	}

	// Check the trivial median case (just one node).
	if dist, err := getDatacenterDistance(s, "dc2"); err != nil || dist != 0.002 {
		t.Fatalf("bad: %v err: %v", dist, err)
	}

	// Check the more interesting median case, note that there's a mystery
	// node4 in there that should be excluded to make the distances sort
	// like this:
	//
	// [0] node3 (0.005), [1] node1 (0.007), [2] node2 (0.008)
	//
	// So the median should be at index 3 / 2 = 1 -> 0.007.
	if dist, err := getDatacenterDistance(s, "dc1"); err != nil || dist != 0.007 {
		t.Fatalf("bad: %v err: %v", dist, err)
	}
}

func TestRTT_sortDatacentersByDistance(t *testing.T) {
	s := newMockServer()

	dcs := []string{"acdc", "dc0", "dc1", "dc2", "dcX"}
	if err := sortDatacentersByDistance(s, dcs); err != nil {
		t.Fatalf("err: %v", err)
	}

	expected := "dc0,dc2,dc1,acdc,dcX"
	if actual := strings.Join(dcs, ","); actual != expected {
		t.Fatalf("bad sort: %s != %s", actual, expected)
	}

	// Make sure the sort is stable and we didn't just get lucky.
	dcs = []string{"dcX", "dc0", "dc1", "dc2", "acdc"}
	if err := sortDatacentersByDistance(s, dcs); err != nil {
		t.Fatalf("err: %v", err)
	}

	expected = "dc0,dc2,dc1,dcX,acdc"
	if actual := strings.Join(dcs, ","); actual != expected {
		t.Fatalf("bad sort: %s != %s", actual, expected)
	}
}

func TestRTT_getDatacenterMaps(t *testing.T) {
	s := newMockServer()

	dcs := []string{"dc0", "acdc", "dc1", "dc2", "dcX"}
	maps := getDatacenterMaps(s, dcs)

	if len(maps) != 5 {
		t.Fatalf("bad: %v", maps)
	}

	if maps[0].Datacenter != "dc0" || len(maps[0].Coordinates) != 1 ||
		maps[0].Coordinates[0].Node != "dc0.node1" {
		t.Fatalf("bad: %v", maps[0])
	}
	verifyCoordinatesEqual(t, maps[0].Coordinates[0].Coord,
		generateCoordinate(10*time.Millisecond))

	if maps[1].Datacenter != "acdc" || len(maps[1].Coordinates) != 0 {
		t.Fatalf("bad: %v", maps[1])
	}

	if maps[2].Datacenter != "dc1" || len(maps[2].Coordinates) != 3 ||
		maps[2].Coordinates[0].Node != "dc1.node1" ||
		maps[2].Coordinates[1].Node != "dc1.node2" ||
		maps[2].Coordinates[2].Node != "dc1.node3" {
		t.Fatalf("bad: %v", maps[2])
	}
	verifyCoordinatesEqual(t, maps[2].Coordinates[0].Coord,
		generateCoordinate(3*time.Millisecond))
	verifyCoordinatesEqual(t, maps[2].Coordinates[1].Coord,
		generateCoordinate(2*time.Millisecond))
	verifyCoordinatesEqual(t, maps[2].Coordinates[2].Coord,
		generateCoordinate(5*time.Millisecond))

	if maps[3].Datacenter != "dc2" || len(maps[3].Coordinates) != 1 ||
		maps[3].Coordinates[0].Node != "dc2.node1" {
		t.Fatalf("bad: %v", maps[3])
	}
	verifyCoordinatesEqual(t, maps[3].Coordinates[0].Coord,
		generateCoordinate(8*time.Millisecond))

	if maps[4].Datacenter != "dcX" || len(maps[4].Coordinates) != 0 {
		t.Fatalf("bad: %v", maps[4])
	}
}

func TestRTT_getDatacentersByDistance(t *testing.T) {
	dir1, s1 := testServerWithConfig(t, func(c *Config) {
		c.Datacenter = "xxx"
	})
	defer os.RemoveAll(dir1)
	defer s1.Shutdown()
	codec1 := rpcClient(t, s1)
	defer codec1.Close()

	dir2, s2 := testServerWithConfig(t, func(c *Config) {
		c.Datacenter = "dc1"
	})
	defer os.RemoveAll(dir2)
	defer s2.Shutdown()
	codec2 := rpcClient(t, s2)
	defer codec2.Close()

	dir3, s3 := testServerWithConfig(t, func(c *Config) {
		c.Datacenter = "dc2"
	})
	defer os.RemoveAll(dir3)
	defer s3.Shutdown()
	codec3 := rpcClient(t, s3)
	defer codec3.Close()

	testutil.WaitForLeader(t, s1.RPC, "xxx")
	testutil.WaitForLeader(t, s2.RPC, "dc1")
	testutil.WaitForLeader(t, s3.RPC, "dc2")

	// Do the WAN joins.
	addr := fmt.Sprintf("127.0.0.1:%d",
		s1.config.SerfWANConfig.MemberlistConfig.BindPort)
	if _, err := s2.JoinWAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}
	if _, err := s3.JoinWAN([]string{addr}); err != nil {
		t.Fatalf("err: %v", err)
	}
	testutil.WaitForResult(
		func() (bool, error) {
			return len(s1.WANMembers()) > 2, nil
		},
		func(err error) {
			t.Fatalf("Failed waiting for WAN join: %v", err)
		})

	// Get the DCs by distance. We don't have coordinate updates yet, but
	// having xxx show up first proves we are calling the distance sort,
	// since it would normally do a string sort.
	dcs, err := s1.getDatacentersByDistance()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(dcs) != 3 || dcs[0] != "xxx" {
		t.Fatalf("bad: %v", dcs)
	}

	// Let's disable coordinates just to be sure.
	s1.config.DisableCoordinates = true
	dcs, err = s1.getDatacentersByDistance()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if len(dcs) != 3 || dcs[0] != "dc1" {
		t.Fatalf("bad: %v", dcs)
	}
}
