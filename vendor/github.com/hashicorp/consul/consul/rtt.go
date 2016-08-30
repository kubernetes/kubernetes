package consul

import (
	"fmt"
	"math"
	"sort"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/serf/coordinate"
)

// computeDistance returns the distance between the two network coordinates in
// seconds. If either of the coordinates is nil then this will return positive
// infinity.
func computeDistance(a *coordinate.Coordinate, b *coordinate.Coordinate) float64 {
	if a == nil || b == nil {
		return math.Inf(1.0)
	}

	return a.DistanceTo(b).Seconds()
}

// nodeSorter takes a list of nodes and a parallel vector of distances and
// implements sort.Interface, keeping both structures coherent and sorting by
// distance.
type nodeSorter struct {
	Nodes structs.Nodes
	Vec   []float64
}

// newNodeSorter returns a new sorter for the given source coordinate and set of
// nodes.
func (s *Server) newNodeSorter(c *coordinate.Coordinate, nodes structs.Nodes) (sort.Interface, error) {
	state := s.fsm.State()
	vec := make([]float64, len(nodes))
	for i, node := range nodes {
		coord, err := state.CoordinateGetRaw(node.Node)
		if err != nil {
			return nil, err
		}
		vec[i] = computeDistance(c, coord)
	}
	return &nodeSorter{nodes, vec}, nil
}

// See sort.Interface.
func (n *nodeSorter) Len() int {
	return len(n.Nodes)
}

// See sort.Interface.
func (n *nodeSorter) Swap(i, j int) {
	n.Nodes[i], n.Nodes[j] = n.Nodes[j], n.Nodes[i]
	n.Vec[i], n.Vec[j] = n.Vec[j], n.Vec[i]
}

// See sort.Interface.
func (n *nodeSorter) Less(i, j int) bool {
	return n.Vec[i] < n.Vec[j]
}

// serviceNodeSorter takes a list of service nodes and a parallel vector of
// distances and implements sort.Interface, keeping both structures coherent and
// sorting by distance.
type serviceNodeSorter struct {
	Nodes structs.ServiceNodes
	Vec   []float64
}

// newServiceNodeSorter returns a new sorter for the given source coordinate and
// set of service nodes.
func (s *Server) newServiceNodeSorter(c *coordinate.Coordinate, nodes structs.ServiceNodes) (sort.Interface, error) {
	state := s.fsm.State()
	vec := make([]float64, len(nodes))
	for i, node := range nodes {
		coord, err := state.CoordinateGetRaw(node.Node)
		if err != nil {
			return nil, err
		}
		vec[i] = computeDistance(c, coord)
	}
	return &serviceNodeSorter{nodes, vec}, nil
}

// See sort.Interface.
func (n *serviceNodeSorter) Len() int {
	return len(n.Nodes)
}

// See sort.Interface.
func (n *serviceNodeSorter) Swap(i, j int) {
	n.Nodes[i], n.Nodes[j] = n.Nodes[j], n.Nodes[i]
	n.Vec[i], n.Vec[j] = n.Vec[j], n.Vec[i]
}

// See sort.Interface.
func (n *serviceNodeSorter) Less(i, j int) bool {
	return n.Vec[i] < n.Vec[j]
}

// serviceNodeSorter takes a list of health checks and a parallel vector of
// distances and implements sort.Interface, keeping both structures coherent and
// sorting by distance.
type healthCheckSorter struct {
	Checks structs.HealthChecks
	Vec    []float64
}

// newHealthCheckSorter returns a new sorter for the given source coordinate and
// set of health checks with nodes.
func (s *Server) newHealthCheckSorter(c *coordinate.Coordinate, checks structs.HealthChecks) (sort.Interface, error) {
	state := s.fsm.State()
	vec := make([]float64, len(checks))
	for i, check := range checks {
		coord, err := state.CoordinateGetRaw(check.Node)
		if err != nil {
			return nil, err
		}
		vec[i] = computeDistance(c, coord)
	}
	return &healthCheckSorter{checks, vec}, nil
}

// See sort.Interface.
func (n *healthCheckSorter) Len() int {
	return len(n.Checks)
}

// See sort.Interface.
func (n *healthCheckSorter) Swap(i, j int) {
	n.Checks[i], n.Checks[j] = n.Checks[j], n.Checks[i]
	n.Vec[i], n.Vec[j] = n.Vec[j], n.Vec[i]
}

// See sort.Interface.
func (n *healthCheckSorter) Less(i, j int) bool {
	return n.Vec[i] < n.Vec[j]
}

// checkServiceNodeSorter takes a list of service nodes and a parallel vector of
// distances and implements sort.Interface, keeping both structures coherent and
// sorting by distance.
type checkServiceNodeSorter struct {
	Nodes structs.CheckServiceNodes
	Vec   []float64
}

// newCheckServiceNodeSorter returns a new sorter for the given source coordinate
// and set of nodes with health checks.
func (s *Server) newCheckServiceNodeSorter(c *coordinate.Coordinate, nodes structs.CheckServiceNodes) (sort.Interface, error) {
	state := s.fsm.State()
	vec := make([]float64, len(nodes))
	for i, node := range nodes {
		coord, err := state.CoordinateGetRaw(node.Node.Node)
		if err != nil {
			return nil, err
		}
		vec[i] = computeDistance(c, coord)
	}
	return &checkServiceNodeSorter{nodes, vec}, nil
}

// See sort.Interface.
func (n *checkServiceNodeSorter) Len() int {
	return len(n.Nodes)
}

// See sort.Interface.
func (n *checkServiceNodeSorter) Swap(i, j int) {
	n.Nodes[i], n.Nodes[j] = n.Nodes[j], n.Nodes[i]
	n.Vec[i], n.Vec[j] = n.Vec[j], n.Vec[i]
}

// See sort.Interface.
func (n *checkServiceNodeSorter) Less(i, j int) bool {
	return n.Vec[i] < n.Vec[j]
}

// newSorterByDistanceFrom returns a sorter for the given type.
func (s *Server) newSorterByDistanceFrom(c *coordinate.Coordinate, subj interface{}) (sort.Interface, error) {
	switch v := subj.(type) {
	case structs.Nodes:
		return s.newNodeSorter(c, v)
	case structs.ServiceNodes:
		return s.newServiceNodeSorter(c, v)
	case structs.HealthChecks:
		return s.newHealthCheckSorter(c, v)
	case structs.CheckServiceNodes:
		return s.newCheckServiceNodeSorter(c, v)
	default:
		panic(fmt.Errorf("Unhandled type passed to newSorterByDistanceFrom: %#v", subj))
	}
}

// sortNodesByDistanceFrom is used to sort results from our service catalog based
// on the round trip time from the given source node. Nodes with missing coordinates
// will get stable sorted at the end of the list.
//
// If coordinates are disabled this will be a no-op.
func (s *Server) sortNodesByDistanceFrom(source structs.QuerySource, subj interface{}) error {
	// Make it safe to call this without having to check if coordinates are
	// disabled first.
	if s.config.DisableCoordinates {
		return nil
	}

	// We can't sort if there's no source node.
	if source.Node == "" {
		return nil
	}

	// We can't compare coordinates across DCs.
	if source.Datacenter != s.config.Datacenter {
		return nil
	}

	// There won't always be a coordinate for the source node. If there's not
	// one then we can bail out because there's no meaning for the sort.
	state := s.fsm.State()
	coord, err := state.CoordinateGetRaw(source.Node)
	if err != nil {
		return err
	}
	if coord == nil {
		return nil
	}

	// Do the sort!
	sorter, err := s.newSorterByDistanceFrom(coord, subj)
	if err != nil {
		return err
	}
	sort.Stable(sorter)
	return nil
}

// serfer provides the coordinate information we need from the Server in an
// interface that's easy to mock out for testing. Without this, we'd have to
// do some really painful setup to get good unit test coverage of all the cases.
type serfer interface {
	GetDatacenter() string
	GetCoordinate() (*coordinate.Coordinate, error)
	GetCachedCoordinate(node string) (*coordinate.Coordinate, bool)
	GetNodesForDatacenter(dc string) []string
}

// serverSerfer wraps a Server with the serfer interface.
type serverSerfer struct {
	server *Server
}

// See serfer.
func (s *serverSerfer) GetDatacenter() string {
	return s.server.config.Datacenter
}

// See serfer.
func (s *serverSerfer) GetCoordinate() (*coordinate.Coordinate, error) {
	return s.server.serfWAN.GetCoordinate()
}

// See serfer.
func (s *serverSerfer) GetCachedCoordinate(node string) (*coordinate.Coordinate, bool) {
	return s.server.serfWAN.GetCachedCoordinate(node)
}

// See serfer.
func (s *serverSerfer) GetNodesForDatacenter(dc string) []string {
	s.server.remoteLock.RLock()
	defer s.server.remoteLock.RUnlock()

	nodes := make([]string, 0)
	for _, part := range s.server.remoteConsuls[dc] {
		nodes = append(nodes, part.Name)
	}
	return nodes
}

// getDatacenterDistance will return the median round trip time estimate for
// the given DC from the given serfer, in seconds. This will return positive
// infinity if no coordinates are available.
func getDatacenterDistance(s serfer, dc string) (float64, error) {
	// If this is the serfer's DC then just bail with zero RTT.
	if dc == s.GetDatacenter() {
		return 0.0, nil
	}

	// Otherwise measure from the serfer to the nodes in the other DC.
	coord, err := s.GetCoordinate()
	if err != nil {
		return 0.0, err
	}

	// Fetch all the nodes in the DC and record their distance, if available.
	nodes := s.GetNodesForDatacenter(dc)
	subvec := make([]float64, 0, len(nodes))
	for _, node := range nodes {
		if other, ok := s.GetCachedCoordinate(node); ok {
			subvec = append(subvec, computeDistance(coord, other))
		}
	}

	// Compute the median by sorting and taking the middle item.
	if len(subvec) > 0 {
		sort.Float64s(subvec)
		return subvec[len(subvec)/2], nil
	}

	// Return the default infinity value.
	return computeDistance(coord, nil), nil
}

// datacenterSorter takes a list of DC names and a parallel vector of distances
// and implements sort.Interface, keeping both structures coherent and sorting
// by distance.
type datacenterSorter struct {
	Names []string
	Vec   []float64
}

// See sort.Interface.
func (n *datacenterSorter) Len() int {
	return len(n.Names)
}

// See sort.Interface.
func (n *datacenterSorter) Swap(i, j int) {
	n.Names[i], n.Names[j] = n.Names[j], n.Names[i]
	n.Vec[i], n.Vec[j] = n.Vec[j], n.Vec[i]
}

// See sort.Interface.
func (n *datacenterSorter) Less(i, j int) bool {
	return n.Vec[i] < n.Vec[j]
}

// sortDatacentersByDistance will sort the given list of DCs based on the
// median RTT to all nodes the given serfer knows about from the WAN gossip
// pool). DCs with missing coordinates will be stable sorted to the end of the
// list.
func sortDatacentersByDistance(s serfer, dcs []string) error {
	// Build up a list of median distances to the other DCs.
	vec := make([]float64, len(dcs))
	for i, dc := range dcs {
		rtt, err := getDatacenterDistance(s, dc)
		if err != nil {
			return err
		}

		vec[i] = rtt
	}

	sorter := &datacenterSorter{dcs, vec}
	sort.Stable(sorter)
	return nil
}

// getDatacenterMaps returns the raw coordinates of all the nodes in the
// given list of DCs (the output list will preserve the incoming order).
func (s *Server) getDatacenterMaps(dcs []string) []structs.DatacenterMap {
	serfer := serverSerfer{s}
	return getDatacenterMaps(&serfer, dcs)
}

// getDatacenterMaps returns the raw coordinates of all the nodes in the
// given list of DCs (the output list will preserve the incoming order).
func getDatacenterMaps(s serfer, dcs []string) []structs.DatacenterMap {
	maps := make([]structs.DatacenterMap, 0, len(dcs))
	for _, dc := range dcs {
		m := structs.DatacenterMap{Datacenter: dc}
		nodes := s.GetNodesForDatacenter(dc)
		for _, node := range nodes {
			if coord, ok := s.GetCachedCoordinate(node); ok {
				entry := &structs.Coordinate{Node: node, Coord: coord}
				m.Coordinates = append(m.Coordinates, entry)
			}
		}
		maps = append(maps, m)
	}
	return maps
}

// getDatacentersByDistance will return the list of DCs, sorted in order
// of increasing distance based on the median distance to that DC from all
// servers we know about in the WAN gossip pool. This will sort by name all
// other things being equal (or if coordinates are disabled).
func (s *Server) getDatacentersByDistance() ([]string, error) {
	s.remoteLock.RLock()
	defer s.remoteLock.RUnlock()

	var dcs []string
	for dc := range s.remoteConsuls {
		dcs = append(dcs, dc)
	}

	// Sort by name first, since the coordinate sort is stable.
	sort.Strings(dcs)

	// Make it safe to call this without having to check if coordinates are
	// disabled first.
	if s.config.DisableCoordinates {
		return dcs, nil
	}

	// Do the sort!
	serfer := serverSerfer{s}
	if err := sortDatacentersByDistance(&serfer, dcs); err != nil {
		return nil, err
	}

	return dcs, nil
}
