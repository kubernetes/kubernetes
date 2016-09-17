package api

import (
	"github.com/hashicorp/serf/coordinate"
)

// CoordinateEntry represents a node and its associated network coordinate.
type CoordinateEntry struct {
	Node  string
	Coord *coordinate.Coordinate
}

// CoordinateDatacenterMap represents a datacenter and its associated WAN
// nodes and their associates coordinates.
type CoordinateDatacenterMap struct {
	Datacenter  string
	Coordinates []CoordinateEntry
}

// Coordinate can be used to query the coordinate endpoints
type Coordinate struct {
	c *Client
}

// Coordinate returns a handle to the coordinate endpoints
func (c *Client) Coordinate() *Coordinate {
	return &Coordinate{c}
}

// Datacenters is used to return the coordinates of all the servers in the WAN
// pool.
func (c *Coordinate) Datacenters() ([]*CoordinateDatacenterMap, error) {
	r := c.c.newRequest("GET", "/v1/coordinate/datacenters")
	_, resp, err := requireOK(c.c.doRequest(r))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out []*CoordinateDatacenterMap
	if err := decodeBody(resp, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Nodes is used to return the coordinates of all the nodes in the LAN pool.
func (c *Coordinate) Nodes(q *QueryOptions) ([]*CoordinateEntry, *QueryMeta, error) {
	r := c.c.newRequest("GET", "/v1/coordinate/nodes")
	r.setQueryOptions(q)
	rtt, resp, err := requireOK(c.c.doRequest(r))
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	qm := &QueryMeta{}
	parseQueryMeta(resp, qm)
	qm.RequestTime = rtt

	var out []*CoordinateEntry
	if err := decodeBody(resp, &out); err != nil {
		return nil, nil, err
	}
	return out, qm, nil
}
