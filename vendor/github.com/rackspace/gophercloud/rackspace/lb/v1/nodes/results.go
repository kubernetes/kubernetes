package nodes

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Node represents a back-end device, usually a virtual machine, that can
// handle traffic. It is assigned traffic based on its parent load balancer.
type Node struct {
	// The IP address or CIDR for this back-end node.
	Address string

	// The unique ID for this node.
	ID int

	// The port on which traffic is sent and received.
	Port int

	// The node's status.
	Status Status

	// The node's condition.
	Condition Condition

	// The priority at which this node will receive traffic if a weighted
	// algorithm is used by its parent load balancer. Ranges from 1 to 100.
	Weight int

	// Type of node.
	Type Type
}

// Type indicates whether the node is of a PRIMARY or SECONDARY nature.
type Type string

const (
	// PRIMARY nodes are in the normal rotation to receive traffic from the load
	// balancer.
	PRIMARY Type = "PRIMARY"

	// SECONDARY nodes are only in the rotation to receive traffic from the load
	// balancer when all the primary nodes fail. This provides a failover feature
	// that automatically routes traffic to the secondary node in the event that
	// the primary node is disabled or in a failing state. Note that active
	// health monitoring must be enabled on the load balancer to enable the
	// failover feature to the secondary node.
	SECONDARY Type = "SECONDARY"
)

// Condition represents the condition of a node.
type Condition string

const (
	// ENABLED indicates that the node is permitted to accept new connections.
	ENABLED Condition = "ENABLED"

	// DISABLED indicates that the node is not permitted to accept any new
	// connections regardless of session persistence configuration. Existing
	// connections are forcibly terminated.
	DISABLED Condition = "DISABLED"

	// DRAINING indicates that the node is allowed to service existing
	// established connections and connections that are being directed to it as a
	// result of the session persistence configuration.
	DRAINING Condition = "DRAINING"
)

// Status indicates whether the node can accept service traffic. If a node is
// not listening on its port or does not meet the conditions of the defined
// active health check for the load balancer, then the load balancer does not
// forward connections, and its status is listed as OFFLINE.
type Status string

const (
	// ONLINE indicates that the node is healthy and capable of receiving traffic
	// from the load balancer.
	ONLINE Status = "ONLINE"

	// OFFLINE indicates that the node is not in a position to receive service
	// traffic. It is usually switched into this state when a health check is not
	// satisfied with the node's response time.
	OFFLINE Status = "OFFLINE"
)

// NodePage is the page returned by a pager when traversing over a collection
// of nodes.
type NodePage struct {
	pagination.SinglePageBase
}

// IsEmpty checks whether a NodePage struct is empty.
func (p NodePage) IsEmpty() (bool, error) {
	is, err := ExtractNodes(p)
	if err != nil {
		return true, nil
	}
	return len(is) == 0, nil
}

func commonExtractNodes(body interface{}) ([]Node, error) {
	var resp struct {
		Nodes []Node `mapstructure:"nodes" json:"nodes"`
	}

	err := mapstructure.Decode(body, &resp)

	return resp.Nodes, err
}

// ExtractNodes accepts a Page struct, specifically a NodePage struct, and
// extracts the elements into a slice of Node structs. In other words, a
// generic collection is mapped into a relevant slice.
func ExtractNodes(page pagination.Page) ([]Node, error) {
	return commonExtractNodes(page.(NodePage).Body)
}

// CreateResult represents the result of a create operation. Since multiple
// nodes can be added in one operation, this result represents multiple nodes
// and should be treated as a typical pagination Page. Use its ExtractNodes
// method to get out a slice of Node structs.
type CreateResult struct {
	pagination.SinglePageBase
}

// ExtractNodes extracts a slice of Node structs from a CreateResult.
func (res CreateResult) ExtractNodes() ([]Node, error) {
	if res.Err != nil {
		return nil, res.Err
	}
	return commonExtractNodes(res.Body)
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
}

type commonResult struct {
	gophercloud.Result
}

// GetResult represents the result of a get operation.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an update operation.
type UpdateResult struct {
	gophercloud.ErrResult
}

func (r commonResult) Extract() (*Node, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Node Node `mapstructure:"node"`
	}

	err := mapstructure.Decode(r.Body, &response)

	return &response.Node, err
}

// NodeEvent represents a service event that occurred between a node and a
// load balancer.
type NodeEvent struct {
	ID              int
	DetailedMessage string
	NodeID          int
	Type            string
	Description     string
	Category        string
	Severity        string
	RelativeURI     string
	AccountID       int
	LoadBalancerID  int
	Title           string
	Author          string
	Created         string
}

// NodeEventPage is a concrete type which embeds the common SinglePageBase
// struct, and is used when traversing node event collections.
type NodeEventPage struct {
	pagination.SinglePageBase
}

// IsEmpty is a concrete function which indicates whether an NodeEventPage is
// empty or not.
func (r NodeEventPage) IsEmpty() (bool, error) {
	is, err := ExtractNodeEvents(r)
	if err != nil {
		return true, err
	}
	return len(is) == 0, nil
}

// ExtractNodeEvents accepts a Page struct, specifically a NodeEventPage
// struct, and extracts the elements into a slice of NodeEvent structs. In
// other words, the collection is mapped into a relevant slice.
func ExtractNodeEvents(page pagination.Page) ([]NodeEvent, error) {
	var resp struct {
		Events []NodeEvent `mapstructure:"nodeServiceEvents" json:"nodeServiceEvents"`
	}

	err := mapstructure.Decode(page.(NodeEventPage).Body, &resp)

	return resp.Events, err
}
