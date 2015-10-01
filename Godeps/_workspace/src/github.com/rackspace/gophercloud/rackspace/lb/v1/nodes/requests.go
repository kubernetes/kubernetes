package nodes

import (
	"errors"
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List is the operation responsible for returning a paginated collection of
// load balancer nodes. It requires the node ID, its parent load balancer ID,
// and optional limit integer (passed in either as a pointer or a nil poitner).
func List(client *gophercloud.ServiceClient, loadBalancerID int, limit *int) pagination.Pager {
	url := rootURL(client, loadBalancerID)
	if limit != nil {
		url += fmt.Sprintf("?limit=%d", limit)
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return NodePage{pagination.SinglePageBase(r)}
	})
}

// CreateOptsBuilder is the interface responsible for generating the JSON
// for a Create operation.
type CreateOptsBuilder interface {
	ToNodeCreateMap() (map[string]interface{}, error)
}

// CreateOpts is a slice of CreateOpt structs, that allow the user to create
// multiple nodes in a single operation (one node per CreateOpt).
type CreateOpts []CreateOpt

// CreateOpt represents the options to create a single node.
type CreateOpt struct {
	// Required - the IP address or CIDR for this back-end node. It can either be
	// a private IP (ServiceNet) or a public IP.
	Address string

	// Optional - the port on which traffic is sent and received.
	Port int

	// Optional - the condition of the node. See the consts in Results.go.
	Condition Condition

	// Optional - the type of the node. See the consts in Results.go.
	Type Type

	// Optional - a pointer to an integer between 0 and 100.
	Weight *int
}

func validateWeight(weight *int) error {
	if weight != nil && (*weight > 100 || *weight < 0) {
		return errors.New("Weight must be a valid int between 0 and 100")
	}
	return nil
}

// ToNodeCreateMap converts a slice of options into a map that can be used for
// the JSON.
func (opts CreateOpts) ToNodeCreateMap() (map[string]interface{}, error) {
	type nodeMap map[string]interface{}
	nodes := []nodeMap{}

	for k, v := range opts {
		if v.Address == "" {
			return nodeMap{}, fmt.Errorf("ID is a required attribute, none provided for %d CreateOpt element", k)
		}
		if weightErr := validateWeight(v.Weight); weightErr != nil {
			return nodeMap{}, weightErr
		}

		node := make(map[string]interface{})
		node["address"] = v.Address

		if v.Port > 0 {
			node["port"] = v.Port
		}
		if v.Condition != "" {
			node["condition"] = v.Condition
		}
		if v.Type != "" {
			node["type"] = v.Type
		}
		if v.Weight != nil {
			node["weight"] = &v.Weight
		}

		nodes = append(nodes, node)
	}

	return nodeMap{"nodes": nodes}, nil
}

// Create is the operation responsible for creating a new node on a load
// balancer. Since every load balancer exists in both ServiceNet and the public
// Internet, both private and public IP addresses can be used for nodes.
//
// If nodes need time to boot up services before they become operational, you
// can temporarily prevent traffic from being sent to that node by setting the
// Condition field to DRAINING. Health checks will still be performed; but once
// your node is ready, you can update its condition to ENABLED and have it
// handle traffic.
func Create(client *gophercloud.ServiceClient, loadBalancerID int, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToNodeCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	resp, err := client.Post(rootURL(client, loadBalancerID), reqBody, &res.Body, nil)

	if err != nil {
		res.Err = err
		return res
	}

	pr := pagination.PageResultFromParsed(resp, res.Body)
	return CreateResult{pagination.SinglePageBase(pr)}
}

// BulkDelete is the operation responsible for batch deleting multiple nodes in
// a single operation. It accepts a slice of integer IDs and will remove them
// from the load balancer. The maximum limit is 10 node removals at once.
func BulkDelete(c *gophercloud.ServiceClient, loadBalancerID int, nodeIDs []int) DeleteResult {
	var res DeleteResult

	if len(nodeIDs) > 10 || len(nodeIDs) == 0 {
		res.Err = errors.New("You must provide a minimum of 1 and a maximum of 10 node IDs")
		return res
	}

	url := rootURL(c, loadBalancerID)
	url += gophercloud.IDSliceToQueryString("id", nodeIDs)

	_, res.Err = c.Delete(url, nil)
	return res
}

// Get is the operation responsible for showing details for a single node.
func Get(c *gophercloud.ServiceClient, lbID, nodeID int) GetResult {
	var res GetResult
	_, res.Err = c.Get(resourceURL(c, lbID, nodeID), &res.Body, nil)
	return res
}

// UpdateOptsBuilder represents a type that can be converted into a JSON-like
// map structure.
type UpdateOptsBuilder interface {
	ToNodeUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represent the options for updating an existing node.
type UpdateOpts struct {
	// Optional - the condition of the node. See the consts in Results.go.
	Condition Condition

	// Optional - the type of the node. See the consts in Results.go.
	Type Type

	// Optional - a pointer to an integer between 0 and 100.
	Weight *int
}

// ToNodeUpdateMap converts an options struct into a JSON-like map.
func (opts UpdateOpts) ToNodeUpdateMap() (map[string]interface{}, error) {
	node := make(map[string]interface{})

	if opts.Condition != "" {
		node["condition"] = opts.Condition
	}
	if opts.Weight != nil {
		if weightErr := validateWeight(opts.Weight); weightErr != nil {
			return node, weightErr
		}
		node["weight"] = &opts.Weight
	}
	if opts.Type != "" {
		node["type"] = opts.Type
	}

	return map[string]interface{}{"node": node}, nil
}

// Update is the operation responsible for updating an existing node. A node's
// IP, port, and status are immutable attributes and cannot be modified.
func Update(c *gophercloud.ServiceClient, lbID, nodeID int, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToNodeUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Put(resourceURL(c, lbID, nodeID), reqBody, nil, nil)
	return res
}

// Delete is the operation responsible for permanently deleting a node.
func Delete(c *gophercloud.ServiceClient, lbID, nodeID int) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, lbID, nodeID), nil)
	return res
}

// ListEventsOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListEventsOptsBuilder interface {
	ToEventsListQuery() (string, error)
}

// ListEventsOpts allows the filtering and sorting of paginated collections through
// the API.
type ListEventsOpts struct {
	Marker string `q:"marker"`
	Limit  int    `q:"limit"`
}

// ToEventsListQuery formats a ListOpts into a query string.
func (opts ListEventsOpts) ToEventsListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// ListEvents is the operation responsible for listing all the events
// associated with the activity between the node and the load balancer. The
// events report errors found with the node. The detailedMessage provides the
// detailed reason for the error.
func ListEvents(client *gophercloud.ServiceClient, loadBalancerID int, opts ListEventsOptsBuilder) pagination.Pager {
	url := eventsURL(client, loadBalancerID)

	if opts != nil {
		query, err := opts.ToEventsListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return NodeEventPage{pagination.SinglePageBase(r)}
	})
}
