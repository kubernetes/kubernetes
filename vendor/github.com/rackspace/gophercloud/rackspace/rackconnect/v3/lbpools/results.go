package lbpools

import (
	"fmt"
	"reflect"
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Pool represents a load balancer pool associated with a RackConnect configuration.
type Pool struct {
	// The unique ID of the load balancer pool.
	ID string `mapstructure:"id"`
	// The name of the load balancer pool.
	Name string `mapstructure:"name"`
	// The node counts associated witht the load balancer pool.
	NodeCounts struct {
		// The number of nodes associated with this LB pool for this account.
		CloudServers int `mapstructure:"cloud_servers"`
		// The number of nodes associated with this LB pool from other accounts.
		External int `mapstructure:"external"`
		// The total number of nodes associated with this LB pool.
		Total int `mapstructure:"total"`
	} `mapstructure:"node_counts"`
	// The port of the LB pool
	Port int `mapstructure:"port"`
	// The status of the LB pool
	Status string `mapstructure:"status"`
	// The details of the status of the LB pool
	StatusDetail string `mapstructure:"status_detail"`
	// The virtual IP of the LB pool
	VirtualIP string `mapstructure:"virtual_ip"`
}

// PoolPage is the page returned by a pager when traversing over a
// collection of Pools.
type PoolPage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a PoolPage contains no Pools.
func (r PoolPage) IsEmpty() (bool, error) {
	cns, err := ExtractPools(r)
	if err != nil {
		return true, err
	}
	return len(cns) == 0, nil
}

// ExtractPools extracts and returns Pools. It is used while iterating over
// an lbpools.List call.
func ExtractPools(page pagination.Page) ([]Pool, error) {
	var res []Pool
	err := mapstructure.Decode(page.(PoolPage).Body, &res)
	return res, err
}

// GetResult represents the result of a Get operation.
type GetResult struct {
	gophercloud.Result
}

// Extract is a function that extracts an LBPool from a GetResult.
func (r GetResult) Extract() (*Pool, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	var res Pool
	err := mapstructure.Decode(r.Body, &res)
	return &res, err
}

// Node represents a load balancer pool node associated with a RackConnect configuration.
type Node struct {
	// The unique ID of the LB node.
	ID string `mapstructure:"id"`
	// The cloud server (node) of the load balancer pool.
	CloudServer struct {
		// The cloud server ID.
		ID string `mapstructure:"id"`
	} `mapstructure:"cloud_server"`
	// The load balancer pool.
	LoadBalancerPool struct {
		// The LB pool ID.
		ID string `mapstructure:"id"`
	} `mapstructure:"load_balancer_pool"`
	// The status of the LB pool.
	Status string `mapstructure:"status"`
	// The details of the status of the LB pool.
	StatusDetail string `mapstructure:"status_detail"`
	// The time the LB node was created.
	CreatedAt time.Time `mapstructure:"-"`
	// The time the LB node was last updated.
	UpdatedAt time.Time `mapstructure:"-"`
}

// NodePage is the page returned by a pager when traversing over a
// collection of Nodes.
type NodePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a NodePage contains no Nodes.
func (r NodePage) IsEmpty() (bool, error) {
	n, err := ExtractNodes(r)
	if err != nil {
		return true, err
	}
	return len(n) == 0, nil
}

// ExtractNodes extracts and returns a slice of Nodes. It is used while iterating over
// an lbpools.ListNodes call.
func ExtractNodes(page pagination.Page) ([]Node, error) {
	var res []Node
	casted := page.(NodePage).Body
	err := mapstructure.Decode(casted, &res)

	var rawNodes []interface{}
	switch casted.(type) {
	case interface{}:
		rawNodes = casted.([]interface{})
	default:
		return res, fmt.Errorf("Unknown type: %v", reflect.TypeOf(casted))
	}

	for i := range rawNodes {
		thisNode := (rawNodes[i]).(map[string]interface{})

		if t, ok := thisNode["created"].(string); ok && t != "" {
			creationTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return res, err
			}
			res[i].CreatedAt = creationTime
		}

		if t, ok := thisNode["updated"].(string); ok && t != "" {
			updatedTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return res, err
			}
			res[i].UpdatedAt = updatedTime
		}
	}

	return res, err
}

// NodeResult represents a result that can be extracted as a Node.
type NodeResult struct {
	gophercloud.Result
}

// CreateNodeResult represents the result of an CreateNode operation.
type CreateNodeResult struct {
	NodeResult
}

// GetNodeResult represents the result of an GetNode operation.
type GetNodeResult struct {
	NodeResult
}

// Extract is a function that extracts a Node from a NodeResult.
func (r NodeResult) Extract() (*Node, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	var res Node
	err := mapstructure.Decode(r.Body, &res)

	b := r.Body.(map[string]interface{})

	if date, ok := b["created"]; ok && date != nil {
		t, err := time.Parse(time.RFC3339, date.(string))
		if err != nil {
			return nil, err
		}
		res.CreatedAt = t
	}

	if date, ok := b["updated"]; ok && date != nil {
		t, err := time.Parse(time.RFC3339, date.(string))
		if err != nil {
			return nil, err
		}
		res.UpdatedAt = t
	}

	return &res, err
}

// NodeDetails represents a load balancer pool node associated with a RackConnect configuration
// with all its details.
type NodeDetails struct {
	// The unique ID of the LB node.
	ID string `mapstructure:"id"`
	// The cloud server (node) of the load balancer pool.
	CloudServer struct {
		// The cloud server ID.
		ID string `mapstructure:"id"`
		// The name of the server.
		Name string `mapstructure:"name"`
		// The cloud network for the cloud server.
		CloudNetwork struct {
			// The network ID.
			ID string `mapstructure:"id"`
			// The network name.
			Name string `mapstructure:"name"`
			// The network's private IPv4 address.
			PrivateIPv4 string `mapstructure:"private_ip_v4"`
			// The IP range for the network.
			CIDR string `mapstructure:"cidr"`
			// The datetime the network was created.
			CreatedAt time.Time `mapstructure:"-"`
			// The last datetime the network was updated.
			UpdatedAt time.Time `mapstructure:"-"`
		} `mapstructure:"cloud_network"`
		// The datetime the server was created.
		CreatedAt time.Time `mapstructure:"-"`
		// The datetime the server was last updated.
		UpdatedAt time.Time `mapstructure:"-"`
	} `mapstructure:"cloud_server"`
	// The load balancer pool.
	LoadBalancerPool Pool `mapstructure:"load_balancer_pool"`
	// The status of the LB pool.
	Status string `mapstructure:"status"`
	// The details of the status of the LB pool.
	StatusDetail string `mapstructure:"status_detail"`
	// The time the LB node was created.
	CreatedAt time.Time `mapstructure:"-"`
	// The time the LB node was last updated.
	UpdatedAt time.Time `mapstructure:"-"`
}

// NodeDetailsPage is the page returned by a pager when traversing over a
// collection of NodeDetails.
type NodeDetailsPage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a NodeDetailsPage contains no NodeDetails.
func (r NodeDetailsPage) IsEmpty() (bool, error) {
	n, err := ExtractNodesDetails(r)
	if err != nil {
		return true, err
	}
	return len(n) == 0, nil
}

// ExtractNodesDetails extracts and returns a slice of NodeDetails. It is used while iterating over
// an lbpools.ListNodesDetails call.
func ExtractNodesDetails(page pagination.Page) ([]NodeDetails, error) {
	var res []NodeDetails
	casted := page.(NodeDetailsPage).Body
	err := mapstructure.Decode(casted, &res)

	var rawNodesDetails []interface{}
	switch casted.(type) {
	case interface{}:
		rawNodesDetails = casted.([]interface{})
	default:
		return res, fmt.Errorf("Unknown type: %v", reflect.TypeOf(casted))
	}

	for i := range rawNodesDetails {
		thisNodeDetails := (rawNodesDetails[i]).(map[string]interface{})

		if t, ok := thisNodeDetails["created"].(string); ok && t != "" {
			creationTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return res, err
			}
			res[i].CreatedAt = creationTime
		}

		if t, ok := thisNodeDetails["updated"].(string); ok && t != "" {
			updatedTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return res, err
			}
			res[i].UpdatedAt = updatedTime
		}

		if cs, ok := thisNodeDetails["cloud_server"].(map[string]interface{}); ok {
			if t, ok := cs["created"].(string); ok && t != "" {
				creationTime, err := time.Parse(time.RFC3339, t)
				if err != nil {
					return res, err
				}
				res[i].CloudServer.CreatedAt = creationTime
			}
			if t, ok := cs["updated"].(string); ok && t != "" {
				updatedTime, err := time.Parse(time.RFC3339, t)
				if err != nil {
					return res, err
				}
				res[i].CloudServer.UpdatedAt = updatedTime
			}
			if cn, ok := cs["cloud_network"].(map[string]interface{}); ok {
				if t, ok := cn["created"].(string); ok && t != "" {
					creationTime, err := time.Parse(time.RFC3339, t)
					if err != nil {
						return res, err
					}
					res[i].CloudServer.CloudNetwork.CreatedAt = creationTime
				}
				if t, ok := cn["updated"].(string); ok && t != "" {
					updatedTime, err := time.Parse(time.RFC3339, t)
					if err != nil {
						return res, err
					}
					res[i].CloudServer.CloudNetwork.UpdatedAt = updatedTime
				}
			}
		}
	}

	return res, err
}

// GetNodeDetailsResult represents the result of an NodeDetails operation.
type GetNodeDetailsResult struct {
	gophercloud.Result
}

// Extract is a function that extracts a NodeDetails from a NodeDetailsResult.
func (r GetNodeDetailsResult) Extract() (*NodeDetails, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	var res NodeDetails
	err := mapstructure.Decode(r.Body, &res)

	b := r.Body.(map[string]interface{})

	if date, ok := b["created"]; ok && date != nil {
		t, err := time.Parse(time.RFC3339, date.(string))
		if err != nil {
			return nil, err
		}
		res.CreatedAt = t
	}

	if date, ok := b["updated"]; ok && date != nil {
		t, err := time.Parse(time.RFC3339, date.(string))
		if err != nil {
			return nil, err
		}
		res.UpdatedAt = t
	}

	if cs, ok := b["cloud_server"].(map[string]interface{}); ok {
		if t, ok := cs["created"].(string); ok && t != "" {
			creationTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return &res, err
			}
			res.CloudServer.CreatedAt = creationTime
		}
		if t, ok := cs["updated"].(string); ok && t != "" {
			updatedTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return &res, err
			}
			res.CloudServer.UpdatedAt = updatedTime
		}
		if cn, ok := cs["cloud_network"].(map[string]interface{}); ok {
			if t, ok := cn["created"].(string); ok && t != "" {
				creationTime, err := time.Parse(time.RFC3339, t)
				if err != nil {
					return &res, err
				}
				res.CloudServer.CloudNetwork.CreatedAt = creationTime
			}
			if t, ok := cn["updated"].(string); ok && t != "" {
				updatedTime, err := time.Parse(time.RFC3339, t)
				if err != nil {
					return &res, err
				}
				res.CloudServer.CloudNetwork.UpdatedAt = updatedTime
			}
		}
	}

	return &res, err
}

// DeleteNodeResult represents the result of a DeleteNode operation.
type DeleteNodeResult struct {
	gophercloud.ErrResult
}

// CreateNodesResult represents the result of a CreateNodes operation.
type CreateNodesResult struct {
	gophercloud.Result
}

// Extract is a function that extracts a slice of Nodes from a CreateNodesResult.
func (r CreateNodesResult) Extract() ([]Node, error) {
	if r.Err != nil {
		return nil, r.Err
	}
	var res []Node
	err := mapstructure.Decode(r.Body, &res)

	b := r.Body.([]interface{})
	for i := range b {
		if date, ok := b[i].(map[string]interface{})["created"]; ok && date != nil {
			t, err := time.Parse(time.RFC3339, date.(string))
			if err != nil {
				return nil, err
			}
			res[i].CreatedAt = t
		}
		if date, ok := b[i].(map[string]interface{})["updated"]; ok && date != nil {
			t, err := time.Parse(time.RFC3339, date.(string))
			if err != nil {
				return nil, err
			}
			res[i].UpdatedAt = t
		}
	}

	return res, err
}

// DeleteNodesResult represents the result of a DeleteNodes operation.
type DeleteNodesResult struct {
	gophercloud.ErrResult
}

// NodeDetailsForServer represents a load balancer pool node associated with a RackConnect configuration
// with all its details for a particular server.
type NodeDetailsForServer struct {
	// The unique ID of the LB node.
	ID string `mapstructure:"id"`
	// The load balancer pool.
	LoadBalancerPool Pool `mapstructure:"load_balancer_pool"`
	// The status of the LB pool.
	Status string `mapstructure:"status"`
	// The details of the status of the LB pool.
	StatusDetail string `mapstructure:"status_detail"`
	// The time the LB node was created.
	CreatedAt time.Time `mapstructure:"-"`
	// The time the LB node was last updated.
	UpdatedAt time.Time `mapstructure:"-"`
}

// NodeDetailsForServerPage is the page returned by a pager when traversing over a
// collection of NodeDetailsForServer.
type NodeDetailsForServerPage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a NodeDetailsForServerPage contains no NodeDetailsForServer.
func (r NodeDetailsForServerPage) IsEmpty() (bool, error) {
	n, err := ExtractNodesDetailsForServer(r)
	if err != nil {
		return true, err
	}
	return len(n) == 0, nil
}

// ExtractNodesDetailsForServer extracts and returns a slice of NodeDetailsForServer. It is used while iterating over
// an lbpools.ListNodesDetailsForServer call.
func ExtractNodesDetailsForServer(page pagination.Page) ([]NodeDetailsForServer, error) {
	var res []NodeDetailsForServer
	casted := page.(NodeDetailsForServerPage).Body
	err := mapstructure.Decode(casted, &res)

	var rawNodesDetails []interface{}
	switch casted.(type) {
	case interface{}:
		rawNodesDetails = casted.([]interface{})
	default:
		return res, fmt.Errorf("Unknown type: %v", reflect.TypeOf(casted))
	}

	for i := range rawNodesDetails {
		thisNodeDetails := (rawNodesDetails[i]).(map[string]interface{})

		if t, ok := thisNodeDetails["created"].(string); ok && t != "" {
			creationTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return res, err
			}
			res[i].CreatedAt = creationTime
		}

		if t, ok := thisNodeDetails["updated"].(string); ok && t != "" {
			updatedTime, err := time.Parse(time.RFC3339, t)
			if err != nil {
				return res, err
			}
			res[i].UpdatedAt = updatedTime
		}
	}

	return res, err
}
