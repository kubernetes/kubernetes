package lbpools

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns all load balancer pools that are associated with RackConnect.
func List(c *gophercloud.ServiceClient) pagination.Pager {
	url := listURL(c)
	createPage := func(r pagination.PageResult) pagination.Page {
		return PoolPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(c, url, createPage)
}

// Get retrieves a specific load balancer pool (that is associated with RackConnect)
// based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(getURL(c, id), &res.Body, nil)
	return res
}

// ListNodes returns all load balancer pool nodes that are associated with RackConnect
// for the given LB pool ID.
func ListNodes(c *gophercloud.ServiceClient, id string) pagination.Pager {
	url := listNodesURL(c, id)
	createPage := func(r pagination.PageResult) pagination.Page {
		return NodePage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(c, url, createPage)
}

// CreateNode adds the cloud server with the given serverID to the load balancer
// pool with the given poolID.
func CreateNode(c *gophercloud.ServiceClient, poolID, serverID string) CreateNodeResult {
	var res CreateNodeResult
	reqBody := map[string]interface{}{
		"cloud_server": map[string]string{
			"id": serverID,
		},
	}
	_, res.Err = c.Post(createNodeURL(c, poolID), reqBody, &res.Body, nil)
	return res
}

// ListNodesDetails returns all load balancer pool nodes that are associated with RackConnect
// for the given LB pool ID with all their details.
func ListNodesDetails(c *gophercloud.ServiceClient, id string) pagination.Pager {
	url := listNodesDetailsURL(c, id)
	createPage := func(r pagination.PageResult) pagination.Page {
		return NodeDetailsPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(c, url, createPage)
}

// GetNode retrieves a specific LB pool node (that is associated with RackConnect)
// based on its unique ID and the LB pool's unique ID.
func GetNode(c *gophercloud.ServiceClient, poolID, nodeID string) GetNodeResult {
	var res GetNodeResult
	_, res.Err = c.Get(nodeURL(c, poolID, nodeID), &res.Body, nil)
	return res
}

// DeleteNode removes the node with the given nodeID from the LB pool with the
// given poolID.
func DeleteNode(c *gophercloud.ServiceClient, poolID, nodeID string) DeleteNodeResult {
	var res DeleteNodeResult
	_, res.Err = c.Delete(deleteNodeURL(c, poolID, nodeID), nil)
	return res
}

// GetNodeDetails retrieves a specific LB pool node's details based on its unique
// ID and the LB pool's unique ID.
func GetNodeDetails(c *gophercloud.ServiceClient, poolID, nodeID string) GetNodeDetailsResult {
	var res GetNodeDetailsResult
	_, res.Err = c.Get(nodeDetailsURL(c, poolID, nodeID), &res.Body, nil)
	return res
}

// NodeOpts are options for bulk adding/deleting nodes to LB pools.
type NodeOpts struct {
	ServerID string
	PoolID   string
}

// NodesOpts are a slice of NodeOpts, passed as options for bulk operations.
type NodesOpts []NodeOpts

// ToLBPoolCreateNodesMap serializes a NodesOpts into a map to send in the request.
func (o NodesOpts) ToLBPoolCreateNodesMap() ([]map[string]interface{}, error) {
	m := make([]map[string]interface{}, len(o))
	for i := range o {
		m[i] = map[string]interface{}{
			"cloud_server": map[string]string{
				"id": o[i].ServerID,
			},
			"load_balancer_pool": map[string]string{
				"id": o[i].PoolID,
			},
		}
	}
	return m, nil
}

// CreateNodes adds the cloud servers with the given serverIDs to the corresponding
// load balancer pools with the given poolIDs.
func CreateNodes(c *gophercloud.ServiceClient, opts NodesOpts) CreateNodesResult {
	var res CreateNodesResult
	reqBody, err := opts.ToLBPoolCreateNodesMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(createNodesURL(c), reqBody, &res.Body, nil)
	return res
}

// DeleteNodes removes the cloud servers with the given serverIDs to the corresponding
// load balancer pools with the given poolIDs.
func DeleteNodes(c *gophercloud.ServiceClient, opts NodesOpts) DeleteNodesResult {
	var res DeleteNodesResult
	reqBody, err := opts.ToLBPoolCreateNodesMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Request("DELETE", createNodesURL(c), gophercloud.RequestOpts{
		JSONBody: &reqBody,
		OkCodes:  []int{204},
	})
	return res
}

// ListNodesDetailsForServer is similar to ListNodesDetails but only returns nodes
// for the given serverID.
func ListNodesDetailsForServer(c *gophercloud.ServiceClient, serverID string) pagination.Pager {
	url := listNodesForServerURL(c, serverID)
	createPage := func(r pagination.PageResult) pagination.Page {
		return NodeDetailsForServerPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(c, url, createPage)
}
