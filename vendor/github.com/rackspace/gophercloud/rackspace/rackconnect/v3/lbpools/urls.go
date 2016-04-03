package lbpools

import "github.com/rackspace/gophercloud"

var root = "load_balancer_pools"

func listURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(root)
}

func getURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(root, id)
}

func listNodesURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(root, id, "nodes")
}

func createNodeURL(c *gophercloud.ServiceClient, id string) string {
	return listNodesURL(c, id)
}

func listNodesDetailsURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(root, id, "nodes", "details")
}

func nodeURL(c *gophercloud.ServiceClient, poolID, nodeID string) string {
	return c.ServiceURL(root, poolID, "nodes", nodeID)
}

func deleteNodeURL(c *gophercloud.ServiceClient, poolID, nodeID string) string {
	return nodeURL(c, poolID, nodeID)
}

func nodeDetailsURL(c *gophercloud.ServiceClient, poolID, nodeID string) string {
	return c.ServiceURL(root, poolID, "nodes", nodeID, "details")
}

func createNodesURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(root, "nodes")
}

func deleteNodesURL(c *gophercloud.ServiceClient) string {
	return createNodesURL(c)
}

func listNodesForServerURL(c *gophercloud.ServiceClient, serverID string) string {
	return c.ServiceURL(root, "nodes", "details?cloud_server_id="+serverID)
}
