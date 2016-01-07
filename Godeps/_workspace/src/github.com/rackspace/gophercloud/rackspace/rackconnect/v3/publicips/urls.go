package publicips

import "github.com/rackspace/gophercloud"

var root = "public_ips"

func listURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(root)
}

func createURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(root)
}

func listForServerURL(c *gophercloud.ServiceClient, serverID string) string {
	return c.ServiceURL(root + "?cloud_server_id=" + serverID)
}

func getURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(root, id)
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return getURL(c, id)
}
