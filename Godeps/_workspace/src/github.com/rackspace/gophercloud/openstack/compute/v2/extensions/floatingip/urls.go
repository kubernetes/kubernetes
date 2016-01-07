package floatingip

import "github.com/rackspace/gophercloud"

const resourcePath = "os-floating-ips"

func resourceURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(resourcePath)
}

func listURL(c *gophercloud.ServiceClient) string {
	return resourceURL(c)
}

func createURL(c *gophercloud.ServiceClient) string {
	return resourceURL(c)
}

func getURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(resourcePath, id)
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return getURL(c, id)
}

func serverURL(c *gophercloud.ServiceClient, serverId string) string {
	return c.ServiceURL("servers/" + serverId + "/action")
}

func associateURL(c *gophercloud.ServiceClient, serverId string) string {
	return serverURL(c, serverId)
}

func disassociateURL(c *gophercloud.ServiceClient, serverId string) string {
	return serverURL(c, serverId)
}
