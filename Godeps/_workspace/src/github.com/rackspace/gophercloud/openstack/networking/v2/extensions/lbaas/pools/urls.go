package pools

import "github.com/rackspace/gophercloud"

const (
	rootPath     = "lb"
	resourcePath = "pools"
	monitorPath  = "health_monitors"
)

func rootURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(rootPath, resourcePath)
}

func resourceURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(rootPath, resourcePath, id)
}

func associateURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(rootPath, resourcePath, id, monitorPath)
}

func disassociateURL(c *gophercloud.ServiceClient, poolID, monitorID string) string {
	return c.ServiceURL(rootPath, resourcePath, poolID, monitorPath, monitorID)
}
