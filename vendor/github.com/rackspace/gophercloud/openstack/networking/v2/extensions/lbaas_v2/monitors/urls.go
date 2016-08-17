package monitors

import "github.com/rackspace/gophercloud"

const (
	rootPath     = "lbaas"
	resourcePath = "healthmonitors"
)

func rootURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(rootPath, resourcePath)
}

func resourceURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(rootPath, resourcePath, id)
}
