package cdncontainers

import "github.com/rackspace/gophercloud"

func enableURL(c *gophercloud.ServiceClient, containerName string) string {
	return c.ServiceURL(containerName)
}
