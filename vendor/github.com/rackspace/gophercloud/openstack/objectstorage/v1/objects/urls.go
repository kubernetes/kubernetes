package objects

import (
	"github.com/rackspace/gophercloud"
)

func listURL(c *gophercloud.ServiceClient, container string) string {
	return c.ServiceURL(container)
}

func copyURL(c *gophercloud.ServiceClient, container, object string) string {
	return c.ServiceURL(container, object)
}

func createURL(c *gophercloud.ServiceClient, container, object string) string {
	return copyURL(c, container, object)
}

func getURL(c *gophercloud.ServiceClient, container, object string) string {
	return copyURL(c, container, object)
}

func deleteURL(c *gophercloud.ServiceClient, container, object string) string {
	return copyURL(c, container, object)
}

func downloadURL(c *gophercloud.ServiceClient, container, object string) string {
	return copyURL(c, container, object)
}

func updateURL(c *gophercloud.ServiceClient, container, object string) string {
	return copyURL(c, container, object)
}
