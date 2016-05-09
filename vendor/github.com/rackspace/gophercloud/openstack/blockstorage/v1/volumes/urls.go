package volumes

import "github.com/rackspace/gophercloud"

func createURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("volumes")
}

func listURL(c *gophercloud.ServiceClient) string {
	return createURL(c)
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("volumes", id)
}

func getURL(c *gophercloud.ServiceClient, id string) string {
	return deleteURL(c, id)
}

func updateURL(c *gophercloud.ServiceClient, id string) string {
	return deleteURL(c, id)
}
