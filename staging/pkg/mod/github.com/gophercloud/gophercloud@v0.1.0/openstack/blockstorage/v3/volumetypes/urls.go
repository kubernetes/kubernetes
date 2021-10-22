package volumetypes

import "github.com/gophercloud/gophercloud"

func listURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("types")
}

func getURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("types", id)
}

func createURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("types")
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("types", id)
}

func updateURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("types", id)
}
