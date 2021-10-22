package backups

import "github.com/gophercloud/gophercloud"

func createURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("backups")
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("backups", id)
}

func getURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("backups", id)
}

func listURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("backups")
}

func updateURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("backups", id)
}
