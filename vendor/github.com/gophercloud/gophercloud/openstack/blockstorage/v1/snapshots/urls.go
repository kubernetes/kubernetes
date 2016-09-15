package snapshots

import "github.com/gophercloud/gophercloud"

func createURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("snapshots")
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("snapshots", id)
}

func getURL(c *gophercloud.ServiceClient, id string) string {
	return deleteURL(c, id)
}

func listURL(c *gophercloud.ServiceClient) string {
	return createURL(c)
}

func metadataURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("snapshots", id, "metadata")
}

func updateMetadataURL(c *gophercloud.ServiceClient, id string) string {
	return metadataURL(c, id)
}
