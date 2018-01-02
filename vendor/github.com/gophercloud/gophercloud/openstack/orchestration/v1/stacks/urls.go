package stacks

import "github.com/gophercloud/gophercloud"

func createURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("stacks")
}

func adoptURL(c *gophercloud.ServiceClient) string {
	return createURL(c)
}

func listURL(c *gophercloud.ServiceClient) string {
	return createURL(c)
}

func getURL(c *gophercloud.ServiceClient, name, id string) string {
	return c.ServiceURL("stacks", name, id)
}

func updateURL(c *gophercloud.ServiceClient, name, id string) string {
	return getURL(c, name, id)
}

func deleteURL(c *gophercloud.ServiceClient, name, id string) string {
	return getURL(c, name, id)
}

func previewURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("stacks", "preview")
}

func abandonURL(c *gophercloud.ServiceClient, name, id string) string {
	return c.ServiceURL("stacks", name, id, "abandon")
}
