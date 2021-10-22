package rbacpolicies

import "github.com/gophercloud/gophercloud"

func resourceURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("rbac-policies", id)
}

func rootURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("rbac-policies")
}

func createURL(c *gophercloud.ServiceClient) string {
	return rootURL(c)
}

func listURL(c *gophercloud.ServiceClient) string {
	return rootURL(c)
}

func getURL(c *gophercloud.ServiceClient, id string) string {
	return resourceURL(c, id)
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return resourceURL(c, id)
}

func updateURL(c *gophercloud.ServiceClient, id string) string {
	return resourceURL(c, id)
}
