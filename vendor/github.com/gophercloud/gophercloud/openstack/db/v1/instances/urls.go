package instances

import "github.com/gophercloud/gophercloud"

func baseURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("instances")
}

func resourceURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("instances", id)
}

func userRootURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("instances", id, "root")
}

func actionURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("instances", id, "action")
}
