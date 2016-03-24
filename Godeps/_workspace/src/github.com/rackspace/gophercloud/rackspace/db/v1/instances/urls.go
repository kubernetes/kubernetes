package instances

import "github.com/rackspace/gophercloud"

func baseURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("instances")
}

func createURL(c *gophercloud.ServiceClient) string {
	return baseURL(c)
}

func resourceURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("instances", id)
}

func configURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("instances", id, "configuration")
}

func backupsURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("instances", id, "backups")
}
