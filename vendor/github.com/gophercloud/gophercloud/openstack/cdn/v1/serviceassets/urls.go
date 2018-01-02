package serviceassets

import "github.com/gophercloud/gophercloud"

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("services", id, "assets")
}
