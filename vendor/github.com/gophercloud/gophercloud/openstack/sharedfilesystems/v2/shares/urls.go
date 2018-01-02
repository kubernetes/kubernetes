package shares

import "github.com/gophercloud/gophercloud"

func createURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("shares")
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id)
}

func getURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id)
}

func getExportLocationsURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id, "export_locations")
}

func grantAccessURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id, "action")
}
