package shares

import "github.com/gophercloud/gophercloud"

func createURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("shares")
}

func listDetailURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("shares", "detail")
}

func deleteURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id)
}

func getURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id)
}

func updateURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id)
}

func getExportLocationsURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id, "export_locations")
}

func grantAccessURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id, "action")
}

func revokeAccessURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id, "action")
}

func listAccessRightsURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id, "action")
}

func extendURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id, "action")
}

func shrinkURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("shares", id, "action")
}
