package volumeattach

import "github.com/rackspace/gophercloud"

const resourcePath = "os-volume_attachments"

func resourceURL(c *gophercloud.ServiceClient, serverId string) string {
	return c.ServiceURL("servers", serverId, resourcePath)
}

func listURL(c *gophercloud.ServiceClient, serverId string) string {
	return resourceURL(c, serverId)
}

func createURL(c *gophercloud.ServiceClient, serverId string) string {
	return resourceURL(c, serverId)
}

func getURL(c *gophercloud.ServiceClient, serverId, aId string) string {
	return c.ServiceURL("servers", serverId, resourcePath, aId)
}

func deleteURL(c *gophercloud.ServiceClient, serverId, aId string) string {
	return getURL(c, serverId, aId)
}
