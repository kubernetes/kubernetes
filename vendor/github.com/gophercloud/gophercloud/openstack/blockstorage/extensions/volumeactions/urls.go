package volumeactions

import "github.com/gophercloud/gophercloud"

func attachURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL("volumes", id, "action")
}

func beginDetachingURL(c *gophercloud.ServiceClient, id string) string {
	return attachURL(c, id)
}

func detachURL(c *gophercloud.ServiceClient, id string) string {
	return attachURL(c, id)
}

func reserveURL(c *gophercloud.ServiceClient, id string) string {
	return attachURL(c, id)
}

func unreserveURL(c *gophercloud.ServiceClient, id string) string {
	return attachURL(c, id)
}

func initializeConnectionURL(c *gophercloud.ServiceClient, id string) string {
	return attachURL(c, id)
}

func teminateConnectionURL(c *gophercloud.ServiceClient, id string) string {
	return attachURL(c, id)
}

func extendSizeURL(c *gophercloud.ServiceClient, id string) string {
	return attachURL(c, id)
}
