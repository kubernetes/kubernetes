package networkipavailabilities

import "github.com/gophercloud/gophercloud"

const resourcePath = "network-ip-availabilities"

func rootURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(resourcePath)
}

func resourceURL(c *gophercloud.ServiceClient, networkIPAvailabilityID string) string {
	return c.ServiceURL(resourcePath, networkIPAvailabilityID)
}

func listURL(c *gophercloud.ServiceClient) string {
	return rootURL(c)
}

func getURL(c *gophercloud.ServiceClient, networkIPAvailabilityID string) string {
	return resourceURL(c, networkIPAvailabilityID)
}
