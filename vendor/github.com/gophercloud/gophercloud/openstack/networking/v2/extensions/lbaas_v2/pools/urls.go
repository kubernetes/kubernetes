package pools

import "github.com/gophercloud/gophercloud"

const (
	rootPath     = "lbaas"
	resourcePath = "pools"
	memberPath   = "members"
)

func rootURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(rootPath, resourcePath)
}

func resourceURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(rootPath, resourcePath, id)
}

func memberRootURL(c *gophercloud.ServiceClient, poolId string) string {
	return c.ServiceURL(rootPath, resourcePath, poolId, memberPath)
}

func memberResourceURL(c *gophercloud.ServiceClient, poolID string, memberID string) string {
	return c.ServiceURL(rootPath, resourcePath, poolID, memberPath, memberID)
}
