package loadbalancers

import "github.com/gophercloud/gophercloud"

const (
	rootPath       = "lbaas"
	resourcePath   = "loadbalancers"
	statusPath     = "status"
	statisticsPath = "stats"
	failoverPath   = "failover"
)

func rootURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(rootPath, resourcePath)
}

func resourceURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(rootPath, resourcePath, id)
}

func statusRootURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(rootPath, resourcePath, id, statusPath)
}

func statisticsRootURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(rootPath, resourcePath, id, statisticsPath)
}

func failoverRootURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(rootPath, resourcePath, id, failoverPath)
}
