package hypervisors

import "github.com/gophercloud/gophercloud"

func hypervisorsListDetailURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("os-hypervisors", "detail")
}
