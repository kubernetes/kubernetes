package apiversions

import "github.com/gophercloud/gophercloud"

func apiVersionsURL(c *gophercloud.ServiceClient) string {
	return c.Endpoint
}
