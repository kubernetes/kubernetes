package ikepolicies

import "github.com/gophercloud/gophercloud"

const (
	rootPath     = "vpn"
	resourcePath = "ikepolicies"
)

func rootURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(rootPath, resourcePath)
}

func resourceURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(rootPath, resourcePath, id)
}
