package buildinfo

import "github.com/gophercloud/gophercloud"

// Get retreives data for the given stack template.
func Get(c *gophercloud.ServiceClient) (r GetResult) {
	_, r.Err = c.Get(getURL(c), &r.Body, nil)
	return
}
