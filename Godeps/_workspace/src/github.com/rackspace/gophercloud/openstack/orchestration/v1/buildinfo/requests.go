package buildinfo

import "github.com/rackspace/gophercloud"

// Get retreives data for the given stack template.
func Get(c *gophercloud.ServiceClient) GetResult {
	var res GetResult
	_, res.Err = c.Request("GET", getURL(c), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})
	return res
}
