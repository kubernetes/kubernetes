package cloudnetworks

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns all cloud networks that are associated with RackConnect. The ID
// returned for each network is the same as the ID returned by the networks package.
func List(c *gophercloud.ServiceClient) pagination.Pager {
	url := listURL(c)
	createPage := func(r pagination.PageResult) pagination.Page {
		return CloudNetworkPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(c, url, createPage)
}

// Get retrieves a specific cloud network (that is associated with RackConnect)
// based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(getURL(c, id), &res.Body, nil)
	return res
}
