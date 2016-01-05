package apiversions

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List lists all the Cinder API versions available to end-users.
func List(c *gophercloud.ServiceClient) pagination.Pager {
	return pagination.NewPager(c, listURL(c), func(r pagination.PageResult) pagination.Page {
		return APIVersionPage{pagination.SinglePageBase(r)}
	})
}

// Get will retrieve the volume type with the provided ID. To extract the volume
// type from the result, call the Extract method on the GetResult.
func Get(client *gophercloud.ServiceClient, v string) GetResult {
	var res GetResult
	_, res.Err = client.Get(getURL(client, v), &res.Body, nil)
	return res
}
