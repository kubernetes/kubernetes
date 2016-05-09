package networks

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager that allows you to iterate over a collection of Network.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	url := listURL(client)
	createPage := func(r pagination.PageResult) pagination.Page {
		return NetworkPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(client, url, createPage)
}

// Get returns data about a previously created Network.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = client.Get(getURL(client, id), &res.Body, nil)
	return res
}
