package extensions

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Get retrieves information for a specific extension using its alias.
func Get(c *gophercloud.ServiceClient, alias string) GetResult {
	var res GetResult
	_, res.Err = c.Request("GET", ExtensionURL(c, alias), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})
	return res
}

// List returns a Pager which allows you to iterate over the full collection of extensions.
// It does not accept query parameters.
func List(c *gophercloud.ServiceClient) pagination.Pager {
	return pagination.NewPager(c, ListExtensionURL(c), func(r pagination.PageResult) pagination.Page {
		return ExtensionPage{pagination.SinglePageBase(r)}
	})
}
