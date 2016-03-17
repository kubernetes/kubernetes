package flavors

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List will list all available hardware flavors that an instance can use. The
// operation is identical to the one supported by the Nova API, but without the
// "disk" property.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	createPage := func(r pagination.PageResult) pagination.Page {
		return FlavorPage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, listURL(client), createPage)
}

// Get will retrieve information for a specified hardware flavor.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var gr GetResult

	_, gr.Err = client.Request("GET", getURL(client, id), gophercloud.RequestOpts{
		JSONResponse: &gr.Body,
		OkCodes:      []int{200},
	})

	return gr
}
