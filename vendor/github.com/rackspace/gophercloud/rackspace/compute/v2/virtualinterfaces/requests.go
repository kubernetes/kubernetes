package virtualinterfaces

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager which allows you to iterate over a collection of
// networks. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
func List(c *gophercloud.ServiceClient, instanceID string) pagination.Pager {
	createPage := func(r pagination.PageResult) pagination.Page {
		return VirtualInterfacePage{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(c, listURL(c, instanceID), createPage)
}

// Create creates a new virtual interface for a network and attaches the network
// to the server instance.
func Create(c *gophercloud.ServiceClient, instanceID, networkID string) CreateResult {
	var res CreateResult

	reqBody := map[string]map[string]string{
		"virtual_interface": {
			"network_id": networkID,
		},
	}

	// Send request to API
	_, res.Err = c.Post(createURL(c, instanceID), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return res
}

// Delete deletes the interface with interfaceID attached to the instance with
// instanceID.
func Delete(c *gophercloud.ServiceClient, instanceID, interfaceID string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(deleteURL(c, instanceID, interfaceID), &gophercloud.RequestOpts{
		OkCodes: []int{200, 204},
	})
	return res
}
