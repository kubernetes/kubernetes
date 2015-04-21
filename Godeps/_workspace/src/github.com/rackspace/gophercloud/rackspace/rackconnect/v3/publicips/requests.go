package publicips

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns all public IPs.
func List(c *gophercloud.ServiceClient) pagination.Pager {
	url := listURL(c)
	createPage := func(r pagination.PageResult) pagination.Page {
		return PublicIPPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(c, url, createPage)
}

// Create adds a public IP to the server with the given serverID.
func Create(c *gophercloud.ServiceClient, serverID string) CreateResult {
	var res CreateResult
	reqBody := map[string]interface{}{
		"cloud_server": map[string]string{
			"id": serverID,
		},
	}
	_, res.Err = c.Post(createURL(c), reqBody, &res.Body, nil)
	return res
}

// ListForServer returns all public IPs for the server with the given serverID.
func ListForServer(c *gophercloud.ServiceClient, serverID string) pagination.Pager {
	url := listForServerURL(c, serverID)
	createPage := func(r pagination.PageResult) pagination.Page {
		return PublicIPPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(c, url, createPage)
}

// Get retrieves the public IP with the given id.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(getURL(c, id), &res.Body, nil)
	return res
}

// Delete removes the public IP with the given id.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(deleteURL(c, id), nil)
	return res
}
