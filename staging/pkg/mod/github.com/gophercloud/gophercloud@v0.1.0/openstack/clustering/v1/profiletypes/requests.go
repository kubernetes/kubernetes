package profiletypes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body,
		&gophercloud.RequestOpts{OkCodes: []int{200}})

	return
}

// List makes a request against the API to list profile types.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	url := listURL(client)
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ProfileTypePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

func ListOps(client *gophercloud.ServiceClient, id string) pagination.Pager {
	url := listOpsURL(client, id)
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return OperationPage{pagination.SinglePageBase(r)}
	})
}
