package policytypes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// List makes a request against the API to list policy types.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	url := policyTypeListURL(client)

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return PolicyTypePage{pagination.SinglePageBase(r)}
	})
}

// Get makes a request against the API to get details for a policy type.
func Get(client *gophercloud.ServiceClient, policyTypeName string) (r GetResult) {
	url := policyTypeGetURL(client, policyTypeName)

	_, r.Err = client.Get(url, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return
}
