package tenants

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// ListOpts filters the Tenants that are returned by the List call.
type ListOpts struct {
	// Marker is the ID of the last Tenant on the previous page.
	Marker string `q:"marker"`

	// Limit specifies the page size.
	Limit int `q:"limit"`
}

// List enumerates the Tenants to which the current token has access.
func List(client *gophercloud.ServiceClient, opts *ListOpts) pagination.Pager {
	createPage := func(r pagination.PageResult) pagination.Page {
		return TenantPage{pagination.LinkedPageBase{PageResult: r}}
	}

	url := listURL(client)
	if opts != nil {
		q, err := gophercloud.BuildQueryString(opts)
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += q.String()
	}

	return pagination.NewPager(client, url, createPage)
}
