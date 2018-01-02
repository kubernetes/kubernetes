package schedulerstats

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToStoragePoolsListQuery() (string, error)
}

// ListOpts controls the view of data returned (e.g globally or per project)
// via tenant_id and the verbosity via detail.
type ListOpts struct {
	// ID of the tenant to look up storage pools for.
	TenantID string `q:"tenant_id"`

	// Whether to list extended details.
	Detail bool `q:"detail"`
}

// ToStoragePoolsListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToStoragePoolsListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List makes a request against the API to list storage pool information.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := storagePoolsListURL(client)
	if opts != nil {
		query, err := opts.ToStoragePoolsListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return StoragePoolPage{pagination.SinglePageBase(r)}
	})
}
