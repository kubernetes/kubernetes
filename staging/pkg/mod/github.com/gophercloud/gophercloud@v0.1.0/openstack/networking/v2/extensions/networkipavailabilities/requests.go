package networkipavailabilities

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToNetworkIPAvailabilityListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the Neutron API.
type ListOpts struct {
	// NetworkName allows to filter on the identifier of a network.
	NetworkID string `q:"network_id"`

	// NetworkName allows to filter on the name of a network.
	NetworkName string `q:"network_name"`

	// IPVersion allows to filter on the version of the IP protocol.
	// You can use the well-known IP versions with the gophercloud.IPVersion type.
	IPVersion string `q:"ip_version"`

	// ProjectID allows to filter on the Identity project field.
	ProjectID string `q:"project_id"`

	// TenantID allows to filter on the Identity project field.
	TenantID string `q:"tenant_id"`
}

// ToNetworkIPAvailabilityListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToNetworkIPAvailabilityListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns a Pager which allows you to iterate over a collection of
// networkipavailabilities. It accepts a ListOpts struct, which allows you to
// filter the returned collection for greater efficiency.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(c)
	if opts != nil {
		query, err := opts.ToNetworkIPAvailabilityListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return NetworkIPAvailabilityPage{pagination.SinglePageBase(r)}
	})
}

// Get retrieves a specific NetworkIPAvailability based on its ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(getURL(c, id), &r.Body, nil)
	return
}
