package regions

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to
// the List request
type ListOptsBuilder interface {
	ToRegionListQuery() (string, error)
}

// ListOpts provides options to filter the List results.
type ListOpts struct {
	// ParentRegionID filters the response by a parent region ID.
	ParentRegionID string `q:"parent_region_id"`
}

// ToRegionListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToRegionListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List enumerates the Regions to which the current token has access.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToRegionListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return RegionPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves details on a single region, by ID.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// CreateOptsBuilder allows extensions to add additional parameters to
// the Create request.
type CreateOptsBuilder interface {
	ToRegionCreateMap() (map[string]interface{}, error)
}

// CreateOpts provides options used to create a region.
type CreateOpts struct {
	// ID is the ID of the new region.
	ID string `json:"id,omitempty"`

	// Description is a description of the region.
	Description string `json:"description,omitempty"`

	// ParentRegionID is the ID of the parent the region to add this region under.
	ParentRegionID string `json:"parent_region_id,omitempty"`

	// Extra is free-form extra key/value pairs to describe the region.
	Extra map[string]interface{} `json:"-"`
}

// ToRegionCreateMap formats a CreateOpts into a create request.
func (opts CreateOpts) ToRegionCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "region")
	if err != nil {
		return nil, err
	}

	if opts.Extra != nil {
		if v, ok := b["region"].(map[string]interface{}); ok {
			for key, value := range opts.Extra {
				v[key] = value
			}
		}
	}

	return b, nil
}

// Create creates a new Region.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToRegionCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{201},
	})
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to
// the Update request.
type UpdateOptsBuilder interface {
	ToRegionUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts provides options for updating a region.
type UpdateOpts struct {
	// Description is a description of the region.
	Description string `json:"description,omitempty"`

	// ParentRegionID is the ID of the parent region.
	ParentRegionID string `json:"parent_region_id,omitempty"`

	/*
		// Due to a bug in Keystone, the Extra column of the Region table
		// is not updatable, see: https://bugs.launchpad.net/keystone/+bug/1729933
		// The following lines should be uncommented once the fix is merged.

		// Extra is free-form extra key/value pairs to describe the region.
		Extra map[string]interface{} `json:"-"`
	*/
}

// ToRegionUpdateMap formats a UpdateOpts into an update request.
func (opts UpdateOpts) ToRegionUpdateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "region")
	if err != nil {
		return nil, err
	}

	/*
		// Due to a bug in Keystone, the Extra column of the Region table
		// is not updatable, see: https://bugs.launchpad.net/keystone/+bug/1729933
		// The following lines should be uncommented once the fix is merged.

		if opts.Extra != nil {
			if v, ok := b["region"].(map[string]interface{}); ok {
				for key, value := range opts.Extra {
					v[key] = value
				}
			}
		}
	*/

	return b, nil
}

// Update updates an existing Region.
func Update(client *gophercloud.ServiceClient, regionID string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToRegionUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Patch(updateURL(client, regionID), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete deletes a region.
func Delete(client *gophercloud.ServiceClient, regionID string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, regionID), nil)
	return
}
