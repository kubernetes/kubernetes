package sharetypes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToShareTypeCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains options for creating a ShareType. This object is
// passed to the sharetypes.Create function. For more information about
// these parameters, see the ShareType object.
type CreateOpts struct {
	// The share type name
	Name string `json:"name" required:"true"`
	// Indicates whether a share type is publicly accessible
	IsPublic bool `json:"os-share-type-access:is_public"`
	// The extra specifications for the share type
	ExtraSpecs ExtraSpecsOpts `json:"extra_specs" required:"true"`
}

// ExtraSpecsOpts represent the extra specifications that can be selected for a share type
type ExtraSpecsOpts struct {
	// An extra specification that defines the driver mode for share server, or storage, life cycle management
	DriverHandlesShareServers bool `json:"driver_handles_share_servers" required:"true"`
	// An extra specification that filters back ends by whether they do or do not support share snapshots
	SnapshotSupport *bool `json:"snapshot_support,omitempty"`
}

// ToShareTypeCreateMap assembles a request body based on the contents of a
// CreateOpts.
func (opts CreateOpts) ToShareTypeCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "share_type")
}

// Create will create a new ShareType based on the values in CreateOpts. To
// extract the ShareType object from the response, call the Extract method
// on the CreateResult.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToShareTypeCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 202},
	})
	return
}

// Delete will delete the existing ShareType with the provided ID.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the List
// request.
type ListOptsBuilder interface {
	ToShareTypeListQuery() (string, error)
}

// ListOpts holds options for listing ShareTypes. It is passed to the
// sharetypes.List function.
type ListOpts struct {
	// Select if public types, private types, or both should be listed
	IsPublic string `q:"is_public"`
}

// ToShareTypeListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToShareTypeListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns ShareTypes optionally limited by the conditions provided in ListOpts.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToShareTypeListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ShareTypePage{pagination.SinglePageBase(r)}
	})
}
