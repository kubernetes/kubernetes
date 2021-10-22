package volumetypes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToVolumeTypeCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains options for creating a Volume Type. This object is passed to
// the volumetypes.Create function. For more information about these parameters,
// see the Volume Type object.
type CreateOpts struct {
	// The name of the volume type
	Name string `json:"name" required:"true"`
	// The volume type description
	Description string `json:"description,omitempty"`
	// the ID of the existing volume snapshot
	IsPublic *bool `json:"os-volume-type-access:is_public,omitempty"`
	// Extra spec key-value pairs defined by the user.
	ExtraSpecs map[string]string `json:"extra_specs"`
}

// ToVolumeTypeCreateMap assembles a request body based on the contents of a
// CreateOpts.
func (opts CreateOpts) ToVolumeTypeCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "volume_type")
}

// Create will create a new Volume Type based on the values in CreateOpts. To extract
// the Volume Type object from the response, call the Extract method on the
// CreateResult.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToVolumeTypeCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete will delete the existing Volume Type with the provided ID.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// Get retrieves the Volume Type with the provided ID. To extract the Volume Type object
// from the response, call the Extract method on the GetResult.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the List
// request.
type ListOptsBuilder interface {
	ToVolumeTypeListQuery() (string, error)
}

// ListOpts holds options for listing Volume Types. It is passed to the volumetypes.List
// function.
type ListOpts struct {
	// Comma-separated list of sort keys and optional sort directions in the
	// form of <key>[:<direction>].
	Sort string `q:"sort"`
	// Requests a page size of items.
	Limit int `q:"limit"`
	// Used in conjunction with limit to return a slice of items.
	Offset int `q:"offset"`
	// The ID of the last-seen item.
	Marker string `q:"marker"`
}

// ToVolumeTypeListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToVolumeTypeListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns Volume types.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)

	if opts != nil {
		query, err := opts.ToVolumeTypeListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return VolumeTypePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToVolumeTypeUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contain options for updating an existing Volume Type. This object is passed
// to the volumetypes.Update function. For more information about the parameters, see
// the Volume Type object.
type UpdateOpts struct {
	Name        *string `json:"name,omitempty"`
	Description *string `json:"description,omitempty"`
	IsPublic    *bool   `json:"is_public,omitempty"`
}

// ToVolumeUpdateMap assembles a request body based on the contents of an
// UpdateOpts.
func (opts UpdateOpts) ToVolumeTypeUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "volume_type")
}

// Update will update the Volume Type with provided information. To extract the updated
// Volume Type from the response, call the Extract method on the UpdateResult.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToVolumeTypeUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Put(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
