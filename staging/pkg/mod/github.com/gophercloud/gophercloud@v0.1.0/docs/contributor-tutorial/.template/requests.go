package RESOURCE

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to
// the List request
type ListOptsBuilder interface {
	ToResourceListQuery() (string, error)
}

// ListOpts provides options to filter the List results.
type ListOpts struct {
}

// ToResourceListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToResourceListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List retrieves a list of RESOURCES.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToResourceListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ResourcePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves details of a RESOURCE.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// CreateOptsBuilder allows extensions to add additional parameters to
// the Create request.
type CreateOptsBuilder interface {
	ToResourceCreateMap() (map[string]interface{}, error)
}

// CreateOpts provides options used to create a RESOURCE.
type CreateOpts struct {
}

// ToResourceCreateMap formats a CreateOpts into a create request.
func (opts CreateOpts) ToResourceCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "resource")
}

// Create creates a new RESOURCE.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToResourceCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{201},
	})
	return
}

// Delete deletes a RESOURCE.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to
// the Update request.
type UpdateOptsBuilder interface {
	ToResourceUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents parameters to update a RESOURCE.
type UpdateOpts struct {
}

// ToUpdateCreateMap formats a UpdateOpts into an update request.
func (opts UpdateOpts) ToResourceUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "resource")
}

// Update modifies the attributes of a RESOURCE.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToResourceUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Patch(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
