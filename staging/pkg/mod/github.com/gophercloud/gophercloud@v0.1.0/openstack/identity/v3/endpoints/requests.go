package endpoints

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type CreateOptsBuilder interface {
	ToEndpointCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains the subset of Endpoint attributes that should be used
// to create an Endpoint.
type CreateOpts struct {
	// Availability is the interface type of the Endpoint (admin, internal,
	// or public), referenced by the gophercloud.Availability type.
	Availability gophercloud.Availability `json:"interface" required:"true"`

	// Name is the name of the Endpoint.
	Name string `json:"name" required:"true"`

	// Region is the region the Endpoint is located in.
	// This field can be omitted or left as a blank string.
	Region string `json:"region,omitempty"`

	// URL is the url of the Endpoint.
	URL string `json:"url" required:"true"`

	// ServiceID is the ID of the service the Endpoint refers to.
	ServiceID string `json:"service_id" required:"true"`
}

// ToEndpointCreateMap builds a request body from the Endpoint Create options.
func (opts CreateOpts) ToEndpointCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "endpoint")
}

// Create inserts a new Endpoint into the service catalog.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToEndpointCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(listURL(client), &b, &r.Body, nil)
	return
}

// ListOptsBuilder allows extensions to add parameters to the List request.
type ListOptsBuilder interface {
	ToEndpointListParams() (string, error)
}

// ListOpts allows finer control over the endpoints returned by a List call.
// All fields are optional.
type ListOpts struct {
	// Availability is the interface type of the Endpoint (admin, internal,
	// or public), referenced by the gophercloud.Availability type.
	Availability gophercloud.Availability `q:"interface"`

	// ServiceID is the ID of the service the Endpoint refers to.
	ServiceID string `q:"service_id"`

	// RegionID is the ID of the region the Endpoint refers to.
	RegionID int `q:"region_id"`
}

// ToEndpointListParams builds a list request from the List options.
func (opts ListOpts) ToEndpointListParams() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List enumerates endpoints in a paginated collection, optionally filtered
// by ListOpts criteria.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	u := listURL(client)
	if opts != nil {
		q, err := gophercloud.BuildQueryString(opts)
		if err != nil {
			return pagination.Pager{Err: err}
		}
		u += q.String()
	}
	return pagination.NewPager(client, u, func(r pagination.PageResult) pagination.Page {
		return EndpointPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// UpdateOptsBuilder allows extensions to add parameters to the Update request.
type UpdateOptsBuilder interface {
	ToEndpointUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the subset of Endpoint attributes that should be used to
// update an Endpoint.
type UpdateOpts struct {
	// Availability is the interface type of the Endpoint (admin, internal,
	// or public), referenced by the gophercloud.Availability type.
	Availability gophercloud.Availability `json:"interface,omitempty"`

	// Name is the name of the Endpoint.
	Name string `json:"name,omitempty"`

	// Region is the region the Endpoint is located in.
	// This field can be omitted or left as a blank string.
	Region string `json:"region,omitempty"`

	// URL is the url of the Endpoint.
	URL string `json:"url,omitempty"`

	// ServiceID is the ID of the service the Endpoint refers to.
	ServiceID string `json:"service_id,omitempty"`
}

// ToEndpointUpdateMap builds an update request body from the Update options.
func (opts UpdateOpts) ToEndpointUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "endpoint")
}

// Update changes an existing endpoint with new data.
func Update(client *gophercloud.ServiceClient, endpointID string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToEndpointUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Patch(endpointURL(client, endpointID), &b, &r.Body, nil)
	return
}

// Delete removes an endpoint from the service catalog.
func Delete(client *gophercloud.ServiceClient, endpointID string) (r DeleteResult) {
	_, r.Err = client.Delete(endpointURL(client, endpointID), nil)
	return
}
