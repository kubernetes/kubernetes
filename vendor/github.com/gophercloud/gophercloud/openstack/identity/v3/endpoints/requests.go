package endpoints

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type CreateOptsBuilder interface {
	ToEndpointCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains the subset of Endpoint attributes that should be used to create an Endpoint.
type CreateOpts struct {
	Availability gophercloud.Availability `json:"interface" required:"true"`
	Name         string                   `json:"name" required:"true"`
	Region       string                   `json:"region,omitempty"`
	URL          string                   `json:"url" required:"true"`
	ServiceID    string                   `json:"service_id" required:"true"`
}

func (opts CreateOpts) ToEndpointCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "endpoint")
}

// Create inserts a new Endpoint into the service catalog.
// Within EndpointOpts, Region may be omitted by being left as "", but all other fields are required.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToEndpointCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(listURL(client), &b, &r.Body, nil)
	return
}

type ListOptsBuilder interface {
	ToEndpointListParams() (string, error)
}

// ListOpts allows finer control over the endpoints returned by a List call.
// All fields are optional.
type ListOpts struct {
	Availability gophercloud.Availability `q:"interface"`
	ServiceID    string                   `q:"service_id"`
	Page         int                      `q:"page"`
	PerPage      int                      `q:"per_page"`
}

func (opts ListOpts) ToEndpointListParams() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List enumerates endpoints in a paginated collection, optionally filtered by ListOpts criteria.
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

type UpdateOptsBuilder interface {
	ToEndpointUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the subset of Endpoint attributes that should be used to update an Endpoint.
type UpdateOpts struct {
	Availability gophercloud.Availability `json:"interface,omitempty"`
	Name         string                   `json:"name,omitempty"`
	Region       string                   `json:"region,omitempty"`
	URL          string                   `json:"url,omitempty"`
	ServiceID    string                   `json:"service_id,omitempty"`
}

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
