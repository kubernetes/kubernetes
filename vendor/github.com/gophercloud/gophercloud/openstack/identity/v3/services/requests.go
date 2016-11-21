package services

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Create adds a new service of the requested type to the catalog.
func Create(client *gophercloud.ServiceClient, serviceType string) (r CreateResult) {
	b := map[string]string{"type": serviceType}
	_, r.Err = client.Post(listURL(client), b, &r.Body, nil)
	return
}

type ListOptsBuilder interface {
	ToServiceListMap() (string, error)
}

// ListOpts allows you to query the List method.
type ListOpts struct {
	ServiceType string `q:"type"`
	PerPage     int    `q:"perPage"`
	Page        int    `q:"page"`
}

func (opts ListOpts) ToServiceListMap() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List enumerates the services available to a specific user.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	u := listURL(client)
	if opts != nil {
		q, err := opts.ToServiceListMap()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		u += q
	}
	return pagination.NewPager(client, u, func(r pagination.PageResult) pagination.Page {
		return ServicePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get returns additional information about a service, given its ID.
func Get(client *gophercloud.ServiceClient, serviceID string) (r GetResult) {
	_, r.Err = client.Get(serviceURL(client, serviceID), &r.Body, nil)
	return
}

// Update changes the service type of an existing service.
func Update(client *gophercloud.ServiceClient, serviceID string, serviceType string) (r UpdateResult) {
	b := map[string]string{"type": serviceType}
	_, r.Err = client.Patch(serviceURL(client, serviceID), &b, &r.Body, nil)
	return
}

// Delete removes an existing service.
// It either deletes all associated endpoints, or fails until all endpoints are deleted.
func Delete(client *gophercloud.ServiceClient, serviceID string) (r DeleteResult) {
	_, r.Err = client.Delete(serviceURL(client, serviceID), nil)
	return
}
