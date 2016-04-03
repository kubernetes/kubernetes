package services

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

type response struct {
	Service Service `json:"service"`
}

// Create adds a new service of the requested type to the catalog.
func Create(client *gophercloud.ServiceClient, serviceType string) CreateResult {
	type request struct {
		Type string `json:"type"`
	}

	req := request{Type: serviceType}

	var result CreateResult
	_, result.Err = client.Post(listURL(client), req, &result.Body, nil)
	return result
}

// ListOpts allows you to query the List method.
type ListOpts struct {
	ServiceType string `q:"type"`
	PerPage     int    `q:"perPage"`
	Page        int    `q:"page"`
}

// List enumerates the services available to a specific user.
func List(client *gophercloud.ServiceClient, opts ListOpts) pagination.Pager {
	u := listURL(client)
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return pagination.Pager{Err: err}
	}
	u += q.String()
	createPage := func(r pagination.PageResult) pagination.Page {
		return ServicePage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, u, createPage)
}

// Get returns additional information about a service, given its ID.
func Get(client *gophercloud.ServiceClient, serviceID string) GetResult {
	var result GetResult
	_, result.Err = client.Get(serviceURL(client, serviceID), &result.Body, nil)
	return result
}

// Update changes the service type of an existing service.
func Update(client *gophercloud.ServiceClient, serviceID string, serviceType string) UpdateResult {
	type request struct {
		Type string `json:"type"`
	}

	req := request{Type: serviceType}

	var result UpdateResult
	_, result.Err = client.Request("PATCH", serviceURL(client, serviceID), gophercloud.RequestOpts{
		JSONBody:     &req,
		JSONResponse: &result.Body,
		OkCodes:      []int{200},
	})
	return result
}

// Delete removes an existing service.
// It either deletes all associated endpoints, or fails until all endpoints are deleted.
func Delete(client *gophercloud.ServiceClient, serviceID string) DeleteResult {
	var res DeleteResult
	_, res.Err = client.Delete(serviceURL(client, serviceID), nil)
	return res
}
