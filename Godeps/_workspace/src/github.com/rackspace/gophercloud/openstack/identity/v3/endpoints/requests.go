package endpoints

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// EndpointOpts contains the subset of Endpoint attributes that should be used to create or update an Endpoint.
type EndpointOpts struct {
	Availability gophercloud.Availability
	Name         string
	Region       string
	URL          string
	ServiceID    string
}

// Create inserts a new Endpoint into the service catalog.
// Within EndpointOpts, Region may be omitted by being left as "", but all other fields are required.
func Create(client *gophercloud.ServiceClient, opts EndpointOpts) CreateResult {
	// Redefined so that Region can be re-typed as a *string, which can be omitted from the JSON output.
	type endpoint struct {
		Interface string  `json:"interface"`
		Name      string  `json:"name"`
		Region    *string `json:"region,omitempty"`
		URL       string  `json:"url"`
		ServiceID string  `json:"service_id"`
	}

	type request struct {
		Endpoint endpoint `json:"endpoint"`
	}

	// Ensure that EndpointOpts is fully populated.
	if opts.Availability == "" {
		return createErr(ErrAvailabilityRequired)
	}
	if opts.Name == "" {
		return createErr(ErrNameRequired)
	}
	if opts.URL == "" {
		return createErr(ErrURLRequired)
	}
	if opts.ServiceID == "" {
		return createErr(ErrServiceIDRequired)
	}

	// Populate the request body.
	reqBody := request{
		Endpoint: endpoint{
			Interface: string(opts.Availability),
			Name:      opts.Name,
			URL:       opts.URL,
			ServiceID: opts.ServiceID,
		},
	}
	reqBody.Endpoint.Region = gophercloud.MaybeString(opts.Region)

	var result CreateResult
	_, result.Err = client.Post(listURL(client), reqBody, &result.Body, nil)
	return result
}

// ListOpts allows finer control over the endpoints returned by a List call.
// All fields are optional.
type ListOpts struct {
	Availability gophercloud.Availability `q:"interface"`
	ServiceID    string                   `q:"service_id"`
	Page         int                      `q:"page"`
	PerPage      int                      `q:"per_page"`
}

// List enumerates endpoints in a paginated collection, optionally filtered by ListOpts criteria.
func List(client *gophercloud.ServiceClient, opts ListOpts) pagination.Pager {
	u := listURL(client)
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return pagination.Pager{Err: err}
	}
	u += q.String()
	createPage := func(r pagination.PageResult) pagination.Page {
		return EndpointPage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, u, createPage)
}

// Update changes an existing endpoint with new data.
// All fields are optional in the provided EndpointOpts.
func Update(client *gophercloud.ServiceClient, endpointID string, opts EndpointOpts) UpdateResult {
	type endpoint struct {
		Interface *string `json:"interface,omitempty"`
		Name      *string `json:"name,omitempty"`
		Region    *string `json:"region,omitempty"`
		URL       *string `json:"url,omitempty"`
		ServiceID *string `json:"service_id,omitempty"`
	}

	type request struct {
		Endpoint endpoint `json:"endpoint"`
	}

	reqBody := request{Endpoint: endpoint{}}
	reqBody.Endpoint.Interface = gophercloud.MaybeString(string(opts.Availability))
	reqBody.Endpoint.Name = gophercloud.MaybeString(opts.Name)
	reqBody.Endpoint.Region = gophercloud.MaybeString(opts.Region)
	reqBody.Endpoint.URL = gophercloud.MaybeString(opts.URL)
	reqBody.Endpoint.ServiceID = gophercloud.MaybeString(opts.ServiceID)

	var result UpdateResult
	_, result.Err = client.Request("PATCH", endpointURL(client, endpointID), gophercloud.RequestOpts{
		JSONBody:     &reqBody,
		JSONResponse: &result.Body,
		OkCodes:      []int{200},
	})
	return result
}

// Delete removes an endpoint from the service catalog.
func Delete(client *gophercloud.ServiceClient, endpointID string) DeleteResult {
	var res DeleteResult
	_, res.Err = client.Delete(endpointURL(client, endpointID), nil)
	return res
}
