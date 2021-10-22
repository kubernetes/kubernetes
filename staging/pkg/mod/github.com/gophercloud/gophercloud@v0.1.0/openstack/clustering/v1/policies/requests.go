package policies

import (
	"net/http"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToPolicyListQuery() (string, error)
}

// ListOpts represents options used to list policies.
type ListOpts struct {
	// Limit limits the number of Policies to return.
	Limit int `q:"limit"`

	// Marker and Limit control paging. Marker instructs List where to start
	// listing from.
	Marker string `q:"marker"`

	// Sorts the response by one or more attribute and optional sort direction
	// combinations.
	Sort string `q:"sort"`

	// GlobalProject indicates whether to include resources for all projects or
	// resources for the current project.
	GlobalProject *bool `q:"global_project"`

	// Name to filter the response by the specified name property of the object.
	Name string `q:"name"`

	// Filter the response by the specified type property of the object.
	Type string `q:"type"`
}

// ToPolicyListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToPolicyListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List instructs OpenStack to retrieve a list of policies.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := policyListURL(client)
	if opts != nil {
		query, err := opts.ToPolicyListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		p := PolicyPage{pagination.MarkerPageBase{PageResult: r}}
		p.MarkerPageBase.Owner = p
		return p
	})
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToPolicyCreateMap() (map[string]interface{}, error)
}

// CreateOpts represents options used to create a policy.
type CreateOpts struct {
	Name string `json:"name"`
	Spec Spec   `json:"spec"`
}

// ToPolicyCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToPolicyCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{"policy": b}, nil
}

// Create makes a request against the API to create a policy
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToPolicyCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(policyCreateURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{201},
	})
	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// Delete makes a request against the API to delete a policy.
func Delete(client *gophercloud.ServiceClient, policyID string) (r DeleteResult) {
	var result *http.Response
	result, r.Err = client.Delete(policyDeleteURL(client, policyID), &gophercloud.RequestOpts{
		OkCodes: []int{204},
	})
	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToPolicyUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents options to update a policy.
type UpdateOpts struct {
	Name string `json:"name,omitempty"`
}

// ToPolicyUpdateMap constructs a request body from UpdateOpts.
func (opts UpdateOpts) ToPolicyUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "policy")
}

// Update updates a specified policy.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToPolicyUpdateMap()
	if err != nil {
		r.Err = err
		return
	}

	var result *http.Response
	result, r.Err = client.Patch(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// ValidateOptsBuilder allows extensions to add additional parameters to the
// Validate request.
type ValidateOptsBuilder interface {
	ToPolicyValidateMap() (map[string]interface{}, error)
}

// ValidateOpts represents options used to validate a policy.
type ValidateOpts struct {
	Spec Spec `json:"spec"`
}

// ToPolicyValidateMap formats a CreateOpts into a body map.
func (opts ValidateOpts) ToPolicyValidateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "policy")
}

// Validate policy will validate a specified policy.
func Validate(client *gophercloud.ServiceClient, opts ValidateOptsBuilder) (r ValidateResult) {
	b, err := opts.ToPolicyValidateMap()
	if err != nil {
		r.Err = err
		return
	}

	var result *http.Response
	result, r.Err = client.Post(validateURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})
	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// Get makes a request against the API to get details for a policy.
func Get(client *gophercloud.ServiceClient, policyTypeName string) (r GetResult) {
	url := policyGetURL(client, policyTypeName)

	_, r.Err = client.Get(url, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return
}
