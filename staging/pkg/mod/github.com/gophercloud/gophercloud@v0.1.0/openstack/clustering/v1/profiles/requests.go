package profiles

import (
	"net/http"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToProfileCreateMap() (map[string]interface{}, error)
}

// CreateOpts represents options used for creating a profile.
type CreateOpts struct {
	Name     string                 `json:"name" required:"true"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Spec     Spec                   `json:"spec" required:"true"`
}

// ToProfileCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToProfileCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "profile")
}

// Create requests the creation of a new profile on the server.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToProfileCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})

	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// Get retrieves detail of a single profile.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	var result *http.Response
	result, r.Err = client.Get(getURL(client, id), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToProfileListQuery() (string, error)
}

// ListOpts represents options used to list profiles.
type ListOpts struct {
	GlobalProject *bool  `q:"global_project"`
	Limit         int    `q:"limit"`
	Marker        string `q:"marker"`
	Name          string `q:"name"`
	Sort          string `q:"sort"`
	Type          string `q:"type"`
}

// ToProfileListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToProfileListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List instructs OpenStack to provide a list of profiles.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToProfileListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ProfilePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToProfileUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents options used to update a profile.
type UpdateOpts struct {
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	Name     string                 `json:"name,omitempty"`
}

// ToProfileUpdateMap constructs a request body from UpdateOpts.
func (opts UpdateOpts) ToProfileUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "profile")
}

// Update updates a profile.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToProfileUpdateMap()
	if err != nil {
		r.Err = err
		return r
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

// Delete deletes the specified profile via profile id.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	var result *http.Response
	result, r.Err = client.Delete(deleteURL(client, id), nil)
	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// ValidateOptsBuilder allows extensions to add additional parameters to the
// Validate request.
type ValidateOptsBuilder interface {
	ToProfileValidateMap() (map[string]interface{}, error)
}

// ValidateOpts params
type ValidateOpts struct {
	Spec Spec `json:"spec" required:"true"`
}

// ToProfileValidateMap formats a CreateOpts into a body map.
func (opts ValidateOpts) ToProfileValidateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "profile")
}

// Validate profile.
func Validate(client *gophercloud.ServiceClient, opts ValidateOpts) (r ValidateResult) {
	b, err := opts.ToProfileValidateMap()
	if err != nil {
		r.Err = err
		return
	}

	var result *http.Response
	result, r.Err = client.Post(validateURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	if r.Err == nil {
		r.Header = result.Header
	}
	return
}
