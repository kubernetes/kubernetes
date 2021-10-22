package projects

import (
	"net/url"
	"strings"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to
// the List request
type ListOptsBuilder interface {
	ToProjectListQuery() (string, error)
}

// ListOpts enables filtering of a list request.
type ListOpts struct {
	// DomainID filters the response by a domain ID.
	DomainID string `q:"domain_id"`

	// Enabled filters the response by enabled projects.
	Enabled *bool `q:"enabled"`

	// IsDomain filters the response by projects that are domains.
	// Setting this to true is effectively listing domains.
	IsDomain *bool `q:"is_domain"`

	// Name filters the response by project name.
	Name string `q:"name"`

	// ParentID filters the response by projects of a given parent project.
	ParentID string `q:"parent_id"`

	// Filters filters the response by custom filters such as
	// 'name__contains=foo'
	Filters map[string]string `q:"-"`
}

// ToProjectListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToProjectListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}

	params := q.Query()
	for k, v := range opts.Filters {
		i := strings.Index(k, "__")
		if i > 0 && i < len(k)-2 {
			params.Add(k, v)
		} else {
			return "", InvalidListFilter{FilterName: k}
		}
	}

	q = &url.URL{RawQuery: params.Encode()}
	return q.String(), err
}

// List enumerates the Projects to which the current token has access.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToProjectListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ProjectPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves details on a single project, by ID.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// CreateOptsBuilder allows extensions to add additional parameters to
// the Create request.
type CreateOptsBuilder interface {
	ToProjectCreateMap() (map[string]interface{}, error)
}

// CreateOpts represents parameters used to create a project.
type CreateOpts struct {
	// DomainID is the ID this project will belong under.
	DomainID string `json:"domain_id,omitempty"`

	// Enabled sets the project status to enabled or disabled.
	Enabled *bool `json:"enabled,omitempty"`

	// IsDomain indicates if this project is a domain.
	IsDomain *bool `json:"is_domain,omitempty"`

	// Name is the name of the project.
	Name string `json:"name" required:"true"`

	// ParentID specifies the parent project of this new project.
	ParentID string `json:"parent_id,omitempty"`

	// Description is the description of the project.
	Description string `json:"description,omitempty"`
}

// ToProjectCreateMap formats a CreateOpts into a create request.
func (opts CreateOpts) ToProjectCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "project")
}

// Create creates a new Project.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToProjectCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), &b, &r.Body, nil)
	return
}

// Delete deletes a project.
func Delete(client *gophercloud.ServiceClient, projectID string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, projectID), nil)
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to
// the Update request.
type UpdateOptsBuilder interface {
	ToProjectUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents parameters to update a project.
type UpdateOpts struct {
	// DomainID is the ID this project will belong under.
	DomainID string `json:"domain_id,omitempty"`

	// Enabled sets the project status to enabled or disabled.
	Enabled *bool `json:"enabled,omitempty"`

	// IsDomain indicates if this project is a domain.
	IsDomain *bool `json:"is_domain,omitempty"`

	// Name is the name of the project.
	Name string `json:"name,omitempty"`

	// ParentID specifies the parent project of this new project.
	ParentID string `json:"parent_id,omitempty"`

	// Description is the description of the project.
	Description *string `json:"description,omitempty"`
}

// ToUpdateCreateMap formats a UpdateOpts into an update request.
func (opts UpdateOpts) ToProjectUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "project")
}

// Update modifies the attributes of a project.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToProjectUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Patch(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
