package addressscopes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToAddressScopeListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the Neutron API. Filtering is achieved by passing in struct field values
// that map to the address-scope attributes you want to see returned.
// SortKey allows you to sort by a particular address-scope attribute.
// SortDir sets the direction, and is either `asc' or `desc'.
// Marker and Limit are used for the pagination.
type ListOpts struct {
	ID          string `q:"id"`
	Name        string `q:"name"`
	TenantID    string `q:"tenant_id"`
	ProjectID   string `q:"project_id"`
	IPVersion   int    `q:"ip_version"`
	Shared      *bool  `q:"shared"`
	Description string `q:"description"`
	Limit       int    `q:"limit"`
	Marker      string `q:"marker"`
	SortKey     string `q:"sort_key"`
	SortDir     string `q:"sort_dir"`
}

// ToAddressScopeListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToAddressScopeListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns a Pager which allows you to iterate over a collection of
// address-scopes. It accepts a ListOpts struct, which allows you to filter and
// sort the returned collection for greater efficiency.
//
// Default policy settings return only the address-scopes owned by the project
// of the user submitting the request, unless the user has the administrative
// role.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(c)
	if opts != nil {
		query, err := opts.ToAddressScopeListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return AddressScopePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves a specific address-scope based on its ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(getURL(c, id), &r.Body, nil)
	return
}

// CreateOptsBuilder allows to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToAddressScopeCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies parameters of a new address-scope.
type CreateOpts struct {
	// Name is the human-readable name of the address-scope.
	Name string `json:"name"`

	// TenantID is the id of the Identity project.
	TenantID string `json:"tenant_id,omitempty"`

	// ProjectID is the id of the Identity project.
	ProjectID string `json:"project_id,omitempty"`

	// IPVersion is the IP protocol version.
	IPVersion int `json:"ip_version"`

	// Shared indicates whether this address-scope is shared across all projects.
	Shared bool `json:"shared,omitempty"`
}

// ToAddressScopeCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToAddressScopeCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "address_scope")
}

// Create requests the creation of a new address-scope on the server.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToAddressScopeCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{201},
	})
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToAddressScopeUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents options used to update an address-scope.
type UpdateOpts struct {
	// Name is the human-readable name of the address-scope.
	Name *string `json:"name,omitempty"`

	// Shared indicates whether this address-scope is shared across all projects.
	Shared *bool `json:"shared,omitempty"`
}

// ToAddressScopeUpdateMap builds a request body from UpdateOpts.
func (opts UpdateOpts) ToAddressScopeUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "address_scope")
}

// Update accepts a UpdateOpts struct and updates an existing address-scope
// using the values provided.
func Update(c *gophercloud.ServiceClient, addressScopeID string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToAddressScopeUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(updateURL(c, addressScopeID), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete accepts a unique ID and deletes the address-scope associated with it.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(deleteURL(c, id), nil)
	return
}
