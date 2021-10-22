package firewalls

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToFirewallListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the firewall attributes you want to see returned. SortKey allows you to sort
// by a particular firewall attribute. SortDir sets the direction, and is either
// `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	TenantID     string `q:"tenant_id"`
	ProjectID    string `q:"project_id"`
	Name         string `q:"name"`
	Description  string `q:"description"`
	AdminStateUp bool   `q:"admin_state_up"`
	Shared       bool   `q:"shared"`
	PolicyID     string `q:"firewall_policy_id"`
	ID           string `q:"id"`
	Limit        int    `q:"limit"`
	Marker       string `q:"marker"`
	SortKey      string `q:"sort_key"`
	SortDir      string `q:"sort_dir"`
}

// ToFirewallListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToFirewallListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns a Pager which allows you to iterate over a collection of
// firewalls. It accepts a ListOpts struct, which allows you to filter
// and sort the returned collection for greater efficiency.
//
// Default policy settings return only those firewalls that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(c)
	if opts != nil {
		query, err := opts.ToFirewallListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return FirewallPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToFirewallCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new firewall.
type CreateOpts struct {
	PolicyID string `json:"firewall_policy_id" required:"true"`
	// TenantID specifies a tenant to own the firewall. The caller must have
	// an admin role in order to set this. Otherwise, this field is left unset
	// and the caller will be the owner.
	TenantID     string `json:"tenant_id,omitempty"`
	ProjectID    string `json:"project_id,omitempty"`
	Name         string `json:"name,omitempty"`
	Description  string `json:"description,omitempty"`
	AdminStateUp *bool  `json:"admin_state_up,omitempty"`
	Shared       *bool  `json:"shared,omitempty"`
}

// ToFirewallCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToFirewallCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "firewall")
}

// Create accepts a CreateOpts struct and uses the values to create a new firewall.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToFirewallCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(rootURL(c), b, &r.Body, nil)
	return
}

// Get retrieves a particular firewall based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(resourceURL(c, id), &r.Body, nil)
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToFirewallUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the values used when updating a firewall.
type UpdateOpts struct {
	PolicyID     string  `json:"firewall_policy_id" required:"true"`
	Name         *string `json:"name,omitempty"`
	Description  *string `json:"description,omitempty"`
	AdminStateUp *bool   `json:"admin_state_up,omitempty"`
	Shared       *bool   `json:"shared,omitempty"`
}

// ToFirewallUpdateMap casts a CreateOpts struct to a map.
func (opts UpdateOpts) ToFirewallUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "firewall")
}

// Update allows firewalls to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToFirewallUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(resourceURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete will permanently delete a particular firewall based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}
