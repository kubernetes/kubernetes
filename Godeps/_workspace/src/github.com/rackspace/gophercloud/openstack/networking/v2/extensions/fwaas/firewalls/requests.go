package firewalls

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// AdminState gives users a solid type to work with for create and update
// operations. It is recommended that users use the `Up` and `Down` enums.
type AdminState *bool

// Shared gives users a solid type to work with for create and update
// operations. It is recommended that users use the `Yes` and `No` enums.
type Shared *bool

// Convenience vars for AdminStateUp and Shared values.
var (
	iTrue             = true
	iFalse            = false
	Up     AdminState = &iTrue
	Down   AdminState = &iFalse
	Yes    Shared     = &iTrue
	No     Shared     = &iFalse
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
	if err != nil {
		return "", err
	}
	return q.String(), nil
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

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToFirewallCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new firewall.
type CreateOpts struct {
	// Only required if the caller has an admin role and wants to create a firewall
	// for another tenant.
	TenantID     string
	Name         string
	Description  string
	AdminStateUp *bool
	Shared       *bool
	PolicyID     string
}

// ToFirewallCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToFirewallCreateMap() (map[string]interface{}, error) {
	if opts.PolicyID == "" {
		return nil, errPolicyRequired
	}

	f := make(map[string]interface{})

	if opts.TenantID != "" {
		f["tenant_id"] = opts.TenantID
	}
	if opts.Name != "" {
		f["name"] = opts.Name
	}
	if opts.Description != "" {
		f["description"] = opts.Description
	}
	if opts.Shared != nil {
		f["shared"] = *opts.Shared
	}
	if opts.AdminStateUp != nil {
		f["admin_state_up"] = *opts.AdminStateUp
	}
	if opts.PolicyID != "" {
		f["firewall_policy_id"] = opts.PolicyID
	}

	return map[string]interface{}{"firewall": f}, nil
}

// Create accepts a CreateOpts struct and uses the values to create a new firewall
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToFirewallCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular firewall based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(resourceURL(c, id), &res.Body, nil)
	return res
}

// UpdateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Update operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type UpdateOptsBuilder interface {
	ToFirewallUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the values used when updating a firewall.
type UpdateOpts struct {
	// Name of the firewall.
	Name         string
	Description  string
	AdminStateUp *bool
	Shared       *bool
	PolicyID     string
}

// ToFirewallUpdateMap casts a CreateOpts struct to a map.
func (opts UpdateOpts) ToFirewallUpdateMap() (map[string]interface{}, error) {
	f := make(map[string]interface{})

	if opts.Name != "" {
		f["name"] = opts.Name
	}
	if opts.Description != "" {
		f["description"] = opts.Description
	}
	if opts.Shared != nil {
		f["shared"] = *opts.Shared
	}
	if opts.AdminStateUp != nil {
		f["admin_state_up"] = *opts.AdminStateUp
	}
	if opts.PolicyID != "" {
		f["firewall_policy_id"] = opts.PolicyID
	}

	return map[string]interface{}{"firewall": f}, nil
}

// Update allows firewalls to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToFirewallUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Put(resourceURL(c, id), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Delete will permanently delete a particular firewall based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}
