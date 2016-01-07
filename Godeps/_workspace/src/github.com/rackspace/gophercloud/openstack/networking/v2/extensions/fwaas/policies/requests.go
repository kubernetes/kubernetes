package policies

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Binary gives users a solid type to work with for create and update
// operations. It is recommended that users use the `Yes` and `No` enums
type Binary *bool

// Convenience vars for Audited and Shared values.
var (
	iTrue         = true
	iFalse        = false
	Yes    Binary = &iTrue
	No     Binary = &iFalse
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToPolicyListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the firewall policy attributes you want to see returned. SortKey allows you
// to sort by a particular firewall policy attribute. SortDir sets the direction,
// and is either `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	TenantID    string `q:"tenant_id"`
	Name        string `q:"name"`
	Description string `q:"description"`
	Shared      bool   `q:"shared"`
	Audited     bool   `q:"audited"`
	ID          string `q:"id"`
	Limit       int    `q:"limit"`
	Marker      string `q:"marker"`
	SortKey     string `q:"sort_key"`
	SortDir     string `q:"sort_dir"`
}

// ToPolicyListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToPolicyListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List returns a Pager which allows you to iterate over a collection of
// firewall policies. It accepts a ListOpts struct, which allows you to filter
// and sort the returned collection for greater efficiency.
//
// Default policy settings return only those firewall policies that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(c)

	if opts != nil {
		query, err := opts.ToPolicyListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return PolicyPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToPolicyCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new firewall policy.
type CreateOpts struct {
	// Only required if the caller has an admin role and wants to create a firewall policy
	// for another tenant.
	TenantID    string
	Name        string
	Description string
	Shared      *bool
	Audited     *bool
	Rules       []string
}

// ToPolicyCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToPolicyCreateMap() (map[string]interface{}, error) {
	p := make(map[string]interface{})

	if opts.TenantID != "" {
		p["tenant_id"] = opts.TenantID
	}
	if opts.Name != "" {
		p["name"] = opts.Name
	}
	if opts.Description != "" {
		p["description"] = opts.Description
	}
	if opts.Shared != nil {
		p["shared"] = *opts.Shared
	}
	if opts.Audited != nil {
		p["audited"] = *opts.Audited
	}
	if opts.Rules != nil {
		p["firewall_rules"] = opts.Rules
	}

	return map[string]interface{}{"firewall_policy": p}, nil
}

// Create accepts a CreateOpts struct and uses the values to create a new firewall policy
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToPolicyCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular firewall policy based on its unique ID.
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
	ToPolicyUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the values used when updating a firewall policy.
type UpdateOpts struct {
	// Name of the firewall policy.
	Name        string
	Description string
	Shared      *bool
	Audited     *bool
	Rules       []string
}

// ToPolicyUpdateMap casts a CreateOpts struct to a map.
func (opts UpdateOpts) ToPolicyUpdateMap() (map[string]interface{}, error) {
	p := make(map[string]interface{})

	if opts.Name != "" {
		p["name"] = opts.Name
	}
	if opts.Description != "" {
		p["description"] = opts.Description
	}
	if opts.Shared != nil {
		p["shared"] = *opts.Shared
	}
	if opts.Audited != nil {
		p["audited"] = *opts.Audited
	}
	if opts.Rules != nil {
		p["firewall_rules"] = opts.Rules
	}

	return map[string]interface{}{"firewall_policy": p}, nil
}

// Update allows firewall policies to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToPolicyUpdateMap()
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

// Delete will permanently delete a particular firewall policy based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}

func InsertRule(c *gophercloud.ServiceClient, policyID, ruleID, beforeID, afterID string) error {
	type request struct {
		RuleId string `json:"firewall_rule_id"`
		Before string `json:"insert_before,omitempty"`
		After  string `json:"insert_after,omitempty"`
	}

	reqBody := request{
		RuleId: ruleID,
		Before: beforeID,
		After:  afterID,
	}

	// Send request to API
	var res commonResult
	_, res.Err = c.Put(insertURL(c, policyID), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res.Err
}

func RemoveRule(c *gophercloud.ServiceClient, policyID, ruleID string) error {
	type request struct {
		RuleId string `json:"firewall_rule_id"`
	}

	reqBody := request{
		RuleId: ruleID,
	}

	// Send request to API
	var res commonResult
	_, res.Err = c.Put(removeURL(c, policyID), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res.Err
}
