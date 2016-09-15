package policies

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
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
	Shared      *bool  `q:"shared"`
	Audited     *bool  `q:"audited"`
	ID          string `q:"id"`
	Limit       int    `q:"limit"`
	Marker      string `q:"marker"`
	SortKey     string `q:"sort_key"`
	SortDir     string `q:"sort_dir"`
}

// ToPolicyListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToPolicyListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
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
	ToFirewallPolicyCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new firewall policy.
type CreateOpts struct {
	// Only required if the caller has an admin role and wants to create a firewall policy
	// for another tenant.
	TenantID    string   `json:"tenant_id,omitempty"`
	Name        string   `json:"name,omitempty"`
	Description string   `json:"description,omitempty"`
	Shared      *bool    `json:"shared,omitempty"`
	Audited     *bool    `json:"audited,omitempty"`
	Rules       []string `json:"firewall_rules,omitempty"`
}

// ToFirewallPolicyCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToFirewallPolicyCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "firewall_policy")
}

// Create accepts a CreateOpts struct and uses the values to create a new firewall policy
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToFirewallPolicyCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(rootURL(c), b, &r.Body, nil)
	return
}

// Get retrieves a particular firewall policy based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(resourceURL(c, id), &r.Body, nil)
	return
}

// UpdateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Update operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type UpdateOptsBuilder interface {
	ToFirewallPolicyUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the values used when updating a firewall policy.
type UpdateOpts struct {
	Name        string   `json:"name,omitempty"`
	Description string   `json:"description,omitempty"`
	Shared      *bool    `json:"shared,omitempty"`
	Audited     *bool    `json:"audited,omitempty"`
	Rules       []string `json:"firewall_rules,omitempty"`
}

// ToFirewallPolicyUpdateMap casts a CreateOpts struct to a map.
func (opts UpdateOpts) ToFirewallPolicyUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "firewall_policy")
}

// Update allows firewall policies to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToFirewallPolicyUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(resourceURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete will permanently delete a particular firewall policy based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}

type InsertRuleOptsBuilder interface {
	ToFirewallPolicyInsertRuleMap() (map[string]interface{}, error)
}

type InsertRuleOpts struct {
	ID           string `json:"firewall_rule_id" required:"true"`
	BeforeRuleID string `json:"insert_before,omitempty"`
	AfterRuleID  string `json:"insert_after,omitempty"`
}

func (opts InsertRuleOpts) ToFirewallPolicyInsertRuleMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

func AddRule(c *gophercloud.ServiceClient, id string, opts InsertRuleOptsBuilder) (r InsertRuleResult) {
	b, err := opts.ToFirewallPolicyInsertRuleMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(insertURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

func RemoveRule(c *gophercloud.ServiceClient, id, ruleID string) (r RemoveRuleResult) {
	b := map[string]interface{}{"firewall_rule_id": ruleID}
	_, r.Err = c.Put(removeURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
