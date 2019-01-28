package l7policies

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToL7PolicyCreateMap() (map[string]interface{}, error)
}

type Action string
type RuleType string
type CompareType string

const (
	ActionRedirectToPool Action = "REDIRECT_TO_POOL"
	ActionRedirectToURL  Action = "REDIRECT_TO_URL"
	ActionReject         Action = "REJECT"

	TypeCookie   RuleType = "COOKIE"
	TypeFileType RuleType = "FILE_TYPE"
	TypeHeader   RuleType = "HEADER"
	TypeHostName RuleType = "HOST_NAME"
	TypePath     RuleType = "PATH"

	CompareTypeContains  CompareType = "CONTAINS"
	CompareTypeEndWith   CompareType = "ENDS_WITH"
	CompareTypeEqual     CompareType = "EQUAL_TO"
	CompareTypeRegex     CompareType = "REGEX"
	CompareTypeStartWith CompareType = "STARTS_WITH"
)

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts struct {
	// Name of the L7 policy.
	Name string `json:"name,omitempty"`

	// The ID of the listener.
	ListenerID string `json:"listener_id" required:"true"`

	// The L7 policy action. One of REDIRECT_TO_POOL, REDIRECT_TO_URL, or REJECT.
	Action Action `json:"action" required:"true"`

	// The position of this policy on the listener.
	Position int32 `json:"position,omitempty"`

	// A human-readable description for the resource.
	Description string `json:"description,omitempty"`

	// TenantID is the UUID of the tenant who owns the L7 policy in octavia.
	// Only administrative users can specify a project UUID other than their own.
	TenantID string `json:"tenant_id,omitempty"`

	// Requests matching this policy will be redirected to the pool with this ID.
	// Only valid if action is REDIRECT_TO_POOL.
	RedirectPoolID string `json:"redirect_pool_id,omitempty"`

	// Requests matching this policy will be redirected to this URL.
	// Only valid if action is REDIRECT_TO_URL.
	RedirectURL string `json:"redirect_url,omitempty"`

	// The administrative state of the Loadbalancer. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

// ToL7PolicyCreateMap builds a request body from CreateOpts.
func (opts CreateOpts) ToL7PolicyCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "l7policy")
}

// Create accepts a CreateOpts struct and uses the values to create a new l7policy.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToL7PolicyCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(rootURL(c), b, &r.Body, nil)
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToL7PolicyListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API.
type ListOpts struct {
	Name           string `q:"name"`
	Description    string `q:"description"`
	ListenerID     string `q:"listener_id"`
	Action         string `q:"action"`
	TenantID       string `q:"tenant_id"`
	RedirectPoolID string `q:"redirect_pool_id"`
	RedirectURL    string `q:"redirect_url"`
	Position       int32  `q:"position"`
	AdminStateUp   bool   `q:"admin_state_up"`
	ID             string `q:"id"`
	Limit          int    `q:"limit"`
	Marker         string `q:"marker"`
	SortKey        string `q:"sort_key"`
	SortDir        string `q:"sort_dir"`
}

// ToL7PolicyListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToL7PolicyListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns a Pager which allows you to iterate over a collection of
// l7policies. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those l7policies that are owned by the
// project who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(c)
	if opts != nil {
		query, err := opts.ToL7PolicyListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return L7PolicyPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves a particular l7policy based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(resourceURL(c, id), &r.Body, nil)
	return
}

// Delete will permanently delete a particular l7policy based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(resourceURL(c, id), nil)
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToL7PolicyUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts is the common options struct used in this package's Update
// operation.
type UpdateOpts struct {
	// Name of the L7 policy, empty string is allowed.
	Name *string `json:"name,omitempty"`

	// The L7 policy action. One of REDIRECT_TO_POOL, REDIRECT_TO_URL, or REJECT.
	Action Action `json:"action,omitempty"`

	// The position of this policy on the listener.
	Position int32 `json:"position,omitempty"`

	// A human-readable description for the resource, empty string is allowed.
	Description *string `json:"description,omitempty"`

	// Requests matching this policy will be redirected to the pool with this ID.
	// Only valid if action is REDIRECT_TO_POOL.
	RedirectPoolID *string `json:"redirect_pool_id,omitempty"`

	// Requests matching this policy will be redirected to this URL.
	// Only valid if action is REDIRECT_TO_URL.
	RedirectURL *string `json:"redirect_url,omitempty"`

	// The administrative state of the Loadbalancer. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

// ToL7PolicyUpdateMap builds a request body from UpdateOpts.
func (opts UpdateOpts) ToL7PolicyUpdateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "l7policy")
	if err != nil {
		return nil, err
	}

	m := b["l7policy"].(map[string]interface{})

	if m["redirect_pool_id"] == "" {
		m["redirect_pool_id"] = nil
	}

	if m["redirect_url"] == "" {
		m["redirect_url"] = nil
	}

	return b, nil
}

// Update allows l7policy to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToL7PolicyUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(resourceURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// CreateRuleOpts is the common options struct used in this package's CreateRule
// operation.
type CreateRuleOpts struct {
	// The L7 rule type. One of COOKIE, FILE_TYPE, HEADER, HOST_NAME, or PATH.
	RuleType RuleType `json:"type" required:"true"`

	// The comparison type for the L7 rule. One of CONTAINS, ENDS_WITH, EQUAL_TO, REGEX, or STARTS_WITH.
	CompareType CompareType `json:"compare_type" required:"true"`

	// The value to use for the comparison. For example, the file type to compare.
	Value string `json:"value" required:"true"`

	// TenantID is the UUID of the tenant who owns the rule in octavia.
	// Only administrative users can specify a project UUID other than their own.
	TenantID string `json:"tenant_id,omitempty"`

	// The key to use for the comparison. For example, the name of the cookie to evaluate.
	Key string `json:"key,omitempty"`

	// When true the logic of the rule is inverted. For example, with invert true,
	// equal to would become not equal to. Default is false.
	Invert bool `json:"invert,omitempty"`

	// The administrative state of the Loadbalancer. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

// ToRuleCreateMap builds a request body from CreateRuleOpts.
func (opts CreateRuleOpts) ToRuleCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "rule")
}

// CreateRule will create and associate a Rule with a particular L7Policy.
func CreateRule(c *gophercloud.ServiceClient, policyID string, opts CreateRuleOpts) (r CreateRuleResult) {
	b, err := opts.ToRuleCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(ruleRootURL(c, policyID), b, &r.Body, nil)
	return
}

// ListRulesOptsBuilder allows extensions to add additional parameters to the
// ListRules request.
type ListRulesOptsBuilder interface {
	ToRulesListQuery() (string, error)
}

// ListRulesOpts allows the filtering and sorting of paginated collections
// through the API.
type ListRulesOpts struct {
	RuleType     RuleType    `q:"type"`
	TenantID     string      `q:"tenant_id"`
	CompareType  CompareType `q:"compare_type"`
	Value        string      `q:"value"`
	Key          string      `q:"key"`
	Invert       bool        `q:"invert"`
	AdminStateUp bool        `q:"admin_state_up"`
	ID           string      `q:"id"`
	Limit        int         `q:"limit"`
	Marker       string      `q:"marker"`
	SortKey      string      `q:"sort_key"`
	SortDir      string      `q:"sort_dir"`
}

// ToRulesListQuery formats a ListOpts into a query string.
func (opts ListRulesOpts) ToRulesListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// ListRules returns a Pager which allows you to iterate over a collection of
// rules. It accepts a ListRulesOptsBuilder, which allows you to filter and
// sort the returned collection for greater efficiency.
//
// Default policy settings return only those rules that are owned by the
// project who submits the request, unless an admin user submits the request.
func ListRules(c *gophercloud.ServiceClient, policyID string, opts ListRulesOptsBuilder) pagination.Pager {
	url := ruleRootURL(c, policyID)
	if opts != nil {
		query, err := opts.ToRulesListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return RulePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// GetRule retrieves a particular L7Policy Rule based on its unique ID.
func GetRule(c *gophercloud.ServiceClient, policyID string, ruleID string) (r GetRuleResult) {
	_, r.Err = c.Get(ruleResourceURL(c, policyID, ruleID), &r.Body, nil)
	return
}

// DeleteRule will remove a Rule from a particular L7Policy.
func DeleteRule(c *gophercloud.ServiceClient, policyID string, ruleID string) (r DeleteRuleResult) {
	_, r.Err = c.Delete(ruleResourceURL(c, policyID, ruleID), nil)
	return
}

// UpdateRuleOptsBuilder allows to add additional parameters to the PUT request.
type UpdateRuleOptsBuilder interface {
	ToRuleUpdateMap() (map[string]interface{}, error)
}

// UpdateRuleOpts is the common options struct used in this package's Update
// operation.
type UpdateRuleOpts struct {
	// The L7 rule type. One of COOKIE, FILE_TYPE, HEADER, HOST_NAME, or PATH.
	RuleType RuleType `json:"type,omitempty"`

	// The comparison type for the L7 rule. One of CONTAINS, ENDS_WITH, EQUAL_TO, REGEX, or STARTS_WITH.
	CompareType CompareType `json:"compare_type,omitempty"`

	// The value to use for the comparison. For example, the file type to compare.
	Value string `json:"value,omitempty"`

	// The key to use for the comparison. For example, the name of the cookie to evaluate.
	Key *string `json:"key,omitempty"`

	// When true the logic of the rule is inverted. For example, with invert true,
	// equal to would become not equal to. Default is false.
	Invert *bool `json:"invert,omitempty"`

	// The administrative state of the Loadbalancer. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool `json:"admin_state_up,omitempty"`
}

// ToRuleUpdateMap builds a request body from UpdateRuleOpts.
func (opts UpdateRuleOpts) ToRuleUpdateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "rule")
	if err != nil {
		return nil, err
	}

	if m := b["rule"].(map[string]interface{}); m["key"] == "" {
		m["key"] = nil
	}

	return b, nil
}

// UpdateRule allows Rule to be updated.
func UpdateRule(c *gophercloud.ServiceClient, policyID string, ruleID string, opts UpdateRuleOptsBuilder) (r UpdateRuleResult) {
	b, err := opts.ToRuleUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(ruleResourceURL(c, policyID, ruleID), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return
}
