package rules

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Binary gives users a solid type to work with for create and update
// operations. It is recommended that users use the `Yes` and `No` enums
type Binary *bool

// Convenience vars for Enabled and Shared values.
var (
	iTrue         = true
	iFalse        = false
	Yes    Binary = &iTrue
	No     Binary = &iFalse
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToRuleListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the Firewall rule attributes you want to see returned. SortKey allows you to
// sort by a particular firewall rule attribute. SortDir sets the direction, and is
// either `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	TenantID             string `q:"tenant_id"`
	Name                 string `q:"name"`
	Description          string `q:"description"`
	Protocol             string `q:"protocol"`
	Action               string `q:"action"`
	IPVersion            int    `q:"ip_version"`
	SourceIPAddress      string `q:"source_ip_address"`
	DestinationIPAddress string `q:"destination_ip_address"`
	SourcePort           string `q:"source_port"`
	DestinationPort      string `q:"destination_port"`
	Enabled              bool   `q:"enabled"`
	ID                   string `q:"id"`
	Limit                int    `q:"limit"`
	Marker               string `q:"marker"`
	SortKey              string `q:"sort_key"`
	SortDir              string `q:"sort_dir"`
}

// ToRuleListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToRuleListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List returns a Pager which allows you to iterate over a collection of
// firewall rules. It accepts a ListOpts struct, which allows you to filter
// and sort the returned collection for greater efficiency.
//
// Default policy settings return only those firewall rules that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(c)

	if opts != nil {
		query, err := opts.ToRuleListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return RulePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToRuleCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains all the values needed to create a new firewall rule.
type CreateOpts struct {
	// Mandatory for create
	Protocol string
	Action   string
	// Optional
	TenantID             string
	Name                 string
	Description          string
	IPVersion            int
	SourceIPAddress      string
	DestinationIPAddress string
	SourcePort           string
	DestinationPort      string
	Shared               *bool
	Enabled              *bool
}

// ToRuleCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToRuleCreateMap() (map[string]interface{}, error) {
	if opts.Protocol == "" {
		return nil, errProtocolRequired
	}

	if opts.Action == "" {
		return nil, errActionRequired
	}

	r := make(map[string]interface{})

	r["protocol"] = opts.Protocol
	r["action"] = opts.Action

	if opts.TenantID != "" {
		r["tenant_id"] = opts.TenantID
	}
	if opts.Name != "" {
		r["name"] = opts.Name
	}
	if opts.Description != "" {
		r["description"] = opts.Description
	}
	if opts.IPVersion != 0 {
		r["ip_version"] = opts.IPVersion
	}
	if opts.SourceIPAddress != "" {
		r["source_ip_address"] = opts.SourceIPAddress
	}
	if opts.DestinationIPAddress != "" {
		r["destination_ip_address"] = opts.DestinationIPAddress
	}
	if opts.SourcePort != "" {
		r["source_port"] = opts.SourcePort
	}
	if opts.DestinationPort != "" {
		r["destination_port"] = opts.DestinationPort
	}
	if opts.Shared != nil {
		r["shared"] = *opts.Shared
	}
	if opts.Enabled != nil {
		r["enabled"] = *opts.Enabled
	}

	return map[string]interface{}{"firewall_rule": r}, nil
}

// Create accepts a CreateOpts struct and uses the values to create a new firewall rule
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToRuleCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular firewall rule based on its unique ID.
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
	ToRuleUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the values used when updating a firewall rule.
// Optional
type UpdateOpts struct {
	Protocol             string
	Action               string
	Name                 string
	Description          string
	IPVersion            int
	SourceIPAddress      *string
	DestinationIPAddress *string
	SourcePort           *string
	DestinationPort      *string
	Shared               *bool
	Enabled              *bool
}

// ToRuleUpdateMap casts a UpdateOpts struct to a map.
func (opts UpdateOpts) ToRuleUpdateMap() (map[string]interface{}, error) {
	r := make(map[string]interface{})

	if opts.Protocol != "" {
		r["protocol"] = opts.Protocol
	}
	if opts.Action != "" {
		r["action"] = opts.Action
	}
	if opts.Name != "" {
		r["name"] = opts.Name
	}
	if opts.Description != "" {
		r["description"] = opts.Description
	}
	if opts.IPVersion != 0 {
		r["ip_version"] = opts.IPVersion
	}
	if opts.SourceIPAddress != nil {
		s := *opts.SourceIPAddress
		if s == "" {
			r["source_ip_address"] = nil
		} else {
			r["source_ip_address"] = s
		}
	}
	if opts.DestinationIPAddress != nil {
		s := *opts.DestinationIPAddress
		if s == "" {
			r["destination_ip_address"] = nil
		} else {
			r["destination_ip_address"] = s
		}
	}
	if opts.SourcePort != nil {
		s := *opts.SourcePort
		if s == "" {
			r["source_port"] = nil
		} else {
			r["source_port"] = s
		}
	}
	if opts.DestinationPort != nil {
		s := *opts.DestinationPort
		if s == "" {
			r["destination_port"] = nil
		} else {
			r["destination_port"] = s
		}
	}
	if opts.Shared != nil {
		r["shared"] = *opts.Shared
	}
	if opts.Enabled != nil {
		r["enabled"] = *opts.Enabled
	}

	return map[string]interface{}{"firewall_rule": r}, nil
}

// Update allows firewall policies to be updated.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToRuleUpdateMap()
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

// Delete will permanently delete a particular firewall rule based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}
