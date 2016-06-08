package loadbalancers

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// AdminState gives users a solid type to work with for create and update
// operations. It is recommended that users use the `Up` and `Down` enums.
type AdminState *bool

type loadbalancerOpts struct {
	// Optional. Human-readable name for the Loadbalancer. Does not have to be unique.
	Name string

	// Optional. Human-readable description for the Loadbalancer.
	Description string

	// Required. The network on which to allocate the Loadbalancer's address. A tenant can
	// only create Loadbalancers on networks authorized by policy (e.g. networks that
	// belong to them or networks that are shared).
	VipSubnetID string

	// Required for admins. The UUID of the tenant who owns the Loadbalancer.
	// Only administrative users can specify a tenant UUID other than their own.
	TenantID string

	// Optional. The IP address of the Loadbalancer.
	VipAddress string

	// Optional. The administrative state of the Loadbalancer. A valid value is true (UP)
	// or false (DOWN).
	AdminStateUp *bool

	// Optional. The UUID of a flavor.
	Flavor string

	// Optional. The name of the provider.
	Provider string
}

// Convenience vars for AdminStateUp values.
var (
	iTrue  = true
	iFalse = false

	Up   AdminState = &iTrue
	Down AdminState = &iFalse
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToLoadbalancerListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the Loadbalancer attributes you want to see returned. SortKey allows you to
// sort by a particular attribute. SortDir sets the direction, and is
// either `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	Description        string `q:"description"`
	AdminStateUp       *bool  `q:"admin_state_up"`
	TenantID           string `q:"tenant_id"`
	ProvisioningStatus string `q:"provisioning_status"`
	VipAddress         string `q:"vip_address"`
	VipSubnetID        string `q:"vip_subnet_id"`
	ID                 string `q:"id"`
	OperatingStatus    string `q:"operating_status"`
	Name               string `q:"name"`
	Flavor             string `q:"flavor"`
	Provider           string `q:"provider"`
	Limit              int    `q:"limit"`
	Marker             string `q:"marker"`
	SortKey            string `q:"sort_key"`
	SortDir            string `q:"sort_dir"`
}

// ToLoadbalancerListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToLoadbalancerListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List returns a Pager which allows you to iterate over a collection of
// routers. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those routers that are owned by the
// tenant who submits the request, unless an admin user submits the request.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := rootURL(c)
	if opts != nil {
		query, err := opts.ToLoadbalancerListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return LoadbalancerPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

var (
	errVipSubnetIDRequried = fmt.Errorf("VipSubnetID is required")
)

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToLoadbalancerCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts loadbalancerOpts

// ToLoadbalancerCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToLoadbalancerCreateMap() (map[string]interface{}, error) {
	l := make(map[string]interface{})

	if opts.VipSubnetID != "" {
		l["vip_subnet_id"] = opts.VipSubnetID
	} else {
		return nil, errVipSubnetIDRequried
	}
	if opts.AdminStateUp != nil {
		l["admin_state_up"] = &opts.AdminStateUp
	}
	if opts.Name != "" {
		l["name"] = opts.Name
	}
	if opts.TenantID != "" {
		l["tenant_id"] = opts.TenantID
	}
	if opts.Description != "" {
		l["description"] = opts.Description
	}
	if opts.VipAddress != "" {
		l["vip_address"] = opts.VipAddress
	}
	if opts.Flavor != "" {
		l["flavor"] = opts.Flavor
	}
	if opts.Provider != "" {
		l["provider"] = opts.Provider
	}

	return map[string]interface{}{"loadbalancer": l}, nil
}

// Create is an operation which provisions a new loadbalancer based on the
// configuration defined in the CreateOpts struct. Once the request is
// validated and progress has started on the provisioning process, a
// CreateResult will be returned.
//
// Users with an admin role can create loadbalancers on behalf of other tenants by
// specifying a TenantID attribute different than their own.
func Create(c *gophercloud.ServiceClient, opts CreateOpts) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToLoadbalancerCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Post(rootURL(c), reqBody, &res.Body, nil)
	return res
}

// Get retrieves a particular Loadbalancer based on its unique ID.
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
	ToLoadbalancerUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts is the common options struct used in this package's Update
// operation.
type UpdateOpts loadbalancerOpts

// ToLoadbalancerUpdateMap casts a UpdateOpts struct to a map.
func (opts UpdateOpts) ToLoadbalancerUpdateMap() (map[string]interface{}, error) {
	l := make(map[string]interface{})

	if opts.Name != "" {
		l["name"] = opts.Name
	}
	if opts.Description != "" {
		l["description"] = opts.Description
	}
	if opts.AdminStateUp != nil {
		l["admin_state_up"] = &opts.AdminStateUp
	}

	return map[string]interface{}{"loadbalancer": l}, nil
}

// Update is an operation which modifies the attributes of the specified Loadbalancer.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOpts) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToLoadbalancerUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Put(resourceURL(c, id), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 202},
	})

	return res
}

// Delete will permanently delete a particular Loadbalancer based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(resourceURL(c, id), nil)
	return res
}

func GetStatuses(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(statusRootURL(c, id), &res.Body, nil)
	return res
}
