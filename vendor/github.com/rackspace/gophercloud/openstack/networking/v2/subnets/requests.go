package subnets

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// AdminState gives users a solid type to work with for create and update
// operations. It is recommended that users use the `Up` and `Down` enums.
type AdminState *bool

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
	ToSubnetListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the subnet attributes you want to see returned. SortKey allows you to sort
// by a particular subnet attribute. SortDir sets the direction, and is either
// `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	Name       string `q:"name"`
	EnableDHCP *bool  `q:"enable_dhcp"`
	NetworkID  string `q:"network_id"`
	TenantID   string `q:"tenant_id"`
	IPVersion  int    `q:"ip_version"`
	GatewayIP  string `q:"gateway_ip"`
	CIDR       string `q:"cidr"`
	ID         string `q:"id"`
	Limit      int    `q:"limit"`
	Marker     string `q:"marker"`
	SortKey    string `q:"sort_key"`
	SortDir    string `q:"sort_dir"`
}

// ToSubnetListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToSubnetListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List returns a Pager which allows you to iterate over a collection of
// subnets. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those subnets that are owned by the tenant
// who submits the request, unless the request is submitted by a user with
// administrative rights.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(c)
	if opts != nil {
		query, err := opts.ToSubnetListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return SubnetPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves a specific subnet based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) GetResult {
	var res GetResult
	_, res.Err = c.Get(getURL(c, id), &res.Body, nil)
	return res
}

// Valid IP types
const (
	IPv4 = 4
	IPv6 = 6
)

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToSubnetCreateMap() (map[string]interface{}, error)
}

// CreateOpts represents the attributes used when creating a new subnet.
type CreateOpts struct {
	// Required
	NetworkID string
	CIDR      string
	// Optional
	Name            string
	TenantID        string
	AllocationPools []AllocationPool
	GatewayIP       string
	NoGateway       bool
	IPVersion       int
	EnableDHCP      *bool
	DNSNameservers  []string
	HostRoutes      []HostRoute
}

// ToSubnetCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToSubnetCreateMap() (map[string]interface{}, error) {
	s := make(map[string]interface{})

	if opts.NetworkID == "" {
		return nil, errNetworkIDRequired
	}
	if opts.CIDR == "" {
		return nil, errCIDRRequired
	}
	if opts.IPVersion != 0 && opts.IPVersion != IPv4 && opts.IPVersion != IPv6 {
		return nil, errInvalidIPType
	}

	// Both GatewayIP and NoGateway should not be set
	if opts.GatewayIP != "" && opts.NoGateway {
		return nil, errInvalidGatewayConfig
	}

	s["network_id"] = opts.NetworkID
	s["cidr"] = opts.CIDR

	if opts.EnableDHCP != nil {
		s["enable_dhcp"] = &opts.EnableDHCP
	}
	if opts.Name != "" {
		s["name"] = opts.Name
	}
	if opts.GatewayIP != "" {
		s["gateway_ip"] = opts.GatewayIP
	} else if opts.NoGateway {
		s["gateway_ip"] = nil
	}
	if opts.TenantID != "" {
		s["tenant_id"] = opts.TenantID
	}
	if opts.IPVersion != 0 {
		s["ip_version"] = opts.IPVersion
	}
	if len(opts.AllocationPools) != 0 {
		s["allocation_pools"] = opts.AllocationPools
	}
	if len(opts.DNSNameservers) != 0 {
		s["dns_nameservers"] = opts.DNSNameservers
	}
	if len(opts.HostRoutes) != 0 {
		s["host_routes"] = opts.HostRoutes
	}

	return map[string]interface{}{"subnet": s}, nil
}

// Create accepts a CreateOpts struct and creates a new subnet using the values
// provided. You must remember to provide a valid NetworkID, CIDR and IP version.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToSubnetCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(createURL(c), reqBody, &res.Body, nil)
	return res
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToSubnetUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents the attributes used when updating an existing subnet.
type UpdateOpts struct {
	Name           string
	GatewayIP      string
	NoGateway      bool
	DNSNameservers []string
	HostRoutes     []HostRoute
	EnableDHCP     *bool
}

// ToSubnetUpdateMap casts an UpdateOpts struct to a map.
func (opts UpdateOpts) ToSubnetUpdateMap() (map[string]interface{}, error) {
	s := make(map[string]interface{})

	// Both GatewayIP and NoGateway should not be set
	if opts.GatewayIP != "" && opts.NoGateway {
		return nil, errInvalidGatewayConfig
	}

	if opts.EnableDHCP != nil {
		s["enable_dhcp"] = &opts.EnableDHCP
	}
	if opts.Name != "" {
		s["name"] = opts.Name
	}
	if opts.GatewayIP != "" {
		s["gateway_ip"] = opts.GatewayIP
	} else if opts.NoGateway {
		s["gateway_ip"] = nil
	}
	if opts.DNSNameservers != nil {
		s["dns_nameservers"] = opts.DNSNameservers
	}
	if opts.HostRoutes != nil {
		s["host_routes"] = opts.HostRoutes
	}

	return map[string]interface{}{"subnet": s}, nil
}

// Update accepts a UpdateOpts struct and updates an existing subnet using the
// values provided.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToSubnetUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Put(updateURL(c, id), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})

	return res
}

// Delete accepts a unique ID and deletes the subnet associated with it.
func Delete(c *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(deleteURL(c, id), nil)
	return res
}

// IDFromName is a convenience function that returns a subnet's ID given its name.
func IDFromName(client *gophercloud.ServiceClient, name string) (string, error) {
	subnetCount := 0
	subnetID := ""
	if name == "" {
		return "", fmt.Errorf("A subnet name must be provided.")
	}
	pager := List(client, nil)
	pager.EachPage(func(page pagination.Page) (bool, error) {
		subnetList, err := ExtractSubnets(page)
		if err != nil {
			return false, err
		}

		for _, s := range subnetList {
			if s.Name == name {
				subnetCount++
				subnetID = s.ID
			}
		}
		return true, nil
	})

	switch subnetCount {
	case 0:
		return "", fmt.Errorf("Unable to find subnet: %s", name)
	case 1:
		return subnetID, nil
	default:
		return "", fmt.Errorf("Found %d subnets matching %s", subnetCount, name)
	}
}
