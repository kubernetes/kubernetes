package subnets

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
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
	Name            string `q:"name"`
	Description     string `q:"description"`
	EnableDHCP      *bool  `q:"enable_dhcp"`
	NetworkID       string `q:"network_id"`
	TenantID        string `q:"tenant_id"`
	ProjectID       string `q:"project_id"`
	IPVersion       int    `q:"ip_version"`
	GatewayIP       string `q:"gateway_ip"`
	CIDR            string `q:"cidr"`
	IPv6AddressMode string `q:"ipv6_address_mode"`
	IPv6RAMode      string `q:"ipv6_ra_mode"`
	ID              string `q:"id"`
	SubnetPoolID    string `q:"subnetpool_id"`
	Limit           int    `q:"limit"`
	Marker          string `q:"marker"`
	SortKey         string `q:"sort_key"`
	SortDir         string `q:"sort_dir"`
	Tags            string `q:"tags"`
	TagsAny         string `q:"tags-any"`
	NotTags         string `q:"not-tags"`
	NotTagsAny      string `q:"not-tags-any"`
}

// ToSubnetListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToSubnetListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
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
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(getURL(c, id), &r.Body, nil)
	return
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// List request.
type CreateOptsBuilder interface {
	ToSubnetCreateMap() (map[string]interface{}, error)
}

// CreateOpts represents the attributes used when creating a new subnet.
type CreateOpts struct {
	// NetworkID is the UUID of the network the subnet will be associated with.
	NetworkID string `json:"network_id" required:"true"`

	// CIDR is the address CIDR of the subnet.
	CIDR string `json:"cidr,omitempty"`

	// Name is a human-readable name of the subnet.
	Name string `json:"name,omitempty"`

	// Description of the subnet.
	Description string `json:"description,omitempty"`

	// The UUID of the project who owns the Subnet. Only administrative users
	// can specify a project UUID other than their own.
	TenantID string `json:"tenant_id,omitempty"`

	// The UUID of the project who owns the Subnet. Only administrative users
	// can specify a project UUID other than their own.
	ProjectID string `json:"project_id,omitempty"`

	// AllocationPools are IP Address pools that will be available for DHCP.
	AllocationPools []AllocationPool `json:"allocation_pools,omitempty"`

	// GatewayIP sets gateway information for the subnet. Setting to nil will
	// cause a default gateway to automatically be created. Setting to an empty
	// string will cause the subnet to be created with no gateway. Setting to
	// an explicit address will set that address as the gateway.
	GatewayIP *string `json:"gateway_ip,omitempty"`

	// IPVersion is the IP version for the subnet.
	IPVersion gophercloud.IPVersion `json:"ip_version,omitempty"`

	// EnableDHCP will either enable to disable the DHCP service.
	EnableDHCP *bool `json:"enable_dhcp,omitempty"`

	// DNSNameservers are the nameservers to be set via DHCP.
	DNSNameservers []string `json:"dns_nameservers,omitempty"`

	// HostRoutes are any static host routes to be set via DHCP.
	HostRoutes []HostRoute `json:"host_routes,omitempty"`

	// The IPv6 address modes specifies mechanisms for assigning IPv6 IP addresses.
	IPv6AddressMode string `json:"ipv6_address_mode,omitempty"`

	// The IPv6 router advertisement specifies whether the networking service
	// should transmit ICMPv6 packets.
	IPv6RAMode string `json:"ipv6_ra_mode,omitempty"`

	// SubnetPoolID is the id of the subnet pool that subnet should be associated to.
	SubnetPoolID string `json:"subnetpool_id,omitempty"`

	// Prefixlen is used when user creates a subnet from the subnetpool. It will
	// overwrite the "default_prefixlen" value of the referenced subnetpool.
	Prefixlen int `json:"prefixlen,omitempty"`
}

// ToSubnetCreateMap builds a request body from CreateOpts.
func (opts CreateOpts) ToSubnetCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "subnet")
	if err != nil {
		return nil, err
	}

	if m := b["subnet"].(map[string]interface{}); m["gateway_ip"] == "" {
		m["gateway_ip"] = nil
	}

	return b, nil
}

// Create accepts a CreateOpts struct and creates a new subnet using the values
// provided. You must remember to provide a valid NetworkID, CIDR and IP
// version.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToSubnetCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(createURL(c), b, &r.Body, nil)
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToSubnetUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents the attributes used when updating an existing subnet.
type UpdateOpts struct {
	// Name is a human-readable name of the subnet.
	Name *string `json:"name,omitempty"`

	// Description of the subnet.
	Description *string `json:"description,omitempty"`

	// AllocationPools are IP Address pools that will be available for DHCP.
	AllocationPools []AllocationPool `json:"allocation_pools,omitempty"`

	// GatewayIP sets gateway information for the subnet. Setting to nil will
	// cause a default gateway to automatically be created. Setting to an empty
	// string will cause the subnet to be created with no gateway. Setting to
	// an explicit address will set that address as the gateway.
	GatewayIP *string `json:"gateway_ip,omitempty"`

	// DNSNameservers are the nameservers to be set via DHCP.
	DNSNameservers *[]string `json:"dns_nameservers,omitempty"`

	// HostRoutes are any static host routes to be set via DHCP.
	HostRoutes *[]HostRoute `json:"host_routes,omitempty"`

	// EnableDHCP will either enable to disable the DHCP service.
	EnableDHCP *bool `json:"enable_dhcp,omitempty"`
}

// ToSubnetUpdateMap builds a request body from UpdateOpts.
func (opts UpdateOpts) ToSubnetUpdateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "subnet")
	if err != nil {
		return nil, err
	}

	if m := b["subnet"].(map[string]interface{}); m["gateway_ip"] == "" {
		m["gateway_ip"] = nil
	}

	return b, nil
}

// Update accepts a UpdateOpts struct and updates an existing subnet using the
// values provided.
func Update(c *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToSubnetUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(updateURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})
	return
}

// Delete accepts a unique ID and deletes the subnet associated with it.
func Delete(c *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = c.Delete(deleteURL(c, id), nil)
	return
}

// IDFromName is a convenience function that returns a subnet's ID,
// given its name.
func IDFromName(client *gophercloud.ServiceClient, name string) (string, error) {
	count := 0
	id := ""

	listOpts := ListOpts{
		Name: name,
	}

	pages, err := List(client, listOpts).AllPages()
	if err != nil {
		return "", err
	}

	all, err := ExtractSubnets(pages)
	if err != nil {
		return "", err
	}

	for _, s := range all {
		if s.Name == name {
			count++
			id = s.ID
		}
	}

	switch count {
	case 0:
		return "", gophercloud.ErrResourceNotFound{Name: name, ResourceType: "subnet"}
	case 1:
		return id, nil
	default:
		return "", gophercloud.ErrMultipleResourcesFound{Name: name, Count: count, ResourceType: "subnet"}
	}
}
