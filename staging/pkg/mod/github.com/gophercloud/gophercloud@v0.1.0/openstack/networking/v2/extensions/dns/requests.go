package dns

import (
	"net/url"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

// PortListOptsExt adds the DNS options to the base port ListOpts.
type PortListOptsExt struct {
	ports.ListOptsBuilder

	DNSName string `q:"dns_name"`
}

// ToPortListQuery adds the DNS options to the base port list options.
func (opts PortListOptsExt) ToPortListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts.ListOptsBuilder)
	if err != nil {
		return "", err
	}

	params := q.Query()

	if opts.DNSName != "" {
		params.Add("dns_name", opts.DNSName)
	}

	q = &url.URL{RawQuery: params.Encode()}
	return q.String(), err
}

// PortCreateOptsExt adds port DNS options to the base ports.CreateOpts.
type PortCreateOptsExt struct {
	// CreateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Create operation in this package.
	ports.CreateOptsBuilder

	// Set DNS name to the port
	DNSName string `json:"dns_name,omitempty"`
}

// ToPortCreateMap casts a CreateOpts struct to a map.
func (opts PortCreateOptsExt) ToPortCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToPortCreateMap()
	if err != nil {
		return nil, err
	}

	port := base["port"].(map[string]interface{})

	if opts.DNSName != "" {
		port["dns_name"] = opts.DNSName
	}

	return base, nil
}

// PortUpdateOptsExt adds DNS options to the base ports.UpdateOpts
type PortUpdateOptsExt struct {
	// UpdateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Update operation in this package.
	ports.UpdateOptsBuilder

	// Set DNS name to the port
	DNSName *string `json:"dns_name,omitempty"`
}

// ToPortUpdateMap casts an UpdateOpts struct to a map.
func (opts PortUpdateOptsExt) ToPortUpdateMap() (map[string]interface{}, error) {
	base, err := opts.UpdateOptsBuilder.ToPortUpdateMap()
	if err != nil {
		return nil, err
	}

	port := base["port"].(map[string]interface{})

	if opts.DNSName != nil {
		port["dns_name"] = *opts.DNSName
	}

	return base, nil
}

// FloatingIPCreateOptsExt adds floating IP DNS options to the base floatingips.CreateOpts.
type FloatingIPCreateOptsExt struct {
	// CreateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Create operation in this package.
	floatingips.CreateOptsBuilder

	// Set DNS name to the floating IPs
	DNSName string `json:"dns_name,omitempty"`

	// Set DNS domain to the floating IPs
	DNSDomain string `json:"dns_domain,omitempty"`
}

// ToFloatingIPCreateMap casts a CreateOpts struct to a map.
func (opts FloatingIPCreateOptsExt) ToFloatingIPCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToFloatingIPCreateMap()
	if err != nil {
		return nil, err
	}

	floatingip := base["floatingip"].(map[string]interface{})

	if opts.DNSName != "" {
		floatingip["dns_name"] = opts.DNSName
	}

	if opts.DNSDomain != "" {
		floatingip["dns_domain"] = opts.DNSDomain
	}

	return base, nil
}

// NetworkCreateOptsExt adds network DNS options to the base networks.CreateOpts.
type NetworkCreateOptsExt struct {
	// CreateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Create operation in this package.
	networks.CreateOptsBuilder

	// Set DNS domain to the network
	DNSDomain string `json:"dns_domain,omitempty"`
}

// ToNetworkCreateMap casts a CreateOpts struct to a map.
func (opts NetworkCreateOptsExt) ToNetworkCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToNetworkCreateMap()
	if err != nil {
		return nil, err
	}

	network := base["network"].(map[string]interface{})

	if opts.DNSDomain != "" {
		network["dns_domain"] = opts.DNSDomain
	}

	return base, nil
}

// NetworkUpdateOptsExt adds network DNS options to the base networks.UpdateOpts
type NetworkUpdateOptsExt struct {
	// UpdateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Update operation in this package.
	networks.UpdateOptsBuilder

	// Set DNS domain to the network
	DNSDomain *string `json:"dns_domain,omitempty"`
}

// ToNetworkUpdateMap casts an UpdateOpts struct to a map.
func (opts NetworkUpdateOptsExt) ToNetworkUpdateMap() (map[string]interface{}, error) {
	base, err := opts.UpdateOptsBuilder.ToNetworkUpdateMap()
	if err != nil {
		return nil, err
	}

	network := base["network"].(map[string]interface{})

	if opts.DNSDomain != nil {
		network["dns_domain"] = *opts.DNSDomain
	}

	return base, nil
}
