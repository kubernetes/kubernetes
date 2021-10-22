package portsecurity

import (
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

// PortCreateOptsExt adds port security options to the base ports.CreateOpts.
type PortCreateOptsExt struct {
	ports.CreateOptsBuilder

	// PortSecurityEnabled toggles port security on a port.
	PortSecurityEnabled *bool `json:"port_security_enabled,omitempty"`
}

// ToPortCreateMap casts a CreateOpts struct to a map.
func (opts PortCreateOptsExt) ToPortCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToPortCreateMap()
	if err != nil {
		return nil, err
	}

	port := base["port"].(map[string]interface{})

	if opts.PortSecurityEnabled != nil {
		port["port_security_enabled"] = &opts.PortSecurityEnabled
	}

	return base, nil
}

// PortUpdateOptsExt adds port security options to the base ports.UpdateOpts.
type PortUpdateOptsExt struct {
	ports.UpdateOptsBuilder

	// PortSecurityEnabled toggles port security on a port.
	PortSecurityEnabled *bool `json:"port_security_enabled,omitempty"`
}

// ToPortUpdateMap casts a UpdateOpts struct to a map.
func (opts PortUpdateOptsExt) ToPortUpdateMap() (map[string]interface{}, error) {
	base, err := opts.UpdateOptsBuilder.ToPortUpdateMap()
	if err != nil {
		return nil, err
	}

	port := base["port"].(map[string]interface{})

	if opts.PortSecurityEnabled != nil {
		port["port_security_enabled"] = &opts.PortSecurityEnabled
	}

	return base, nil
}

// NetworkCreateOptsExt adds port security options to the base
// networks.CreateOpts.
type NetworkCreateOptsExt struct {
	networks.CreateOptsBuilder

	// PortSecurityEnabled toggles port security on a port.
	PortSecurityEnabled *bool `json:"port_security_enabled,omitempty"`
}

// ToNetworkCreateMap casts a CreateOpts struct to a map.
func (opts NetworkCreateOptsExt) ToNetworkCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToNetworkCreateMap()
	if err != nil {
		return nil, err
	}

	network := base["network"].(map[string]interface{})

	if opts.PortSecurityEnabled != nil {
		network["port_security_enabled"] = &opts.PortSecurityEnabled
	}

	return base, nil
}

// NetworkUpdateOptsExt adds port security options to the base
// networks.UpdateOpts.
type NetworkUpdateOptsExt struct {
	networks.UpdateOptsBuilder

	// PortSecurityEnabled toggles port security on a port.
	PortSecurityEnabled *bool `json:"port_security_enabled,omitempty"`
}

// ToNetworkUpdateMap casts a UpdateOpts struct to a map.
func (opts NetworkUpdateOptsExt) ToNetworkUpdateMap() (map[string]interface{}, error) {
	base, err := opts.UpdateOptsBuilder.ToNetworkUpdateMap()
	if err != nil {
		return nil, err
	}

	network := base["network"].(map[string]interface{})

	if opts.PortSecurityEnabled != nil {
		network["port_security_enabled"] = &opts.PortSecurityEnabled
	}

	return base, nil
}
