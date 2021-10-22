package portsbinding

import (
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

// CreateOptsExt adds port binding options to the base ports.CreateOpts.
type CreateOptsExt struct {
	// CreateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Create operation in this package.
	ports.CreateOptsBuilder

	// The ID of the host where the port is allocated
	HostID string `json:"binding:host_id,omitempty"`

	// The virtual network interface card (vNIC) type that is bound to the
	// neutron port.
	VNICType string `json:"binding:vnic_type,omitempty"`

	// A dictionary that enables the application running on the specified
	// host to pass and receive virtual network interface (VIF) port-specific
	// information to the plug-in.
	Profile map[string]interface{} `json:"binding:profile,omitempty"`
}

// ToPortCreateMap casts a CreateOpts struct to a map.
func (opts CreateOptsExt) ToPortCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToPortCreateMap()
	if err != nil {
		return nil, err
	}

	port := base["port"].(map[string]interface{})

	if opts.HostID != "" {
		port["binding:host_id"] = opts.HostID
	}

	if opts.VNICType != "" {
		port["binding:vnic_type"] = opts.VNICType
	}

	if opts.Profile != nil {
		port["binding:profile"] = opts.Profile
	}

	return base, nil
}

// UpdateOptsExt adds port binding options to the base ports.UpdateOpts
type UpdateOptsExt struct {
	// UpdateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Update operation in this package.
	ports.UpdateOptsBuilder

	// The ID of the host where the port is allocated.
	HostID *string `json:"binding:host_id,omitempty"`

	// The virtual network interface card (vNIC) type that is bound to the
	// neutron port.
	VNICType string `json:"binding:vnic_type,omitempty"`

	// A dictionary that enables the application running on the specified
	// host to pass and receive virtual network interface (VIF) port-specific
	// information to the plug-in.
	Profile map[string]interface{} `json:"binding:profile,omitempty"`
}

// ToPortUpdateMap casts an UpdateOpts struct to a map.
func (opts UpdateOptsExt) ToPortUpdateMap() (map[string]interface{}, error) {
	base, err := opts.UpdateOptsBuilder.ToPortUpdateMap()
	if err != nil {
		return nil, err
	}

	port := base["port"].(map[string]interface{})

	if opts.HostID != nil {
		port["binding:host_id"] = *opts.HostID
	}

	if opts.VNICType != "" {
		port["binding:vnic_type"] = opts.VNICType
	}

	if opts.Profile != nil {
		if len(opts.Profile) == 0 {
			// send null instead of the empty json object ("{}")
			port["binding:profile"] = nil
		} else {
			port["binding:profile"] = opts.Profile
		}
	}

	return base, nil
}
