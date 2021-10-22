package extradhcpopts

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

// CreateOptsExt adds extra DHCP options to the base ports.CreateOpts.
type CreateOptsExt struct {
	// CreateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Create operation in this package.
	ports.CreateOptsBuilder

	// ExtraDHCPOpts field is a set of DHCP options for a single port.
	ExtraDHCPOpts []CreateExtraDHCPOpt `json:"extra_dhcp_opts,omitempty"`
}

// CreateExtraDHCPOpt represents the options required to create an extra DHCP
// option on a port.
type CreateExtraDHCPOpt struct {
	// OptName is the name of a DHCP option.
	OptName string `json:"opt_name" required:"true"`

	// OptValue is the value of the DHCP option.
	OptValue string `json:"opt_value" required:"true"`

	// IPVersion is the IP protocol version of a DHCP option.
	IPVersion gophercloud.IPVersion `json:"ip_version,omitempty"`
}

// ToPortCreateMap casts a CreateOptsExt struct to a map.
func (opts CreateOptsExt) ToPortCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToPortCreateMap()
	if err != nil {
		return nil, err
	}

	port := base["port"].(map[string]interface{})

	// Convert opts.ExtraDHCPOpts to a slice of maps.
	if opts.ExtraDHCPOpts != nil {
		extraDHCPOpts := make([]map[string]interface{}, len(opts.ExtraDHCPOpts))
		for i, opt := range opts.ExtraDHCPOpts {
			b, err := gophercloud.BuildRequestBody(opt, "")
			if err != nil {
				return nil, err
			}
			extraDHCPOpts[i] = b
		}
		port["extra_dhcp_opts"] = extraDHCPOpts
	}

	return base, nil
}

// UpdateOptsExt adds extra DHCP options to the base ports.UpdateOpts.
type UpdateOptsExt struct {
	// UpdateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Update operation in this package.
	ports.UpdateOptsBuilder

	// ExtraDHCPOpts field is a set of DHCP options for a single port.
	ExtraDHCPOpts []UpdateExtraDHCPOpt `json:"extra_dhcp_opts,omitempty"`
}

// UpdateExtraDHCPOpt represents the options required to update an extra DHCP
// option on a port.
type UpdateExtraDHCPOpt struct {
	// OptName is the name of a DHCP option.
	OptName string `json:"opt_name" required:"true"`

	// OptValue is the value of the DHCP option.
	OptValue *string `json:"opt_value"`

	// IPVersion is the IP protocol version of a DHCP option.
	IPVersion gophercloud.IPVersion `json:"ip_version,omitempty"`
}

// ToPortUpdateMap casts an UpdateOpts struct to a map.
func (opts UpdateOptsExt) ToPortUpdateMap() (map[string]interface{}, error) {
	base, err := opts.UpdateOptsBuilder.ToPortUpdateMap()
	if err != nil {
		return nil, err
	}

	port := base["port"].(map[string]interface{})

	// Convert opts.ExtraDHCPOpts to a slice of maps.
	if opts.ExtraDHCPOpts != nil {
		extraDHCPOpts := make([]map[string]interface{}, len(opts.ExtraDHCPOpts))
		for i, opt := range opts.ExtraDHCPOpts {
			b, err := gophercloud.BuildRequestBody(opt, "")
			if err != nil {
				return nil, err
			}
			extraDHCPOpts[i] = b
		}
		port["extra_dhcp_opts"] = extraDHCPOpts
	}

	return base, nil
}
