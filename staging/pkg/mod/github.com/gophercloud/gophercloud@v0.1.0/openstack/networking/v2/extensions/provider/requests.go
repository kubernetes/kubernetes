package provider

import (
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
)

// CreateOptsExt adds a Segments option to the base Network CreateOpts.
type CreateOptsExt struct {
	networks.CreateOptsBuilder
	Segments []Segment `json:"segments,omitempty"`
}

// ToNetworkCreateMap adds segments to the base network creation options.
func (opts CreateOptsExt) ToNetworkCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToNetworkCreateMap()
	if err != nil {
		return nil, err
	}

	if opts.Segments == nil {
		return base, nil
	}

	providerMap := base["network"].(map[string]interface{})
	providerMap["segments"] = opts.Segments

	return base, nil
}
