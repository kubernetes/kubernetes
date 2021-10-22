package mtu

import (
	"fmt"
	"net/url"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
)

// ListOptsExt adds an MTU option to the base ListOpts.
type ListOptsExt struct {
	networks.ListOptsBuilder

	// The maximum transmission unit (MTU) value to address fragmentation.
	// Minimum value is 68 for IPv4, and 1280 for IPv6.
	MTU int `q:"mtu"`
}

// ToNetworkListQuery adds the router:external option to the base network
// list options.
func (opts ListOptsExt) ToNetworkListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts.ListOptsBuilder)
	if err != nil {
		return "", err
	}

	params := q.Query()
	if opts.MTU > 0 {
		params.Add("mtu", fmt.Sprintf("%d", opts.MTU))
	}

	q = &url.URL{RawQuery: params.Encode()}
	return q.String(), err
}

// CreateOptsExt adds an MTU option to the base Network CreateOpts.
type CreateOptsExt struct {
	networks.CreateOptsBuilder

	// The maximum transmission unit (MTU) value to address fragmentation.
	// Minimum value is 68 for IPv4, and 1280 for IPv6.
	MTU int `json:"mtu,omitempty"`
}

// ToNetworkCreateMap adds an MTU to the base network creation options.
func (opts CreateOptsExt) ToNetworkCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToNetworkCreateMap()
	if err != nil {
		return nil, err
	}

	if opts.MTU == 0 {
		return base, nil
	}

	networkMap := base["network"].(map[string]interface{})
	networkMap["mtu"] = opts.MTU

	return base, nil
}

// CreateOptsExt adds an MTU option to the base Network UpdateOpts.
type UpdateOptsExt struct {
	networks.UpdateOptsBuilder

	// The maximum transmission unit (MTU) value to address fragmentation.
	// Minimum value is 68 for IPv4, and 1280 for IPv6.
	MTU int `json:"mtu,omitempty"`
}

// ToNetworkUpdateMap adds an MTU to the base network uptade options.
func (opts UpdateOptsExt) ToNetworkUpdateMap() (map[string]interface{}, error) {
	base, err := opts.UpdateOptsBuilder.ToNetworkUpdateMap()
	if err != nil {
		return nil, err
	}

	if opts.MTU == 0 {
		return base, nil
	}

	networkMap := base["network"].(map[string]interface{})
	networkMap["mtu"] = opts.MTU

	return base, nil
}
