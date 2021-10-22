package vlantransparent

import (
	"net/url"
	"strconv"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
)

// ListOptsExt adds the vlan-transparent network options to the base ListOpts.
type ListOptsExt struct {
	networks.ListOptsBuilder
	VLANTransparent *bool `q:"vlan_transparent"`
}

// ToNetworkListQuery adds the vlan_transparent option to the base network
// list options.
func (opts ListOptsExt) ToNetworkListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts.ListOptsBuilder)
	if err != nil {
		return "", err
	}

	params := q.Query()
	if opts.VLANTransparent != nil {
		v := strconv.FormatBool(*opts.VLANTransparent)
		params.Add("vlan_transparent", v)
	}

	q = &url.URL{RawQuery: params.Encode()}
	return q.String(), err
}

// CreateOptsExt is the structure used when creating new vlan-transparent
// network resources. It embeds networks.CreateOpts and so inherits all of its
// required and optional fields, with the addition of the VLANTransparent field.
type CreateOptsExt struct {
	networks.CreateOptsBuilder
	VLANTransparent *bool `json:"vlan_transparent,omitempty"`
}

// ToNetworkCreateMap adds the vlan_transparent option to the base network
// creation options.
func (opts CreateOptsExt) ToNetworkCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToNetworkCreateMap()
	if err != nil {
		return nil, err
	}

	if opts.VLANTransparent == nil {
		return base, nil
	}

	networkMap := base["network"].(map[string]interface{})
	networkMap["vlan_transparent"] = opts.VLANTransparent

	return base, nil
}

// UpdateOptsExt is the structure used when updating existing vlan-transparent
// network resources. It embeds networks.UpdateOpts and so inherits all of its
// required and optional fields, with the addition of the VLANTransparent field.
type UpdateOptsExt struct {
	networks.UpdateOptsBuilder
	VLANTransparent *bool `json:"vlan_transparent,omitempty"`
}

// ToNetworkUpdateMap casts an UpdateOpts struct to a map.
func (opts UpdateOptsExt) ToNetworkUpdateMap() (map[string]interface{}, error) {
	base, err := opts.UpdateOptsBuilder.ToNetworkUpdateMap()
	if err != nil {
		return nil, err
	}

	if opts.VLANTransparent == nil {
		return base, nil
	}

	networkMap := base["network"].(map[string]interface{})
	networkMap["vlan_transparent"] = opts.VLANTransparent

	return base, nil
}
