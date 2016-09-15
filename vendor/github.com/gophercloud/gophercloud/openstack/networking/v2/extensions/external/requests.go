package external

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
)

// CreateOpts is the structure used when creating new external network
// resources. It embeds networks.CreateOpts and so inherits all of its required
// and optional fields, with the addition of the External field.
type CreateOpts struct {
	networks.CreateOpts
	External *bool `json:"router:external,omitempty"`
}

// ToNetworkCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToNetworkCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "network")
}

// UpdateOpts is the structure used when updating existing external network
// resources. It embeds networks.UpdateOpts and so inherits all of its required
// and optional fields, with the addition of the External field.
type UpdateOpts struct {
	networks.UpdateOpts
	External *bool `json:"router:external,omitempty"`
}

// ToNetworkUpdateMap casts an UpdateOpts struct to a map.
func (opts UpdateOpts) ToNetworkUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "network")
}
