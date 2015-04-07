package external

import "github.com/rackspace/gophercloud/openstack/networking/v2/networks"

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

// CreateOpts is the structure used when creating new external network
// resources. It embeds networks.CreateOpts and so inherits all of its required
// and optional fields, with the addition of the External field.
type CreateOpts struct {
	Parent   networks.CreateOpts
	External bool
}

// ToNetworkCreateMap casts a CreateOpts struct to a map.
func (o CreateOpts) ToNetworkCreateMap() (map[string]interface{}, error) {
	outer, err := o.Parent.ToNetworkCreateMap()
	if err != nil {
		return nil, err
	}

	outer["network"].(map[string]interface{})["router:external"] = o.External

	return outer, nil
}

// UpdateOpts is the structure used when updating existing external network
// resources. It embeds networks.UpdateOpts and so inherits all of its required
// and optional fields, with the addition of the External field.
type UpdateOpts struct {
	Parent   networks.UpdateOpts
	External bool
}

// ToNetworkUpdateMap casts an UpdateOpts struct to a map.
func (o UpdateOpts) ToNetworkUpdateMap() (map[string]interface{}, error) {
	outer, err := o.Parent.ToNetworkUpdateMap()
	if err != nil {
		return nil, err
	}

	outer["network"].(map[string]interface{})["router:external"] = o.External

	return outer, nil
}
