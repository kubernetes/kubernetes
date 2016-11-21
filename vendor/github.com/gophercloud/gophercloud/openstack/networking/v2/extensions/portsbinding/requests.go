package portsbinding

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
)

// Get retrieves a specific port based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = c.Get(getURL(c, id), &r.Body, nil)
	return
}

// CreateOpts represents the attributes used when creating a new
// port with extended attributes.
type CreateOpts struct {
	// CreateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Create operation in this package.
	ports.CreateOptsBuilder `json:"-"`
	// The ID of the host where the port is allocated
	HostID string `json:"binding:host_id,omitempty"`
	// The virtual network interface card (vNIC) type that is bound to the
	// neutron port
	VNICType string `json:"binding:vnic_type,omitempty"`
	// A dictionary that enables the application running on the specified
	// host to pass and receive virtual network interface (VIF) port-specific
	// information to the plug-in
	Profile map[string]string `json:"binding:profile,omitempty"`
}

// ToPortCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToPortCreateMap() (map[string]interface{}, error) {
	b1, err := opts.CreateOptsBuilder.ToPortCreateMap()
	if err != nil {
		return nil, err
	}

	b2, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	port := b1["port"].(map[string]interface{})

	for k, v := range b2 {
		port[k] = v
	}

	return map[string]interface{}{"port": port}, nil
}

// Create accepts a CreateOpts struct and creates a new port with extended attributes.
// You must remember to provide a NetworkID value.
func Create(c *gophercloud.ServiceClient, opts ports.CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToPortCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(createURL(c), b, &r.Body, nil)
	return
}

// UpdateOpts represents the attributes used when updating an existing port.
type UpdateOpts struct {
	// UpdateOptsBuilder is the interface options structs have to satisfy in order
	// to be used in the main Update operation in this package.
	ports.UpdateOptsBuilder `json:"-"`
	// The ID of the host where the port is allocated
	HostID string `json:"binding:host_id,omitempty"`
	// The virtual network interface card (vNIC) type that is bound to the
	// neutron port
	VNICType string `json:"binding:vnic_type,omitempty"`
	// A dictionary that enables the application running on the specified
	// host to pass and receive virtual network interface (VIF) port-specific
	// information to the plug-in
	Profile map[string]string `json:"binding:profile,omitempty"`
}

// ToPortUpdateMap casts an UpdateOpts struct to a map.
func (opts UpdateOpts) ToPortUpdateMap() (map[string]interface{}, error) {
	b1, err := opts.UpdateOptsBuilder.ToPortUpdateMap()
	if err != nil {
		return nil, err
	}

	b2, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	port := b1["port"].(map[string]interface{})

	for k, v := range b2 {
		port[k] = v
	}

	return map[string]interface{}{"port": port}, nil
}

// Update accepts a UpdateOpts struct and updates an existing port using the
// values provided.
func Update(c *gophercloud.ServiceClient, id string, opts ports.UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToPortUpdateMap()
	if err != nil {
		r.Err = err
		return r
	}
	_, r.Err = c.Put(updateURL(c, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})
	return
}
