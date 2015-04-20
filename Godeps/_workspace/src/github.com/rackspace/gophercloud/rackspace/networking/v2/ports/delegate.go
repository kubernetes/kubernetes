package ports

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/networking/v2/ports"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager which allows you to iterate over a collection of
// ports. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those ports that are owned by the tenant
// who submits the request, unless the request is submitted by a user with
// administrative rights.
func List(c *gophercloud.ServiceClient, opts os.ListOptsBuilder) pagination.Pager {
	return os.List(c, opts)
}

// Get retrieves a specific port based on its unique ID.
func Get(c *gophercloud.ServiceClient, networkID string) os.GetResult {
	return os.Get(c, networkID)
}

// Create accepts a CreateOpts struct and creates a new network using the values
// provided. You must remember to provide a NetworkID value.
//
// NOTE: Currently the SecurityGroup option is not implemented to work with
// Rackspace.
func Create(c *gophercloud.ServiceClient, opts os.CreateOptsBuilder) os.CreateResult {
	return os.Create(c, opts)
}

// Update accepts a UpdateOpts struct and updates an existing port using the
// values provided.
func Update(c *gophercloud.ServiceClient, networkID string, opts os.UpdateOptsBuilder) os.UpdateResult {
	return os.Update(c, networkID, opts)
}

// Delete accepts a unique ID and deletes the port associated with it.
func Delete(c *gophercloud.ServiceClient, networkID string) os.DeleteResult {
	return os.Delete(c, networkID)
}
