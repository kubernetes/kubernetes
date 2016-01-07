package subnets

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/networking/v2/subnets"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager which allows you to iterate over a collection of
// subnets. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
//
// Default policy settings return only those subnets that are owned by the tenant
// who submits the request, unless the request is submitted by a user with
// administrative rights.
func List(c *gophercloud.ServiceClient, opts os.ListOptsBuilder) pagination.Pager {
	return os.List(c, opts)
}

// Get retrieves a specific subnet based on its unique ID.
func Get(c *gophercloud.ServiceClient, networkID string) os.GetResult {
	return os.Get(c, networkID)
}

// Create accepts a CreateOpts struct and creates a new subnet using the values
// provided. You must remember to provide a valid NetworkID, CIDR and IP version.
func Create(c *gophercloud.ServiceClient, opts os.CreateOptsBuilder) os.CreateResult {
	return os.Create(c, opts)
}

// Update accepts a UpdateOpts struct and updates an existing subnet using the
// values provided.
func Update(c *gophercloud.ServiceClient, networkID string, opts os.UpdateOptsBuilder) os.UpdateResult {
	return os.Update(c, networkID, opts)
}

// Delete accepts a unique ID and deletes the subnet associated with it.
func Delete(c *gophercloud.ServiceClient, networkID string) os.DeleteResult {
	return os.Delete(c, networkID)
}
