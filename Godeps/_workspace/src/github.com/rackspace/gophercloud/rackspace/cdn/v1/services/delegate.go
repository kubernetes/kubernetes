package services

import (
	"github.com/rackspace/gophercloud"

	os "github.com/rackspace/gophercloud/openstack/cdn/v1/services"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager which allows you to iterate over a collection of
// CDN services. It accepts a ListOpts struct, which allows for pagination via
// marker and limit.
func List(c *gophercloud.ServiceClient, opts os.ListOptsBuilder) pagination.Pager {
	return os.List(c, opts)
}

// Create accepts a CreateOpts struct and creates a new CDN service using the
// values provided.
func Create(c *gophercloud.ServiceClient, opts os.CreateOptsBuilder) os.CreateResult {
	return os.Create(c, opts)
}

// Get retrieves a specific service based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) os.GetResult {
	return os.Get(c, id)
}

// Update accepts a UpdateOpts struct and updates an existing CDN service using
// the values provided.
func Update(c *gophercloud.ServiceClient, id string, patches []os.Patch) os.UpdateResult {
	return os.Update(c, id, patches)
}

// Delete accepts a unique ID and deletes the CDN service associated with it.
func Delete(c *gophercloud.ServiceClient, id string) os.DeleteResult {
	return os.Delete(c, id)
}
