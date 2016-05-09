package rules

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a Pager which allows you to iterate over a collection of
// security group rules. It accepts a ListOpts struct, which allows you to filter
// and sort the returned collection for greater efficiency.
func List(c *gophercloud.ServiceClient, opts os.ListOpts) pagination.Pager {
	return os.List(c, opts)
}

// Create is an operation which provisions a new security group with default
// security group rules for the IPv4 and IPv6 ether types.
func Create(c *gophercloud.ServiceClient, opts os.CreateOpts) os.CreateResult {
	return os.Create(c, opts)
}

// Get retrieves a particular security group based on its unique ID.
func Get(c *gophercloud.ServiceClient, id string) os.GetResult {
	return os.Get(c, id)
}

// Delete will permanently delete a particular security group based on its unique ID.
func Delete(c *gophercloud.ServiceClient, id string) os.DeleteResult {
	return os.Delete(c, id)
}
