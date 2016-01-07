package stacks

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/orchestration/v1/stacks"
	"github.com/rackspace/gophercloud/pagination"
)

// Create accepts an os.CreateOpts struct and creates a new stack using the values
// provided.
func Create(c *gophercloud.ServiceClient, opts os.CreateOptsBuilder) os.CreateResult {
	return os.Create(c, opts)
}

// Adopt accepts an os.AdoptOpts struct and creates a new stack from existing stack
// resources using the values provided.
func Adopt(c *gophercloud.ServiceClient, opts os.AdoptOptsBuilder) os.AdoptResult {
	return os.Adopt(c, opts)
}

// List accepts an os.ListOpts struct and lists stacks based on the options provided.
func List(c *gophercloud.ServiceClient, opts os.ListOptsBuilder) pagination.Pager {
	return os.List(c, opts)
}

// Get retreives a stack based on the stack name and stack ID.
func Get(c *gophercloud.ServiceClient, stackName, stackID string) os.GetResult {
	return os.Get(c, stackName, stackID)
}

// Update accepts an os.UpdateOpts struct and updates a stack based on the options provided.
func Update(c *gophercloud.ServiceClient, stackName, stackID string, opts os.UpdateOptsBuilder) os.UpdateResult {
	return os.Update(c, stackName, stackID, opts)
}

// Delete deletes a stack based on the stack name and stack ID provided.
func Delete(c *gophercloud.ServiceClient, stackName, stackID string) os.DeleteResult {
	return os.Delete(c, stackName, stackID)
}

// Preview provides a preview of a stack based on the options provided.
func Preview(c *gophercloud.ServiceClient, opts os.PreviewOptsBuilder) os.PreviewResult {
	return os.Preview(c, opts)
}

// Abandon abandons a stack, keeping the resources available.
func Abandon(c *gophercloud.ServiceClient, stackName, stackID string) os.AbandonResult {
	return os.Abandon(c, stackName, stackID)
}
