package stackevents

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/orchestration/v1/stackevents"
	"github.com/rackspace/gophercloud/pagination"
)

// Find retreives stack events for the given stack name.
func Find(c *gophercloud.ServiceClient, stackName string) os.FindResult {
	return os.Find(c, stackName)
}

// List makes a request against the API to list resources for the given stack.
func List(c *gophercloud.ServiceClient, stackName, stackID string, opts os.ListOptsBuilder) pagination.Pager {
	return os.List(c, stackName, stackID, opts)
}

// ListResourceEvents makes a request against the API to list resources for the given stack.
func ListResourceEvents(c *gophercloud.ServiceClient, stackName, stackID, resourceName string, opts os.ListResourceEventsOptsBuilder) pagination.Pager {
	return os.ListResourceEvents(c, stackName, stackID, resourceName, opts)
}

// Get retreives data for the given stack resource.
func Get(c *gophercloud.ServiceClient, stackName, stackID, resourceName, eventID string) os.GetResult {
	return os.Get(c, stackName, stackID, resourceName, eventID)
}
