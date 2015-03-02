package stackevents

import "github.com/rackspace/gophercloud"

func findURL(c *gophercloud.ServiceClient, stackName string) string {
	return c.ServiceURL("stacks", stackName, "events")
}

func listURL(c *gophercloud.ServiceClient, stackName, stackID string) string {
	return c.ServiceURL("stacks", stackName, stackID, "events")
}

func listResourceEventsURL(c *gophercloud.ServiceClient, stackName, stackID, resourceName string) string {
	return c.ServiceURL("stacks", stackName, stackID, "resources", resourceName, "events")
}

func getURL(c *gophercloud.ServiceClient, stackName, stackID, resourceName, eventID string) string {
	return c.ServiceURL("stacks", stackName, stackID, "resources", resourceName, "events", eventID)
}
