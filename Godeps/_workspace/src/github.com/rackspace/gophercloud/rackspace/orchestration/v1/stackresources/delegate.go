package stackresources

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/orchestration/v1/stackresources"
	"github.com/rackspace/gophercloud/pagination"
)

// Find retreives stack resources for the given stack name.
func Find(c *gophercloud.ServiceClient, stackName string) os.FindResult {
	return os.Find(c, stackName)
}

// List makes a request against the API to list resources for the given stack.
func List(c *gophercloud.ServiceClient, stackName, stackID string, opts os.ListOptsBuilder) pagination.Pager {
	return os.List(c, stackName, stackID, opts)
}

// Get retreives data for the given stack resource.
func Get(c *gophercloud.ServiceClient, stackName, stackID, resourceName string) os.GetResult {
	return os.Get(c, stackName, stackID, resourceName)
}

// Metadata retreives the metadata for the given stack resource.
func Metadata(c *gophercloud.ServiceClient, stackName, stackID, resourceName string) os.MetadataResult {
	return os.Metadata(c, stackName, stackID, resourceName)
}

// ListTypes makes a request against the API to list resource types.
func ListTypes(c *gophercloud.ServiceClient) pagination.Pager {
	return os.ListTypes(c)
}

// Schema retreives the schema for the given resource type.
func Schema(c *gophercloud.ServiceClient, resourceType string) os.SchemaResult {
	return os.Schema(c, resourceType)
}

// Template retreives the template representation for the given resource type.
func Template(c *gophercloud.ServiceClient, resourceType string) os.TemplateResult {
	return os.Template(c, resourceType)
}
