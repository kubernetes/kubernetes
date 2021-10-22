package stackresources

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Find retrieves stack resources for the given stack name.
func Find(c *gophercloud.ServiceClient, stackName string) (r FindResult) {
	_, r.Err = c.Get(findURL(c, stackName), &r.Body, nil)
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToStackResourceListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Marker and Limit are used for pagination.
type ListOpts struct {
	// Include resources from nest stacks up to Depth levels of recursion.
	Depth int `q:"nested_depth"`
}

// ToStackResourceListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToStackResourceListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List makes a request against the API to list resources for the given stack.
func List(client *gophercloud.ServiceClient, stackName, stackID string, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client, stackName, stackID)
	if opts != nil {
		query, err := opts.ToStackResourceListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ResourcePage{pagination.SinglePageBase(r)}
	})
}

// Get retreives data for the given stack resource.
func Get(c *gophercloud.ServiceClient, stackName, stackID, resourceName string) (r GetResult) {
	_, r.Err = c.Get(getURL(c, stackName, stackID, resourceName), &r.Body, nil)
	return
}

// Metadata retreives the metadata for the given stack resource.
func Metadata(c *gophercloud.ServiceClient, stackName, stackID, resourceName string) (r MetadataResult) {
	_, r.Err = c.Get(metadataURL(c, stackName, stackID, resourceName), &r.Body, nil)
	return
}

// ListTypes makes a request against the API to list resource types.
func ListTypes(client *gophercloud.ServiceClient) pagination.Pager {
	return pagination.NewPager(client, listTypesURL(client), func(r pagination.PageResult) pagination.Page {
		return ResourceTypePage{pagination.SinglePageBase(r)}
	})
}

// Schema retreives the schema for the given resource type.
func Schema(c *gophercloud.ServiceClient, resourceType string) (r SchemaResult) {
	_, r.Err = c.Get(schemaURL(c, resourceType), &r.Body, nil)
	return
}

// Template retreives the template representation for the given resource type.
func Template(c *gophercloud.ServiceClient, resourceType string) (r TemplateResult) {
	_, r.Err = c.Get(templateURL(c, resourceType), &r.Body, nil)
	return
}

// MarkUnhealthyOpts contains the common options struct used in this package's
// MarkUnhealthy operations.
type MarkUnhealthyOpts struct {
	// A boolean indicating whether the target resource should be marked as unhealthy.
	MarkUnhealthy bool `json:"mark_unhealthy"`
	// The reason for the current stack resource state.
	ResourceStatusReason string `json:"resource_status_reason,omitempty"`
}

// MarkUnhealthyOptsBuilder is the interface options structs have to satisfy in order
// to be used in the MarkUnhealthy operation in this package
type MarkUnhealthyOptsBuilder interface {
	ToMarkUnhealthyMap() (map[string]interface{}, error)
}

// ToMarkUnhealthyMap validates that a template was supplied and calls
// the ToMarkUnhealthyMap private function.
func (opts MarkUnhealthyOpts) ToMarkUnhealthyMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}
	return b, nil
}

// MarkUnhealthy marks the specified resource in the stack as unhealthy.
func MarkUnhealthy(c *gophercloud.ServiceClient, stackName, stackID, resourceName string, opts MarkUnhealthyOptsBuilder) (r MarkUnhealthyResult) {
	b, err := opts.ToMarkUnhealthyMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Patch(markUnhealthyURL(c, stackName, stackID, resourceName), b, nil, nil)
	return
}
