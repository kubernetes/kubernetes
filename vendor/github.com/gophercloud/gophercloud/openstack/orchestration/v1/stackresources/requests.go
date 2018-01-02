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
