package stackresources

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Find retrieves stack resources for the given stack name.
func Find(c *gophercloud.ServiceClient, stackName string) FindResult {
	var res FindResult

	// Send request to API
	_, res.Err = c.Request("GET", findURL(c, stackName), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
	})
	return res
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
	if err != nil {
		return "", err
	}
	return q.String(), nil
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

	createPageFn := func(r pagination.PageResult) pagination.Page {
		return ResourcePage{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(client, url, createPageFn)
}

// Get retreives data for the given stack resource.
func Get(c *gophercloud.ServiceClient, stackName, stackID, resourceName string) GetResult {
	var res GetResult

	// Send request to API
	_, res.Err = c.Get(getURL(c, stackName, stackID, resourceName), &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Metadata retreives the metadata for the given stack resource.
func Metadata(c *gophercloud.ServiceClient, stackName, stackID, resourceName string) MetadataResult {
	var res MetadataResult

	// Send request to API
	_, res.Err = c.Get(metadataURL(c, stackName, stackID, resourceName), &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// ListTypes makes a request against the API to list resource types.
func ListTypes(client *gophercloud.ServiceClient) pagination.Pager {
	url := listTypesURL(client)

	createPageFn := func(r pagination.PageResult) pagination.Page {
		return ResourceTypePage{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(client, url, createPageFn)
}

// Schema retreives the schema for the given resource type.
func Schema(c *gophercloud.ServiceClient, resourceType string) SchemaResult {
	var res SchemaResult

	// Send request to API
	_, res.Err = c.Get(schemaURL(c, resourceType), &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Template retreives the template representation for the given resource type.
func Template(c *gophercloud.ServiceClient, resourceType string) TemplateResult {
	var res TemplateResult

	// Send request to API
	_, res.Err = c.Get(templateURL(c, resourceType), &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}
