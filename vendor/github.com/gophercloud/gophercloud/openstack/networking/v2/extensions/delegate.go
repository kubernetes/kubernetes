package extensions

import (
	"github.com/gophercloud/gophercloud"
	common "github.com/gophercloud/gophercloud/openstack/common/extensions"
	"github.com/gophercloud/gophercloud/pagination"
)

// Extension is a single OpenStack extension.
type Extension struct {
	common.Extension
}

// GetResult wraps a GetResult from common.
type GetResult struct {
	common.GetResult
}

// ExtractExtensions interprets a Page as a slice of Extensions.
func ExtractExtensions(page pagination.Page) ([]Extension, error) {
	inner, err := common.ExtractExtensions(page)
	if err != nil {
		return nil, err
	}
	outer := make([]Extension, len(inner))
	for index, ext := range inner {
		outer[index] = Extension{ext}
	}
	return outer, nil
}

// Get retrieves information for a specific extension using its alias.
func Get(c *gophercloud.ServiceClient, alias string) GetResult {
	return GetResult{common.Get(c, alias)}
}

// List returns a Pager which allows you to iterate over the full collection of extensions.
// It does not accept query parameters.
func List(c *gophercloud.ServiceClient) pagination.Pager {
	return common.List(c)
}
