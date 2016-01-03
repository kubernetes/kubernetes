package extensions

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	common "github.com/rackspace/gophercloud/openstack/common/extensions"
	"github.com/rackspace/gophercloud/pagination"
)

// ExtensionPage is a single page of Extension results.
type ExtensionPage struct {
	common.ExtensionPage
}

// IsEmpty returns true if the current page contains at least one Extension.
func (page ExtensionPage) IsEmpty() (bool, error) {
	is, err := ExtractExtensions(page)
	if err != nil {
		return true, err
	}
	return len(is) == 0, nil
}

// ExtractExtensions accepts a Page struct, specifically an ExtensionPage struct, and extracts the
// elements into a slice of Extension structs.
func ExtractExtensions(page pagination.Page) ([]common.Extension, error) {
	// Identity v2 adds an intermediate "values" object.

	var resp struct {
		Extensions struct {
			Values []common.Extension `mapstructure:"values"`
		} `mapstructure:"extensions"`
	}

	err := mapstructure.Decode(page.(ExtensionPage).Body, &resp)
	return resp.Extensions.Values, err
}

// Get retrieves information for a specific extension using its alias.
func Get(c *gophercloud.ServiceClient, alias string) common.GetResult {
	return common.Get(c, alias)
}

// List returns a Pager which allows you to iterate over the full collection of extensions.
// It does not accept query parameters.
func List(c *gophercloud.ServiceClient) pagination.Pager {
	return common.List(c).WithPageCreator(func(r pagination.PageResult) pagination.Page {
		return ExtensionPage{
			ExtensionPage: common.ExtensionPage{SinglePageBase: pagination.SinglePageBase(r)},
		}
	})
}
