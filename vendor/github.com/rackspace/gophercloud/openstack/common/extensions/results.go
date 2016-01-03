package extensions

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// GetResult temporarily stores the result of a Get call.
// Use its Extract() method to interpret it as an Extension.
type GetResult struct {
	gophercloud.Result
}

// Extract interprets a GetResult as an Extension.
func (r GetResult) Extract() (*Extension, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		Extension *Extension `json:"extension"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.Extension, err
}

// Extension is a struct that represents an OpenStack extension.
type Extension struct {
	Updated     string        `json:"updated" mapstructure:"updated"`
	Name        string        `json:"name" mapstructure:"name"`
	Links       []interface{} `json:"links" mapstructure:"links"`
	Namespace   string        `json:"namespace" mapstructure:"namespace"`
	Alias       string        `json:"alias" mapstructure:"alias"`
	Description string        `json:"description" mapstructure:"description"`
}

// ExtensionPage is the page returned by a pager when traversing over a collection of extensions.
type ExtensionPage struct {
	pagination.SinglePageBase
}

// IsEmpty checks whether an ExtensionPage struct is empty.
func (r ExtensionPage) IsEmpty() (bool, error) {
	is, err := ExtractExtensions(r)
	if err != nil {
		return true, err
	}
	return len(is) == 0, nil
}

// ExtractExtensions accepts a Page struct, specifically an ExtensionPage struct, and extracts the
// elements into a slice of Extension structs.
// In other words, a generic collection is mapped into a relevant slice.
func ExtractExtensions(page pagination.Page) ([]Extension, error) {
	var resp struct {
		Extensions []Extension `mapstructure:"extensions"`
	}

	err := mapstructure.Decode(page.(ExtensionPage).Body, &resp)

	return resp.Extensions, err
}
