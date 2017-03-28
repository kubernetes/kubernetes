package extensions

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// GetResult temporarily stores the result of a Get call.
// Use its Extract() method to interpret it as an Extension.
type GetResult struct {
	gophercloud.Result
}

// Extract interprets a GetResult as an Extension.
func (r GetResult) Extract() (*Extension, error) {
	var s struct {
		Extension *Extension `json:"extension"`
	}
	err := r.ExtractInto(&s)
	return s.Extension, err
}

// Extension is a struct that represents an OpenStack extension.
type Extension struct {
	Updated     string        `json:"updated"`
	Name        string        `json:"name"`
	Links       []interface{} `json:"links"`
	Namespace   string        `json:"namespace"`
	Alias       string        `json:"alias"`
	Description string        `json:"description"`
}

// ExtensionPage is the page returned by a pager when traversing over a collection of extensions.
type ExtensionPage struct {
	pagination.SinglePageBase
}

// IsEmpty checks whether an ExtensionPage struct is empty.
func (r ExtensionPage) IsEmpty() (bool, error) {
	is, err := ExtractExtensions(r)
	return len(is) == 0, err
}

// ExtractExtensions accepts a Page struct, specifically an ExtensionPage struct, and extracts the
// elements into a slice of Extension structs.
// In other words, a generic collection is mapped into a relevant slice.
func ExtractExtensions(r pagination.Page) ([]Extension, error) {
	var s struct {
		Extensions []Extension `json:"extensions"`
	}
	err := (r.(ExtensionPage)).ExtractInto(&s)
	return s.Extensions, err
}
