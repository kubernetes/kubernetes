package profiletypes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// commonResult is the response of a base result.
type commonResult struct {
	gophercloud.Result
}

// GetResult is the response of a Get operations.
type GetResult struct {
	commonResult
}

type Schema map[string]interface{}
type SupportStatus map[string]interface{}

type ProfileType struct {
	Name          string                     `json:"name"`
	Schema        map[string]Schema          `json:"schema"`
	SupportStatus map[string][]SupportStatus `json:"support_status"`
}

func (r commonResult) Extract() (*ProfileType, error) {
	var s struct {
		ProfileType *ProfileType `json:"profile_type"`
	}
	err := r.ExtractInto(&s)
	return s.ProfileType, err
}

// ExtractProfileTypes provides access to the list of profiles in a page acquired from the List operation.
func ExtractProfileTypes(r pagination.Page) ([]ProfileType, error) {
	var s struct {
		ProfileTypes []ProfileType `json:"profile_types"`
	}
	err := (r.(ProfileTypePage)).ExtractInto(&s)
	return s.ProfileTypes, err
}

// ProfileTypePage contains a single page of all profiles from a List call.
type ProfileTypePage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines if ExtractProfileTypes contains any results.
func (page ProfileTypePage) IsEmpty() (bool, error) {
	profileTypes, err := ExtractProfileTypes(page)
	return len(profileTypes) == 0, err
}

// OperationPage contains a single page of all profile type operations from a ListOps call.
type OperationPage struct {
	pagination.SinglePageBase
}

// ExtractOps provides access to the list of operations in a page acquired from the ListOps operation.
func ExtractOps(r pagination.Page) (map[string]interface{}, error) {
	var s struct {
		Operations map[string]interface{} `json:"operations"`
	}
	err := (r.(OperationPage)).ExtractInto(&s)
	return s.Operations, err
}
