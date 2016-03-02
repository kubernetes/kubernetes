package volumetypes

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// VolumeType contains all information associated with an OpenStack Volume Type.
type VolumeType struct {
	ExtraSpecs map[string]interface{} `json:"extra_specs" mapstructure:"extra_specs"` // user-defined metadata
	ID         string                 `json:"id" mapstructure:"id"`                   // unique identifier
	Name       string                 `json:"name" mapstructure:"name"`               // display name
}

// CreateResult contains the response body and error from a Create request.
type CreateResult struct {
	commonResult
}

// GetResult contains the response body and error from a Get request.
type GetResult struct {
	commonResult
}

// DeleteResult contains the response error from a Delete request.
type DeleteResult struct {
	gophercloud.ErrResult
}

// ListResult is a pagination.Pager that is returned from a call to the List function.
type ListResult struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a ListResult contains no Volume Types.
func (r ListResult) IsEmpty() (bool, error) {
	volumeTypes, err := ExtractVolumeTypes(r)
	if err != nil {
		return true, err
	}
	return len(volumeTypes) == 0, nil
}

// ExtractVolumeTypes extracts and returns Volume Types.
func ExtractVolumeTypes(page pagination.Page) ([]VolumeType, error) {
	var response struct {
		VolumeTypes []VolumeType `mapstructure:"volume_types"`
	}

	err := mapstructure.Decode(page.(ListResult).Body, &response)
	return response.VolumeTypes, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract will get the Volume Type object out of the commonResult object.
func (r commonResult) Extract() (*VolumeType, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var res struct {
		VolumeType *VolumeType `json:"volume_type" mapstructure:"volume_type"`
	}

	err := mapstructure.Decode(r.Body, &res)

	return res.VolumeType, err
}
