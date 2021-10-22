package volumetypes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// VolumeType contains all information associated with an OpenStack Volume Type.
type VolumeType struct {
	ExtraSpecs map[string]interface{} `json:"extra_specs"` // user-defined metadata
	ID         string                 `json:"id"`          // unique identifier
	Name       string                 `json:"name"`        // display name
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

// VolumeTypePage is a pagination.Pager that is returned from a call to the List function.
type VolumeTypePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a VolumeTypePage contains no Volume Types.
func (r VolumeTypePage) IsEmpty() (bool, error) {
	volumeTypes, err := ExtractVolumeTypes(r)
	return len(volumeTypes) == 0, err
}

// ExtractVolumeTypes extracts and returns Volume Types.
func ExtractVolumeTypes(r pagination.Page) ([]VolumeType, error) {
	var s struct {
		VolumeTypes []VolumeType `json:"volume_types"`
	}
	err := (r.(VolumeTypePage)).ExtractInto(&s)
	return s.VolumeTypes, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract will get the Volume Type object out of the commonResult object.
func (r commonResult) Extract() (*VolumeType, error) {
	var s struct {
		VolumeType *VolumeType `json:"volume_type"`
	}
	err := r.ExtractInto(&s)
	return s.VolumeType, err
}
