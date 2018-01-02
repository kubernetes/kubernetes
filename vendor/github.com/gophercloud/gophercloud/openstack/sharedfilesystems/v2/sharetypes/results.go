package sharetypes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ShareType contains all the information associated with an OpenStack
// ShareType.
type ShareType struct {
	// The Share Type ID
	ID string `json:"id"`
	// The Share Type name
	Name string `json:"name"`
	// Indicates whether a share type is publicly accessible
	IsPublic bool `json:"os-share-type-access:is_public"`
	// The required extra specifications for the share type
	RequiredExtraSpecs map[string]interface{} `json:"required_extra_specs"`
	// The extra specifications for the share type
	ExtraSpecs map[string]interface{} `json:"extra_specs"`
}

type commonResult struct {
	gophercloud.Result
}

// Extract will get the ShareType object out of the commonResult object.
func (r commonResult) Extract() (*ShareType, error) {
	var s struct {
		ShareType *ShareType `json:"share_type"`
	}
	err := r.ExtractInto(&s)
	return s.ShareType, err
}

// CreateResult contains the response body and error from a Create request.
type CreateResult struct {
	commonResult
}

// DeleteResult contains the response body and error from a Delete request.
type DeleteResult struct {
	gophercloud.ErrResult
}

// ShareTypePage is a pagination.pager that is returned from a call to the List function.
type ShareTypePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a ListResult contains no ShareTypes.
func (r ShareTypePage) IsEmpty() (bool, error) {
	shareTypes, err := ExtractShareTypes(r)
	return len(shareTypes) == 0, err
}

// ExtractShareTypes extracts and returns ShareTypes. It is used while
// iterating over a sharetypes.List call.
func ExtractShareTypes(r pagination.Page) ([]ShareType, error) {
	var s struct {
		ShareTypes []ShareType `json:"share_types"`
	}
	err := (r.(ShareTypePage)).ExtractInto(&s)
	return s.ShareTypes, err
}

// GetDefaultResult contains the response body and error from a Get Default request.
type GetDefaultResult struct {
	commonResult
}

// ExtraSpecs contains all the information associated with extra specifications
// for an Openstack ShareType.
type ExtraSpecs map[string]interface{}

type extraSpecsResult struct {
	gophercloud.Result
}

// Extract will get the ExtraSpecs object out of the commonResult object.
func (r extraSpecsResult) Extract() (ExtraSpecs, error) {
	var s struct {
		Specs ExtraSpecs `json:"extra_specs"`
	}
	err := r.ExtractInto(&s)
	return s.Specs, err
}

// GetExtraSpecsResult contains the response body and error from a Get Extra Specs request.
type GetExtraSpecsResult struct {
	extraSpecsResult
}

// SetExtraSpecsResult contains the response body and error from a Set Extra Specs request.
type SetExtraSpecsResult struct {
	extraSpecsResult
}

// UnsetExtraSpecsResult contains the response body and error from a Unset Extra Specs request.
type UnsetExtraSpecsResult struct {
	gophercloud.ErrResult
}

// ShareTypeAccess contains all the information associated with an OpenStack
// ShareTypeAccess.
type ShareTypeAccess struct {
	// The share type ID of the member.
	ShareTypeID string `json:"share_type_id"`
	// The UUID of the project for which access to the share type is granted.
	ProjectID string `json:"project_id"`
}

type shareTypeAccessResult struct {
	gophercloud.Result
}

// ShowAccessResult contains the response body and error from a Show access request.
type ShowAccessResult struct {
	shareTypeAccessResult
}

// Extract will get the ShareTypeAccess objects out of the shareTypeAccessResult object.
func (r ShowAccessResult) Extract() ([]ShareTypeAccess, error) {
	var s struct {
		ShareTypeAccess []ShareTypeAccess `json:"share_type_access"`
	}
	err := r.ExtractInto(&s)
	return s.ShareTypeAccess, err
}

// AddAccessResult contains the response body and error from a Add Access request.
type AddAccessResult struct {
	gophercloud.ErrResult
}

// RemoveAccessResult contains the response body and error from a Remove Access request.
type RemoveAccessResult struct {
	gophercloud.ErrResult
}
