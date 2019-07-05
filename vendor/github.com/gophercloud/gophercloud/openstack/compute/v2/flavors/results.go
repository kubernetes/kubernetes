package flavors

import (
	"encoding/json"
	"strconv"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// CreateResult is the response of a Get operations. Call its Extract method to
// interpret it as a Flavor.
type CreateResult struct {
	commonResult
}

// GetResult is the response of a Get operations. Call its Extract method to
// interpret it as a Flavor.
type GetResult struct {
	commonResult
}

// DeleteResult is the result from a Delete operation. Call its ExtractErr
// method to determine if the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// Extract provides access to the individual Flavor returned by the Get and
// Create functions.
func (r commonResult) Extract() (*Flavor, error) {
	var s struct {
		Flavor *Flavor `json:"flavor"`
	}
	err := r.ExtractInto(&s)
	return s.Flavor, err
}

// Flavor represent (virtual) hardware configurations for server resources
// in a region.
type Flavor struct {
	// ID is the flavor's unique ID.
	ID string `json:"id"`

	// Disk is the amount of root disk, measured in GB.
	Disk int `json:"disk"`

	// RAM is the amount of memory, measured in MB.
	RAM int `json:"ram"`

	// Name is the name of the flavor.
	Name string `json:"name"`

	// RxTxFactor describes bandwidth alterations of the flavor.
	RxTxFactor float64 `json:"rxtx_factor"`

	// Swap is the amount of swap space, measured in MB.
	Swap int `json:"-"`

	// VCPUs indicates how many (virtual) CPUs are available for this flavor.
	VCPUs int `json:"vcpus"`

	// IsPublic indicates whether the flavor is public.
	IsPublic bool `json:"os-flavor-access:is_public"`

	// Ephemeral is the amount of ephemeral disk space, measured in GB.
	Ephemeral int `json:"OS-FLV-EXT-DATA:ephemeral"`
}

func (r *Flavor) UnmarshalJSON(b []byte) error {
	type tmp Flavor
	var s struct {
		tmp
		Swap interface{} `json:"swap"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = Flavor(s.tmp)

	switch t := s.Swap.(type) {
	case float64:
		r.Swap = int(t)
	case string:
		switch t {
		case "":
			r.Swap = 0
		default:
			swap, err := strconv.ParseFloat(t, 64)
			if err != nil {
				return err
			}
			r.Swap = int(swap)
		}
	}

	return nil
}

// FlavorPage contains a single page of all flavors from a ListDetails call.
type FlavorPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines if a FlavorPage contains any results.
func (page FlavorPage) IsEmpty() (bool, error) {
	flavors, err := ExtractFlavors(page)
	return len(flavors) == 0, err
}

// NextPageURL uses the response's embedded link reference to navigate to the
// next page of results.
func (page FlavorPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"flavors_links"`
	}
	err := page.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// ExtractFlavors provides access to the list of flavors in a page acquired
// from the ListDetail operation.
func ExtractFlavors(r pagination.Page) ([]Flavor, error) {
	var s struct {
		Flavors []Flavor `json:"flavors"`
	}
	err := (r.(FlavorPage)).ExtractInto(&s)
	return s.Flavors, err
}

// AccessPage contains a single page of all FlavorAccess entries for a flavor.
type AccessPage struct {
	pagination.SinglePageBase
}

// IsEmpty indicates whether an AccessPage is empty.
func (page AccessPage) IsEmpty() (bool, error) {
	v, err := ExtractAccesses(page)
	return len(v) == 0, err
}

// ExtractAccesses interprets a page of results as a slice of FlavorAccess.
func ExtractAccesses(r pagination.Page) ([]FlavorAccess, error) {
	var s struct {
		FlavorAccesses []FlavorAccess `json:"flavor_access"`
	}
	err := (r.(AccessPage)).ExtractInto(&s)
	return s.FlavorAccesses, err
}

type accessResult struct {
	gophercloud.Result
}

// AddAccessResult is the response of an AddAccess operation. Call its
// Extract method to interpret it as a slice of FlavorAccess.
type AddAccessResult struct {
	accessResult
}

// RemoveAccessResult is the response of a RemoveAccess operation. Call its
// Extract method to interpret it as a slice of FlavorAccess.
type RemoveAccessResult struct {
	accessResult
}

// Extract provides access to the result of an access create or delete.
// The result will be all accesses that the flavor has.
func (r accessResult) Extract() ([]FlavorAccess, error) {
	var s struct {
		FlavorAccesses []FlavorAccess `json:"flavor_access"`
	}
	err := r.ExtractInto(&s)
	return s.FlavorAccesses, err
}

// FlavorAccess represents an ACL of tenant access to a specific Flavor.
type FlavorAccess struct {
	// FlavorID is the unique ID of the flavor.
	FlavorID string `json:"flavor_id"`

	// TenantID is the unique ID of the tenant.
	TenantID string `json:"tenant_id"`
}

// Extract interprets any extraSpecsResult as ExtraSpecs, if possible.
func (r extraSpecsResult) Extract() (map[string]string, error) {
	var s struct {
		ExtraSpecs map[string]string `json:"extra_specs"`
	}
	err := r.ExtractInto(&s)
	return s.ExtraSpecs, err
}

// extraSpecsResult contains the result of a call for (potentially) multiple
// key-value pairs. Call its Extract method to interpret it as a
// map[string]interface.
type extraSpecsResult struct {
	gophercloud.Result
}

// ListExtraSpecsResult contains the result of a Get operation. Call its Extract
// method to interpret it as a map[string]interface.
type ListExtraSpecsResult struct {
	extraSpecsResult
}

// CreateExtraSpecResult contains the result of a Create operation. Call its
// Extract method to interpret it as a map[string]interface.
type CreateExtraSpecsResult struct {
	extraSpecsResult
}

// extraSpecResult contains the result of a call for individual a single
// key-value pair.
type extraSpecResult struct {
	gophercloud.Result
}

// GetExtraSpecResult contains the result of a Get operation. Call its Extract
// method to interpret it as a map[string]interface.
type GetExtraSpecResult struct {
	extraSpecResult
}

// UpdateExtraSpecResult contains the result of an Update operation. Call its
// Extract method to interpret it as a map[string]interface.
type UpdateExtraSpecResult struct {
	extraSpecResult
}

// DeleteExtraSpecResult contains the result of a Delete operation. Call its
// ExtractErr method to determine if the call succeeded or failed.
type DeleteExtraSpecResult struct {
	gophercloud.ErrResult
}

// Extract interprets any extraSpecResult as an ExtraSpec, if possible.
func (r extraSpecResult) Extract() (map[string]string, error) {
	var s map[string]string
	err := r.ExtractInto(&s)
	return s, err
}
