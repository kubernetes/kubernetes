package regions

import (
	"encoding/json"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/internal"
	"github.com/gophercloud/gophercloud/pagination"
)

// Region helps manage related users.
type Region struct {
	// Description describes the region purpose.
	Description string `json:"description"`

	// ID is the unique ID of the region.
	ID string `json:"id"`

	// Extra is a collection of miscellaneous key/values.
	Extra map[string]interface{} `json:"-"`

	// Links contains referencing links to the region.
	Links map[string]interface{} `json:"links"`

	// ParentRegionID is the ID of the parent region.
	ParentRegionID string `json:"parent_region_id"`
}

func (r *Region) UnmarshalJSON(b []byte) error {
	type tmp Region
	var s struct {
		tmp
		Extra map[string]interface{} `json:"extra"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Region(s.tmp)

	// Collect other fields and bundle them into Extra
	// but only if a field titled "extra" wasn't sent.
	if s.Extra != nil {
		r.Extra = s.Extra
	} else {
		var result interface{}
		err := json.Unmarshal(b, &result)
		if err != nil {
			return err
		}
		if resultMap, ok := result.(map[string]interface{}); ok {
			r.Extra = internal.RemainingKeys(Region{}, resultMap)
		}
	}

	return err
}

type regionResult struct {
	gophercloud.Result
}

// GetResult is the response from a Get operation. Call its Extract method
// to interpret it as a Region.
type GetResult struct {
	regionResult
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a Region.
type CreateResult struct {
	regionResult
}

// UpdateResult is the response from an Update operation. Call its Extract
// method to interpret it as a Region.
type UpdateResult struct {
	regionResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// RegionPage is a single page of Region results.
type RegionPage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a page of Regions contains any results.
func (r RegionPage) IsEmpty() (bool, error) {
	regions, err := ExtractRegions(r)
	return len(regions) == 0, err
}

// NextPageURL extracts the "next" link from the links section of the result.
func (r RegionPage) NextPageURL() (string, error) {
	var s struct {
		Links struct {
			Next     string `json:"next"`
			Previous string `json:"previous"`
		} `json:"links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return s.Links.Next, err
}

// ExtractRegions returns a slice of Regions contained in a single page of results.
func ExtractRegions(r pagination.Page) ([]Region, error) {
	var s struct {
		Regions []Region `json:"regions"`
	}
	err := (r.(RegionPage)).ExtractInto(&s)
	return s.Regions, err
}

// Extract interprets any region results as a Region.
func (r regionResult) Extract() (*Region, error) {
	var s struct {
		Region *Region `json:"region"`
	}
	err := r.ExtractInto(&s)
	return s.Region, err
}
