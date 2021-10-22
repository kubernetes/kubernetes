package imageimport

import "github.com/gophercloud/gophercloud"

type commonResult struct {
	gophercloud.Result
}

// GetResult represents the result of a get operation. Call its Extract method
// to interpret it as ImportInfo.
type GetResult struct {
	commonResult
}

// CreateResult is the result of import Create operation. Call its ExtractErr
// method to determine if the request succeeded or failed.
type CreateResult struct {
	gophercloud.ErrResult
}

// ImportInfo represents information data for the Import API.
type ImportInfo struct {
	ImportMethods ImportMethods `json:"import-methods"`
}

// ImportMethods contains information about available Import API methods.
type ImportMethods struct {
	Description string   `json:"description"`
	Type        string   `json:"type"`
	Value       []string `json:"value"`
}

// Extract is a function that accepts a result and extracts ImportInfo.
func (r commonResult) Extract() (*ImportInfo, error) {
	var s *ImportInfo
	err := r.ExtractInto(&s)
	return s, err
}
