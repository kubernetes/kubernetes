package policytypes

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// PolicyType represents a clustering policy type in the Openstack cloud.
type PolicyType struct {
	Name          string                         `json:"name"`
	Version       string                         `json:"version"`
	SupportStatus map[string][]SupportStatusType `json:"support_status"`
}

// SupportStatusType represents the support status information for a
// clustering policy type.
type SupportStatusType struct {
	Status string `json:"status"`
	Since  string `json:"since"`
}

// PolicyTypeDetail represents the detailed policy type information for a
// clustering policy type.
type PolicyTypeDetail struct {
	Name          string                         `json:"name"`
	Schema        map[string]interface{}         `json:"schema"`
	SupportStatus map[string][]SupportStatusType `json:"support_status,omitempty"`
}

// policyTypeResult is the base result of a Policy Type operation.
type policyTypeResult struct {
	gophercloud.Result
}

// Extract interprets any policyTypeResult result as a PolicyTypeDetail.
func (r policyTypeResult) Extract() (*PolicyTypeDetail, error) {
	var s struct {
		PolicyType *PolicyTypeDetail `json:"policy_type"`
	}
	err := r.ExtractInto(&s)
	return s.PolicyType, err
}

// GetResult is the result of a Get operation. Call its Extract method to
// interpret it as a PolicyTypeDetail.
type GetResult struct {
	policyTypeResult
}

// PolicyTypePage contains a single page of all policy types from a List call.
type PolicyTypePage struct {
	pagination.SinglePageBase
}

// IsEmpty determines if a PolicyType contains any results.
func (page PolicyTypePage) IsEmpty() (bool, error) {
	policyTypes, err := ExtractPolicyTypes(page)
	return len(policyTypes) == 0, err
}

// ExtractPolicyTypes returns a slice of PolicyTypes from a List operation.
func ExtractPolicyTypes(r pagination.Page) ([]PolicyType, error) {
	var s struct {
		PolicyTypes []PolicyType `json:"policy_types"`
	}
	err := (r.(PolicyTypePage)).ExtractInto(&s)
	return s.PolicyTypes, err
}
