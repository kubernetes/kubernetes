package rbacpolicies

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts RBAC Policy resource.
func (r commonResult) Extract() (*RBACPolicy, error) {
	var s RBACPolicy
	err := r.ExtractInto(&s)
	return &s, err
}

func (r commonResult) ExtractInto(v interface{}) error {
	return r.Result.ExtractIntoStructPtr(v, "rbac_policy")
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a RBAC Policy.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as a RBAC Policy.
type GetResult struct {
	commonResult
}

// DeleteResult represents the result of a delete operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// UpdateResult represents the result of an update operation. Call its Extract
// method to interpret it as a RBAC Policy.
type UpdateResult struct {
	commonResult
}

// RBACPolicy represents a RBAC policy.
type RBACPolicy struct {
	// UUID of the RBAC policy.
	ID string `json:"id"`

	// Action for the RBAC policy which is access_as_external or access_as_shared.
	Action PolicyAction `json:"action"`

	// ObjectID is the ID of the object_type resource.
	// An object_type of network returns a network ID and
	// object_type of qos-policy returns a QoS ID.
	ObjectID string `json:"object_id"`

	// ObjectType is the type of the object that the RBAC policy affects.
	// Types include qos-policy or network.
	ObjectType string `json:"object_type"`

	// TenantID is the ID of the project that owns the resource.
	TenantID string `json:"tenant_id"`

	// TargetTenant is the ID of the tenant to which the RBAC policy will be enforced.
	TargetTenant string `json:"target_tenant"`

	// ProjectID is the ID of the project.
	ProjectID string `json:"project_id"`

	// Tags optionally set via extensions/attributestags
	Tags []string `json:"tags"`
}

// RBACPolicyPage is the page returned by a pager when traversing over a
// collection of rbac policies.
type RBACPolicyPage struct {
	pagination.LinkedPageBase
}

// IsEmpty checks whether a RBACPolicyPage struct is empty.
func (r RBACPolicyPage) IsEmpty() (bool, error) {
	is, err := ExtractRBACPolicies(r)
	return len(is) == 0, err
}

// ExtractRBACPolicies accepts a Page struct, specifically a RBAC Policy struct,
// and extracts the elements into a slice of RBAC Policy structs. In other words,
// a generic collection is mapped into a relevant slice.
func ExtractRBACPolicies(r pagination.Page) ([]RBACPolicy, error) {
	var s []RBACPolicy
	err := ExtractRBACPolicesInto(r, &s)
	return s, err
}

// ExtractRBACPolicesInto extracts the elements into a slice of RBAC Policy structs.
func ExtractRBACPolicesInto(r pagination.Page, v interface{}) error {
	return r.(RBACPolicyPage).Result.ExtractIntoSlicePtr(v, "rbac_policies")
}
