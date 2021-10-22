package roles

import (
	"encoding/json"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/internal"
	"github.com/gophercloud/gophercloud/pagination"
)

// Role grants permissions to a user.
type Role struct {
	// DomainID is the domain ID the role belongs to.
	DomainID string `json:"domain_id"`

	// ID is the unique ID of the role.
	ID string `json:"id"`

	// Links contains referencing links to the role.
	Links map[string]interface{} `json:"links"`

	// Name is the role name
	Name string `json:"name"`

	// Extra is a collection of miscellaneous key/values.
	Extra map[string]interface{} `json:"-"`
}

func (r *Role) UnmarshalJSON(b []byte) error {
	type tmp Role
	var s struct {
		tmp
		Extra map[string]interface{} `json:"extra"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Role(s.tmp)

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
			r.Extra = internal.RemainingKeys(Role{}, resultMap)
		}
	}

	return err
}

type roleResult struct {
	gophercloud.Result
}

// GetResult is the response from a Get operation. Call its Extract method
// to interpret it as a Role.
type GetResult struct {
	roleResult
}

// CreateResult is the response from a Create operation. Call its Extract method
// to interpret it as a Role
type CreateResult struct {
	roleResult
}

// UpdateResult is the response from an Update operation. Call its Extract
// method to interpret it as a Role.
type UpdateResult struct {
	roleResult
}

// DeleteResult is the response from a Delete operation. Call its ExtractErr to
// determine if the request succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// RolePage is a single page of Role results.
type RolePage struct {
	pagination.LinkedPageBase
}

// IsEmpty determines whether or not a page of Roles contains any results.
func (r RolePage) IsEmpty() (bool, error) {
	roles, err := ExtractRoles(r)
	return len(roles) == 0, err
}

// NextPageURL extracts the "next" link from the links section of the result.
func (r RolePage) NextPageURL() (string, error) {
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

// ExtractProjects returns a slice of Roles contained in a single page of
// results.
func ExtractRoles(r pagination.Page) ([]Role, error) {
	var s struct {
		Roles []Role `json:"roles"`
	}
	err := (r.(RolePage)).ExtractInto(&s)
	return s.Roles, err
}

// Extract interprets any roleResults as a Role.
func (r roleResult) Extract() (*Role, error) {
	var s struct {
		Role *Role `json:"role"`
	}
	err := r.ExtractInto(&s)
	return s.Role, err
}

// RoleAssignment is the result of a role assignments query.
type RoleAssignment struct {
	Role  AssignedRole `json:"role,omitempty"`
	Scope Scope        `json:"scope,omitempty"`
	User  User         `json:"user,omitempty"`
	Group Group        `json:"group,omitempty"`
}

// AssignedRole represents a Role in an assignment.
type AssignedRole struct {
	ID string `json:"id,omitempty"`
}

// Scope represents a scope in a Role assignment.
type Scope struct {
	Domain  Domain  `json:"domain,omitempty"`
	Project Project `json:"project,omitempty"`
}

// Domain represents a domain in a role assignment scope.
type Domain struct {
	ID string `json:"id,omitempty"`
}

// Project represents a project in a role assignment scope.
type Project struct {
	ID string `json:"id,omitempty"`
}

// User represents a user in a role assignment scope.
type User struct {
	ID string `json:"id,omitempty"`
}

// Group represents a group in a role assignment scope.
type Group struct {
	ID string `json:"id,omitempty"`
}

// RoleAssignmentPage is a single page of RoleAssignments results.
type RoleAssignmentPage struct {
	pagination.LinkedPageBase
}

// IsEmpty returns true if the RoleAssignmentPage contains no results.
func (r RoleAssignmentPage) IsEmpty() (bool, error) {
	roleAssignments, err := ExtractRoleAssignments(r)
	return len(roleAssignments) == 0, err
}

// NextPageURL uses the response's embedded link reference to navigate to
// the next page of results.
func (r RoleAssignmentPage) NextPageURL() (string, error) {
	var s struct {
		Links struct {
			Next string `json:"next"`
		} `json:"links"`
	}
	err := r.ExtractInto(&s)
	return s.Links.Next, err
}

// ExtractRoleAssignments extracts a slice of RoleAssignments from a Collection
// acquired from List.
func ExtractRoleAssignments(r pagination.Page) ([]RoleAssignment, error) {
	var s struct {
		RoleAssignments []RoleAssignment `json:"role_assignments"`
	}
	err := (r.(RoleAssignmentPage)).ExtractInto(&s)
	return s.RoleAssignments, err
}

// AssignmentResult represents the result of an assign operation.
// Call ExtractErr method to determine if the request succeeded or failed.
type AssignmentResult struct {
	gophercloud.ErrResult
}

// UnassignmentResult represents the result of an unassign operation.
// Call ExtractErr method to determine if the request succeeded or failed.
type UnassignmentResult struct {
	gophercloud.ErrResult
}
