package roles

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Role represents an API role resource.
type Role struct {
	// ID is the unique ID for the role.
	ID string

	// Name is the human-readable name of the role.
	Name string

	// Description is the description of the role.
	Description string

	// ServiceID is the associated service for this role.
	ServiceID string
}

// RolePage is a single page of a user Role collection.
type RolePage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a page of Roles contains any results.
func (r RolePage) IsEmpty() (bool, error) {
	users, err := ExtractRoles(r)
	return len(users) == 0, err
}

// ExtractRoles returns a slice of roles contained in a single page of results.
func ExtractRoles(r pagination.Page) ([]Role, error) {
	var s struct {
		Roles []Role `json:"roles"`
	}
	err := (r.(RolePage)).ExtractInto(&s)
	return s.Roles, err
}

// UserRoleResult represents the result of either an AddUserRole or
// a DeleteUserRole operation. Call its ExtractErr method to determine
// if the request succeeded or failed.
type UserRoleResult struct {
	gophercloud.ErrResult
}
