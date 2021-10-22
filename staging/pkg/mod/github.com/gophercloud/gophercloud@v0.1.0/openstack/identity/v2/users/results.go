package users

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// User represents a user resource that exists on the API.
type User struct {
	// ID is the UUID for this user.
	ID string

	// Name is the human name for this user.
	Name string

	// Username is the username for this user.
	Username string

	// Enabled indicates whether the user is enabled (true) or disabled (false).
	Enabled bool

	// Email is the email address for this user.
	Email string

	// TenantID is the ID of the tenant to which this user belongs.
	TenantID string `json:"tenant_id"`
}

// Role assigns specific responsibilities to users, allowing them to accomplish
// certain API operations whilst scoped to a service.
type Role struct {
	// ID is the UUID of the role.
	ID string

	// Name is the name of the role.
	Name string
}

// UserPage is a single page of a User collection.
type UserPage struct {
	pagination.SinglePageBase
}

// RolePage is a single page of a user Role collection.
type RolePage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a page of Users contains any results.
func (r UserPage) IsEmpty() (bool, error) {
	users, err := ExtractUsers(r)
	return len(users) == 0, err
}

// ExtractUsers returns a slice of Users contained in a single page of results.
func ExtractUsers(r pagination.Page) ([]User, error) {
	var s struct {
		Users []User `json:"users"`
	}
	err := (r.(UserPage)).ExtractInto(&s)
	return s.Users, err
}

// IsEmpty determines whether or not a page of Roles contains any results.
func (r RolePage) IsEmpty() (bool, error) {
	users, err := ExtractRoles(r)
	return len(users) == 0, err
}

// ExtractRoles returns a slice of Roles contained in a single page of results.
func ExtractRoles(r pagination.Page) ([]Role, error) {
	var s struct {
		Roles []Role `json:"roles"`
	}
	err := (r.(RolePage)).ExtractInto(&s)
	return s.Roles, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult as a User, if possible.
func (r commonResult) Extract() (*User, error) {
	var s struct {
		User *User `json:"user"`
	}
	err := r.ExtractInto(&s)
	return s.User, err
}

// CreateResult represents the result of a Create operation. Call its Extract
// method to interpret the result as a User.
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a Get operation. Call its Extract method
// to interpret the result as a User.
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an Update operation. Call its Extract
// method to interpret the result as a User.
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a Delete operation. Call its
// ExtractErr method to determine if the request succeeded or failed.
type DeleteResult struct {
	commonResult
}
