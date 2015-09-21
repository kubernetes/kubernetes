package users

import (
	"github.com/mitchellh/mapstructure"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// User represents a user resource that exists on the API.
type User struct {
	// The UUID for this user.
	ID string

	// The human name for this user.
	Name string

	// The username for this user.
	Username string

	// Indicates whether the user is enabled (true) or disabled (false).
	Enabled bool

	// The email address for this user.
	Email string

	// The ID of the tenant to which this user belongs.
	TenantID string `mapstructure:"tenant_id"`
}

// Role assigns specific responsibilities to users, allowing them to accomplish
// certain API operations whilst scoped to a service.
type Role struct {
	// UUID of the role
	ID string

	// Name of the role
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

// IsEmpty determines whether or not a page of Tenants contains any results.
func (page UserPage) IsEmpty() (bool, error) {
	users, err := ExtractUsers(page)
	if err != nil {
		return false, err
	}
	return len(users) == 0, nil
}

// ExtractUsers returns a slice of Tenants contained in a single page of results.
func ExtractUsers(page pagination.Page) ([]User, error) {
	casted := page.(UserPage).Body
	var response struct {
		Users []User `mapstructure:"users"`
	}

	err := mapstructure.Decode(casted, &response)
	return response.Users, err
}

// IsEmpty determines whether or not a page of Tenants contains any results.
func (page RolePage) IsEmpty() (bool, error) {
	users, err := ExtractRoles(page)
	if err != nil {
		return false, err
	}
	return len(users) == 0, nil
}

// ExtractRoles returns a slice of Roles contained in a single page of results.
func ExtractRoles(page pagination.Page) ([]Role, error) {
	casted := page.(RolePage).Body
	var response struct {
		Roles []Role `mapstructure:"roles"`
	}

	err := mapstructure.Decode(casted, &response)
	return response.Roles, err
}

type commonResult struct {
	gophercloud.Result
}

// Extract interprets any commonResult as a User, if possible.
func (r commonResult) Extract() (*User, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		User User `mapstructure:"user"`
	}

	err := mapstructure.Decode(r.Body, &response)

	return &response.User, err
}

// CreateResult represents the result of a Create operation
type CreateResult struct {
	commonResult
}

// GetResult represents the result of a Get operation
type GetResult struct {
	commonResult
}

// UpdateResult represents the result of an Update operation
type UpdateResult struct {
	commonResult
}

// DeleteResult represents the result of a Delete operation
type DeleteResult struct {
	commonResult
}
