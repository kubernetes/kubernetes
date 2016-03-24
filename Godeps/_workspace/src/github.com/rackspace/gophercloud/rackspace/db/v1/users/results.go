package users

import (
	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	db "github.com/rackspace/gophercloud/openstack/db/v1/databases"
	"github.com/rackspace/gophercloud/pagination"
)

// User represents a database user
type User struct {
	// The user name
	Name string

	// The user password
	Password string

	// Specifies the host from which a user is allowed to connect to the database.
	// Possible values are a string containing an IPv4 address or "%" to allow
	// connecting from any host.
	Host string

	// The databases associated with this user
	Databases []db.Database
}

// UpdatePasswordsResult represents the result of changing a user password.
type UpdatePasswordsResult struct {
	gophercloud.ErrResult
}

// UpdateResult represents the result of updating a user.
type UpdateResult struct {
	gophercloud.ErrResult
}

// GetResult represents the result of getting a user.
type GetResult struct {
	gophercloud.Result
}

// Extract will retrieve a User struct from a getresult.
func (r GetResult) Extract() (*User, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		User User `mapstructure:"user"`
	}

	err := mapstructure.Decode(r.Body, &response)
	return &response.User, err
}

// AccessPage represents a single page of a paginated user collection.
type AccessPage struct {
	pagination.LinkedPageBase
}

// IsEmpty checks to see whether the collection is empty.
func (page AccessPage) IsEmpty() (bool, error) {
	users, err := ExtractDBs(page)
	if err != nil {
		return true, err
	}
	return len(users) == 0, nil
}

// NextPageURL will retrieve the next page URL.
func (page AccessPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"databases_links"`
	}

	var r resp
	err := mapstructure.Decode(page.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// ExtractDBs will convert a generic pagination struct into a more
// relevant slice of DB structs.
func ExtractDBs(page pagination.Page) ([]db.Database, error) {
	casted := page.(AccessPage).Body

	var response struct {
		DBs []db.Database `mapstructure:"databases"`
	}

	err := mapstructure.Decode(casted, &response)
	return response.DBs, err
}

// UserPage represents a single page of a paginated user collection.
type UserPage struct {
	pagination.LinkedPageBase
}

// IsEmpty checks to see whether the collection is empty.
func (page UserPage) IsEmpty() (bool, error) {
	users, err := ExtractUsers(page)
	if err != nil {
		return true, err
	}
	return len(users) == 0, nil
}

// NextPageURL will retrieve the next page URL.
func (page UserPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"users_links"`
	}

	var r resp
	err := mapstructure.Decode(page.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// ExtractUsers will convert a generic pagination struct into a more
// relevant slice of User structs.
func ExtractUsers(page pagination.Page) ([]User, error) {
	casted := page.(UserPage).Body

	var response struct {
		Users []User `mapstructure:"users"`
	}

	err := mapstructure.Decode(casted, &response)

	return response.Users, err
}

// GrantAccessResult represents the result of granting access to a user.
type GrantAccessResult struct {
	gophercloud.ErrResult
}

// RevokeAccessResult represents the result of revoking access to a user.
type RevokeAccessResult struct {
	gophercloud.ErrResult
}
