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

	// The databases associated with this user
	Databases []db.Database
}

// CreateResult represents the result of a create operation.
type CreateResult struct {
	gophercloud.ErrResult
}

// DeleteResult represents the result of a delete operation.
type DeleteResult struct {
	gophercloud.ErrResult
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
