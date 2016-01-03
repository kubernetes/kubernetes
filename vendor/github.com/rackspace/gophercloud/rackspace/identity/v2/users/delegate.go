package users

import (
	"errors"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/identity/v2/users"
	"github.com/rackspace/gophercloud/pagination"
)

// List returns a pager that allows traversal over a collection of users.
func List(client *gophercloud.ServiceClient) pagination.Pager {
	return os.List(client)
}

// CommonOpts are the options which are shared between CreateOpts and
// UpdateOpts
type CommonOpts struct {
	// Required. The username to assign to the user. When provided, the username
	// must:
	// - start with an alphabetical (A-Za-z) character
	// - have a minimum length of 1 character
	//
	// The username may contain upper and lowercase characters, as well as any of
	// the following special character: . - @ _
	Username string

	// Required. Email address for the user account.
	Email string

	// Required. Indicates whether the user can authenticate after the user
	// account is created. If no value is specified, the default value is true.
	Enabled os.EnabledState

	// Optional. The password to assign to the user. If provided, the password
	// must:
	// - start with an alphabetical (A-Za-z) character
	// - have a minimum length of 8 characters
	// - contain at least one uppercase character, one lowercase character, and
	//   one numeric character.
	//
	// The password may contain any of the following special characters: . - @ _
	Password string
}

// CreateOpts represents the options needed when creating new users.
type CreateOpts CommonOpts

// ToUserCreateMap assembles a request body based on the contents of a CreateOpts.
func (opts CreateOpts) ToUserCreateMap() (map[string]interface{}, error) {
	m := make(map[string]interface{})

	if opts.Username == "" {
		return m, errors.New("Username is a required field")
	}
	if opts.Enabled == nil {
		return m, errors.New("Enabled is a required field")
	}
	if opts.Email == "" {
		return m, errors.New("Email is a required field")
	}

	if opts.Username != "" {
		m["username"] = opts.Username
	}
	if opts.Email != "" {
		m["email"] = opts.Email
	}
	if opts.Enabled != nil {
		m["enabled"] = opts.Enabled
	}
	if opts.Password != "" {
		m["OS-KSADM:password"] = opts.Password
	}

	return map[string]interface{}{"user": m}, nil
}

// Create is the operation responsible for creating new users.
func Create(client *gophercloud.ServiceClient, opts os.CreateOptsBuilder) CreateResult {
	return CreateResult{os.Create(client, opts)}
}

// Get requests details on a single user, either by ID.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	return GetResult{os.Get(client, id)}
}

// UpdateOptsBuilder allows extensions to add additional attributes to the Update request.
type UpdateOptsBuilder interface {
	ToUserUpdateMap() map[string]interface{}
}

// UpdateOpts specifies the base attributes that may be updated on an existing server.
type UpdateOpts CommonOpts

// ToUserUpdateMap formats an UpdateOpts structure into a request body.
func (opts UpdateOpts) ToUserUpdateMap() map[string]interface{} {
	m := make(map[string]interface{})

	if opts.Username != "" {
		m["username"] = opts.Username
	}
	if opts.Enabled != nil {
		m["enabled"] = &opts.Enabled
	}
	if opts.Email != "" {
		m["email"] = opts.Email
	}

	return map[string]interface{}{"user": m}
}

// Update is the operation responsible for updating exist users by their UUID.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) UpdateResult {
	var result UpdateResult

	_, result.Err = client.Request("POST", os.ResourceURL(client, id), gophercloud.RequestOpts{
		JSONResponse: &result.Body,
		JSONBody:     opts.ToUserUpdateMap(),
		OkCodes:      []int{200},
	})

	return result
}

// Delete is the operation responsible for permanently deleting an API user.
func Delete(client *gophercloud.ServiceClient, id string) os.DeleteResult {
	return os.Delete(client, id)
}

// ResetAPIKey resets the User's API key.
func ResetAPIKey(client *gophercloud.ServiceClient, id string) ResetAPIKeyResult {
	var result ResetAPIKeyResult

	_, result.Err = client.Request("POST", resetAPIKeyURL(client, id), gophercloud.RequestOpts{
		JSONResponse: &result.Body,
		OkCodes:      []int{200},
	})

	return result
}
