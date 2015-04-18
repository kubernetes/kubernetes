package users

import (
	"errors"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

func List(client *gophercloud.ServiceClient) pagination.Pager {
	createPage := func(r pagination.PageResult) pagination.Page {
		return UserPage{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(client, rootURL(client), createPage)
}

// EnabledState represents whether the user is enabled or not.
type EnabledState *bool

// Useful variables to use when creating or updating users.
var (
	iTrue  = true
	iFalse = false

	Enabled  EnabledState = &iTrue
	Disabled EnabledState = &iFalse
)

// CommonOpts are the parameters that are shared between CreateOpts and
// UpdateOpts
type CommonOpts struct {
	// Either a name or username is required. When provided, the value must be
	// unique or a 409 conflict error will be returned. If you provide a name but
	// omit a username, the latter will be set to the former; and vice versa.
	Name, Username string

	// The ID of the tenant to which you want to assign this user.
	TenantID string

	// Indicates whether this user is enabled or not.
	Enabled EnabledState

	// The email address of this user.
	Email string
}

// CreateOpts represents the options needed when creating new users.
type CreateOpts CommonOpts

// CreateOptsBuilder describes struct types that can be accepted by the Create call.
type CreateOptsBuilder interface {
	ToUserCreateMap() (map[string]interface{}, error)
}

// ToUserCreateMap assembles a request body based on the contents of a CreateOpts.
func (opts CreateOpts) ToUserCreateMap() (map[string]interface{}, error) {
	m := make(map[string]interface{})

	if opts.Name == "" && opts.Username == "" {
		return m, errors.New("Either a Name or Username must be provided")
	}

	if opts.Name != "" {
		m["name"] = opts.Name
	}
	if opts.Username != "" {
		m["username"] = opts.Username
	}
	if opts.Enabled != nil {
		m["enabled"] = &opts.Enabled
	}
	if opts.Email != "" {
		m["email"] = opts.Email
	}
	if opts.TenantID != "" {
		m["tenant_id"] = opts.TenantID
	}

	return map[string]interface{}{"user": m}, nil
}

// Create is the operation responsible for creating new users.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToUserCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Post(rootURL(client), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})

	return res
}

// Get requests details on a single user, either by ID.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var result GetResult
	_, result.Err = client.Get(ResourceURL(client, id), &result.Body, nil)
	return result
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

	if opts.Name != "" {
		m["name"] = opts.Name
	}
	if opts.Username != "" {
		m["username"] = opts.Username
	}
	if opts.Enabled != nil {
		m["enabled"] = &opts.Enabled
	}
	if opts.Email != "" {
		m["email"] = opts.Email
	}
	if opts.TenantID != "" {
		m["tenant_id"] = opts.TenantID
	}

	return map[string]interface{}{"user": m}
}

// Update is the operation responsible for updating exist users by their UUID.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) UpdateResult {
	var result UpdateResult
	reqBody := opts.ToUserUpdateMap()
	_, result.Err = client.Put(ResourceURL(client, id), reqBody, &result.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return result
}

// Delete is the operation responsible for permanently deleting an API user.
func Delete(client *gophercloud.ServiceClient, id string) DeleteResult {
	var result DeleteResult
	_, result.Err = client.Delete(ResourceURL(client, id), nil)
	return result
}

func ListRoles(client *gophercloud.ServiceClient, tenantID, userID string) pagination.Pager {
	createPage := func(r pagination.PageResult) pagination.Page {
		return RolePage{pagination.SinglePageBase(r)}
	}

	return pagination.NewPager(client, listRolesURL(client, tenantID, userID), createPage)
}
