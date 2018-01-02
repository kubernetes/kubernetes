package users

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/groups"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/projects"
	"github.com/gophercloud/gophercloud/pagination"
)

// Option is a specific option defined at the API to enable features
// on a user account.
type Option string

const (
	IgnoreChangePasswordUponFirstUse Option = "ignore_change_password_upon_first_use"
	IgnorePasswordExpiry             Option = "ignore_password_expiry"
	IgnoreLockoutFailureAttempts     Option = "ignore_lockout_failure_attempts"
	MultiFactorAuthRules             Option = "multi_factor_auth_rules"
	MultiFactorAuthEnabled           Option = "multi_factor_auth_enabled"
)

// ListOptsBuilder allows extensions to add additional parameters to
// the List request
type ListOptsBuilder interface {
	ToUserListQuery() (string, error)
}

// ListOpts provides options to filter the List results.
type ListOpts struct {
	// DomainID filters the response by a domain ID.
	DomainID string `q:"domain_id"`

	// Enabled filters the response by enabled users.
	Enabled *bool `q:"enabled"`

	// IdpID filters the response by an Identity Provider ID.
	IdPID string `q:"idp_id"`

	// Name filters the response by username.
	Name string `q:"name"`

	// PasswordExpiresAt filters the response based on expiring passwords.
	PasswordExpiresAt string `q:"password_expires_at"`

	// ProtocolID filters the response by protocol ID.
	ProtocolID string `q:"protocol_id"`

	// UniqueID filters the response by unique ID.
	UniqueID string `q:"unique_id"`
}

// ToUserListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToUserListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List enumerates the Users to which the current token has access.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToUserListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return UserPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get retrieves details on a single user, by ID.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// CreateOptsBuilder allows extensions to add additional parameters to
// the Create request.
type CreateOptsBuilder interface {
	ToUserCreateMap() (map[string]interface{}, error)
}

// CreateOpts provides options used to create a user.
type CreateOpts struct {
	// Name is the name of the new user.
	Name string `json:"name" required:"true"`

	// DefaultProjectID is the ID of the default project of the user.
	DefaultProjectID string `json:"default_project_id,omitempty"`

	// Description is a description of the user.
	Description string `json:"description,omitempty"`

	// DomainID is the ID of the domain the user belongs to.
	DomainID string `json:"domain_id,omitempty"`

	// Enabled sets the user status to enabled or disabled.
	Enabled *bool `json:"enabled,omitempty"`

	// Extra is free-form extra key/value pairs to describe the user.
	Extra map[string]interface{} `json:"-"`

	// Options are defined options in the API to enable certain features.
	Options map[Option]interface{} `json:"options,omitempty"`

	// Password is the password of the new user.
	Password string `json:"password,omitempty"`
}

// ToUserCreateMap formats a CreateOpts into a create request.
func (opts CreateOpts) ToUserCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "user")
	if err != nil {
		return nil, err
	}

	if opts.Extra != nil {
		if v, ok := b["user"].(map[string]interface{}); ok {
			for key, value := range opts.Extra {
				v[key] = value
			}
		}
	}

	return b, nil
}

// Create creates a new User.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToUserCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{201},
	})
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to
// the Update request.
type UpdateOptsBuilder interface {
	ToUserUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts provides options for updating a user account.
type UpdateOpts struct {
	// Name is the name of the new user.
	Name string `json:"name,omitempty"`

	// DefaultProjectID is the ID of the default project of the user.
	DefaultProjectID string `json:"default_project_id,omitempty"`

	// Description is a description of the user.
	Description string `json:"description,omitempty"`

	// DomainID is the ID of the domain the user belongs to.
	DomainID string `json:"domain_id,omitempty"`

	// Enabled sets the user status to enabled or disabled.
	Enabled *bool `json:"enabled,omitempty"`

	// Extra is free-form extra key/value pairs to describe the user.
	Extra map[string]interface{} `json:"-"`

	// Options are defined options in the API to enable certain features.
	Options map[Option]interface{} `json:"options,omitempty"`

	// Password is the password of the new user.
	Password string `json:"password,omitempty"`
}

// ToUserUpdateMap formats a UpdateOpts into an update request.
func (opts UpdateOpts) ToUserUpdateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "user")
	if err != nil {
		return nil, err
	}

	if opts.Extra != nil {
		if v, ok := b["user"].(map[string]interface{}); ok {
			for key, value := range opts.Extra {
				v[key] = value
			}
		}
	}

	return b, nil
}

// Update updates an existing User.
func Update(client *gophercloud.ServiceClient, userID string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToUserUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Patch(updateURL(client, userID), &b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete deletes a user.
func Delete(client *gophercloud.ServiceClient, userID string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, userID), nil)
	return
}

// ListGroups enumerates groups user belongs to.
func ListGroups(client *gophercloud.ServiceClient, userID string) pagination.Pager {
	url := listGroupsURL(client, userID)
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return groups.GroupPage{LinkedPageBase: pagination.LinkedPageBase{PageResult: r}}
	})
}

// ListProjects enumerates groups user belongs to.
func ListProjects(client *gophercloud.ServiceClient, userID string) pagination.Pager {
	url := listProjectsURL(client, userID)
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return projects.ProjectPage{LinkedPageBase: pagination.LinkedPageBase{PageResult: r}}
	})
}

// ListInGroup enumerates users that belong to a group.
func ListInGroup(client *gophercloud.ServiceClient, groupID string, opts ListOptsBuilder) pagination.Pager {
	url := listInGroupURL(client, groupID)
	if opts != nil {
		query, err := opts.ToUserListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return UserPage{pagination.LinkedPageBase{PageResult: r}}
	})
}
