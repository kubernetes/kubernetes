//
// Copyright 2021, Sander van Harmelen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package gitlab

import (
	"errors"
	"fmt"
	"time"
)

// List a couple of standard errors.
var (
	ErrUserActivatePrevented   = errors.New("Cannot activate a user that is blocked by admin or by LDAP synchronization")
	ErrUserBlockPrevented      = errors.New("Cannot block a user that is already blocked by LDAP synchronization")
	ErrUserDeactivatePrevented = errors.New("Cannot deactivate a user that is blocked by admin or by LDAP synchronization, or that has any activity in past 180 days")
	ErrUserNotFound            = errors.New("User does not exist")
	ErrUserUnblockPrevented    = errors.New("Cannot unblock a user that is blocked by LDAP synchronization")
)

// UsersService handles communication with the user related methods of
// the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html
type UsersService struct {
	client *Client
}

// BasicUser included in other service responses (such as merge requests, pipelines, etc).
type BasicUser struct {
	ID        int        `json:"id"`
	Username  string     `json:"username"`
	Name      string     `json:"name"`
	State     string     `json:"state"`
	CreatedAt *time.Time `json:"created_at"`
	AvatarURL string     `json:"avatar_url"`
	WebURL    string     `json:"web_url"`
}

// User represents a GitLab user.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/users.html
type User struct {
	ID                             int                `json:"id"`
	Username                       string             `json:"username"`
	Email                          string             `json:"email"`
	Name                           string             `json:"name"`
	State                          string             `json:"state"`
	WebURL                         string             `json:"web_url"`
	CreatedAt                      *time.Time         `json:"created_at"`
	Bio                            string             `json:"bio"`
	Location                       string             `json:"location"`
	PublicEmail                    string             `json:"public_email"`
	Skype                          string             `json:"skype"`
	Linkedin                       string             `json:"linkedin"`
	Twitter                        string             `json:"twitter"`
	WebsiteURL                     string             `json:"website_url"`
	Organization                   string             `json:"organization"`
	ExternUID                      string             `json:"extern_uid"`
	Provider                       string             `json:"provider"`
	ThemeID                        int                `json:"theme_id"`
	LastActivityOn                 *ISOTime           `json:"last_activity_on"`
	ColorSchemeID                  int                `json:"color_scheme_id"`
	IsAdmin                        bool               `json:"is_admin"`
	AvatarURL                      string             `json:"avatar_url"`
	CanCreateGroup                 bool               `json:"can_create_group"`
	CanCreateProject               bool               `json:"can_create_project"`
	ProjectsLimit                  int                `json:"projects_limit"`
	CurrentSignInAt                *time.Time         `json:"current_sign_in_at"`
	LastSignInAt                   *time.Time         `json:"last_sign_in_at"`
	ConfirmedAt                    *time.Time         `json:"confirmed_at"`
	TwoFactorEnabled               bool               `json:"two_factor_enabled"`
	Note                           string             `json:"note"`
	Identities                     []*UserIdentity    `json:"identities"`
	External                       bool               `json:"external"`
	PrivateProfile                 bool               `json:"private_profile"`
	SharedRunnersMinutesLimit      int                `json:"shared_runners_minutes_limit"`
	ExtraSharedRunnersMinutesLimit int                `json:"extra_shared_runners_minutes_limit"`
	UsingLicenseSeat               bool               `json:"using_license_seat"`
	CustomAttributes               []*CustomAttribute `json:"custom_attributes"`
}

// UserIdentity represents a user identity.
type UserIdentity struct {
	Provider  string `json:"provider"`
	ExternUID string `json:"extern_uid"`
}

// ListUsersOptions represents the available ListUsers() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#list-users
type ListUsersOptions struct {
	ListOptions
	Active          *bool `url:"active,omitempty" json:"active,omitempty"`
	Blocked         *bool `url:"blocked,omitempty" json:"blocked,omitempty"`
	ExcludeInternal *bool `url:"exclude_internal,omitempty" json:"exclude_internal,omitempty"`

	// The options below are only available for admins.
	Search               *string    `url:"search,omitempty" json:"search,omitempty"`
	Username             *string    `url:"username,omitempty" json:"username,omitempty"`
	ExternalUID          *string    `url:"extern_uid,omitempty" json:"extern_uid,omitempty"`
	Provider             *string    `url:"provider,omitempty" json:"provider,omitempty"`
	CreatedBefore        *time.Time `url:"created_before,omitempty" json:"created_before,omitempty"`
	CreatedAfter         *time.Time `url:"created_after,omitempty" json:"created_after,omitempty"`
	OrderBy              *string    `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort                 *string    `url:"sort,omitempty" json:"sort,omitempty"`
	TwoFactor            *string    `url:"two_factor,omitempty" json:"two_factor,omitempty"`
	Admins               *bool      `url:"admins,omitempty" json:"admins,omitempty"`
	External             *bool      `url:"external,omitempty" json:"external,omitempty"`
	WithoutProjects      *bool      `url:"without_projects,omitempty" json:"without_projects,omitempty"`
	WithCustomAttributes *bool      `url:"with_custom_attributes,omitempty" json:"with_custom_attributes,omitempty"`
}

// ListUsers gets a list of users.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#list-users
func (s *UsersService) ListUsers(opt *ListUsersOptions, options ...RequestOptionFunc) ([]*User, *Response, error) {
	req, err := s.client.NewRequest("GET", "users", opt, options)
	if err != nil {
		return nil, nil, err
	}

	var usr []*User
	resp, err := s.client.Do(req, &usr)
	if err != nil {
		return nil, resp, err
	}

	return usr, resp, err
}

// GetUser gets a single user.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#single-user
func (s *UsersService) GetUser(user int, options ...RequestOptionFunc) (*User, *Response, error) {
	u := fmt.Sprintf("users/%d", user)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	usr := new(User)
	resp, err := s.client.Do(req, usr)
	if err != nil {
		return nil, resp, err
	}

	return usr, resp, err
}

// CreateUserOptions represents the available CreateUser() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#user-creation
type CreateUserOptions struct {
	Email               *string `url:"email,omitempty" json:"email,omitempty"`
	Password            *string `url:"password,omitempty" json:"password,omitempty"`
	ResetPassword       *bool   `url:"reset_password,omitempty" json:"reset_password,omitempty"`
	ForceRandomPassword *bool   `url:"force_random_password,omitempty" json:"force_random_password,omitempty"`
	Username            *string `url:"username,omitempty" json:"username,omitempty"`
	Name                *string `url:"name,omitempty" json:"name,omitempty"`
	Skype               *string `url:"skype,omitempty" json:"skype,omitempty"`
	Linkedin            *string `url:"linkedin,omitempty" json:"linkedin,omitempty"`
	Twitter             *string `url:"twitter,omitempty" json:"twitter,omitempty"`
	WebsiteURL          *string `url:"website_url,omitempty" json:"website_url,omitempty"`
	Organization        *string `url:"organization,omitempty" json:"organization,omitempty"`
	ProjectsLimit       *int    `url:"projects_limit,omitempty" json:"projects_limit,omitempty"`
	ExternUID           *string `url:"extern_uid,omitempty" json:"extern_uid,omitempty"`
	Provider            *string `url:"provider,omitempty" json:"provider,omitempty"`
	Bio                 *string `url:"bio,omitempty" json:"bio,omitempty"`
	Location            *string `url:"location,omitempty" json:"location,omitempty"`
	Admin               *bool   `url:"admin,omitempty" json:"admin,omitempty"`
	CanCreateGroup      *bool   `url:"can_create_group,omitempty" json:"can_create_group,omitempty"`
	SkipConfirmation    *bool   `url:"skip_confirmation,omitempty" json:"skip_confirmation,omitempty"`
	External            *bool   `url:"external,omitempty" json:"external,omitempty"`
	PrivateProfile      *bool   `url:"private_profile,omitempty" json:"private_profile,omitempty"`
}

// CreateUser creates a new user. Note only administrators can create new users.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#user-creation
func (s *UsersService) CreateUser(opt *CreateUserOptions, options ...RequestOptionFunc) (*User, *Response, error) {
	req, err := s.client.NewRequest("POST", "users", opt, options)
	if err != nil {
		return nil, nil, err
	}

	usr := new(User)
	resp, err := s.client.Do(req, usr)
	if err != nil {
		return nil, resp, err
	}

	return usr, resp, err
}

// ModifyUserOptions represents the available ModifyUser() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#user-modification
type ModifyUserOptions struct {
	Email              *string `url:"email,omitempty" json:"email,omitempty"`
	Password           *string `url:"password,omitempty" json:"password,omitempty"`
	Username           *string `url:"username,omitempty" json:"username,omitempty"`
	Name               *string `url:"name,omitempty" json:"name,omitempty"`
	Skype              *string `url:"skype,omitempty" json:"skype,omitempty"`
	Linkedin           *string `url:"linkedin,omitempty" json:"linkedin,omitempty"`
	Twitter            *string `url:"twitter,omitempty" json:"twitter,omitempty"`
	WebsiteURL         *string `url:"website_url,omitempty" json:"website_url,omitempty"`
	Organization       *string `url:"organization,omitempty" json:"organization,omitempty"`
	ProjectsLimit      *int    `url:"projects_limit,omitempty" json:"projects_limit,omitempty"`
	ExternUID          *string `url:"extern_uid,omitempty" json:"extern_uid,omitempty"`
	Provider           *string `url:"provider,omitempty" json:"provider,omitempty"`
	Bio                *string `url:"bio,omitempty" json:"bio,omitempty"`
	Location           *string `url:"location,omitempty" json:"location,omitempty"`
	Admin              *bool   `url:"admin,omitempty" json:"admin,omitempty"`
	CanCreateGroup     *bool   `url:"can_create_group,omitempty" json:"can_create_group,omitempty"`
	SkipReconfirmation *bool   `url:"skip_reconfirmation,omitempty" json:"skip_reconfirmation,omitempty"`
	External           *bool   `url:"external,omitempty" json:"external,omitempty"`
	PrivateProfile     *bool   `url:"private_profile,omitempty" json:"private_profile,omitempty"`
}

// ModifyUser modifies an existing user. Only administrators can change attributes
// of a user.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#user-modification
func (s *UsersService) ModifyUser(user int, opt *ModifyUserOptions, options ...RequestOptionFunc) (*User, *Response, error) {
	u := fmt.Sprintf("users/%d", user)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	usr := new(User)
	resp, err := s.client.Do(req, usr)
	if err != nil {
		return nil, resp, err
	}

	return usr, resp, err
}

// DeleteUser deletes a user. Available only for administrators. This is an
// idempotent function, calling this function for a non-existent user id still
// returns a status code 200 OK. The JSON response differs if the user was
// actually deleted or not. In the former the user is returned and in the
// latter not.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#user-deletion
func (s *UsersService) DeleteUser(user int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("users/%d", user)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// CurrentUser gets currently authenticated user.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#current-user
func (s *UsersService) CurrentUser(options ...RequestOptionFunc) (*User, *Response, error) {
	req, err := s.client.NewRequest("GET", "user", nil, options)
	if err != nil {
		return nil, nil, err
	}

	usr := new(User)
	resp, err := s.client.Do(req, usr)
	if err != nil {
		return nil, resp, err
	}

	return usr, resp, err
}

// SSHKey represents a SSH key.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#list-ssh-keys
type SSHKey struct {
	ID        int        `json:"id"`
	Title     string     `json:"title"`
	Key       string     `json:"key"`
	CreatedAt *time.Time `json:"created_at"`
}

// ListSSHKeys gets a list of currently authenticated user's SSH keys.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#list-ssh-keys
func (s *UsersService) ListSSHKeys(options ...RequestOptionFunc) ([]*SSHKey, *Response, error) {
	req, err := s.client.NewRequest("GET", "user/keys", nil, options)
	if err != nil {
		return nil, nil, err
	}

	var k []*SSHKey
	resp, err := s.client.Do(req, &k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, err
}

// ListSSHKeysForUserOptions represents the available ListSSHKeysForUser() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#list-ssh-keys-for-user
type ListSSHKeysForUserOptions ListOptions

// ListSSHKeysForUser gets a list of a specified user's SSH keys. Available
// only for admin
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#list-ssh-keys-for-user
func (s *UsersService) ListSSHKeysForUser(user int, opt *ListSSHKeysForUserOptions, options ...RequestOptionFunc) ([]*SSHKey, *Response, error) {
	u := fmt.Sprintf("users/%d/keys", user)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var k []*SSHKey
	resp, err := s.client.Do(req, &k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, err
}

// GetSSHKey gets a single key.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#single-ssh-key
func (s *UsersService) GetSSHKey(key int, options ...RequestOptionFunc) (*SSHKey, *Response, error) {
	u := fmt.Sprintf("user/keys/%d", key)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	k := new(SSHKey)
	resp, err := s.client.Do(req, k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, err
}

// AddSSHKeyOptions represents the available AddSSHKey() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#add-ssh-key
type AddSSHKeyOptions struct {
	Title     *string  `url:"title,omitempty" json:"title,omitempty"`
	Key       *string  `url:"key,omitempty" json:"key,omitempty"`
	ExpiresAt *ISOTime `url:"expires_at,omitempty" json:"expires_at,omitempty"`
}

// AddSSHKey creates a new key owned by the currently authenticated user.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#add-ssh-key
func (s *UsersService) AddSSHKey(opt *AddSSHKeyOptions, options ...RequestOptionFunc) (*SSHKey, *Response, error) {
	req, err := s.client.NewRequest("POST", "user/keys", opt, options)
	if err != nil {
		return nil, nil, err
	}

	k := new(SSHKey)
	resp, err := s.client.Do(req, k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, err
}

// AddSSHKeyForUser creates new key owned by specified user. Available only for
// admin.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#add-ssh-key-for-user
func (s *UsersService) AddSSHKeyForUser(user int, opt *AddSSHKeyOptions, options ...RequestOptionFunc) (*SSHKey, *Response, error) {
	u := fmt.Sprintf("users/%d/keys", user)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	k := new(SSHKey)
	resp, err := s.client.Do(req, k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, err
}

// DeleteSSHKey deletes key owned by currently authenticated user. This is an
// idempotent function and calling it on a key that is already deleted or not
// available results in 200 OK.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#delete-ssh-key-for-current-owner
func (s *UsersService) DeleteSSHKey(key int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("user/keys/%d", key)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// DeleteSSHKeyForUser deletes key owned by a specified user. Available only
// for admin.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#delete-ssh-key-for-given-user
func (s *UsersService) DeleteSSHKeyForUser(user, key int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("users/%d/keys/%d", user, key)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// BlockUser blocks the specified user. Available only for admin.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#block-user
func (s *UsersService) BlockUser(user int, options ...RequestOptionFunc) error {
	u := fmt.Sprintf("users/%d/block", user)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return err
	}

	resp, err := s.client.Do(req, nil)
	if err != nil && resp == nil {
		return err
	}

	switch resp.StatusCode {
	case 201:
		return nil
	case 403:
		return ErrUserBlockPrevented
	case 404:
		return ErrUserNotFound
	default:
		return fmt.Errorf("Received unexpected result code: %d", resp.StatusCode)
	}
}

// UnblockUser unblocks the specified user. Available only for admin.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#unblock-user
func (s *UsersService) UnblockUser(user int, options ...RequestOptionFunc) error {
	u := fmt.Sprintf("users/%d/unblock", user)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return err
	}

	resp, err := s.client.Do(req, nil)
	if err != nil && resp == nil {
		return err
	}

	switch resp.StatusCode {
	case 201:
		return nil
	case 403:
		return ErrUserUnblockPrevented
	case 404:
		return ErrUserNotFound
	default:
		return fmt.Errorf("Received unexpected result code: %d", resp.StatusCode)
	}
}

// DeactivateUser deactivate the specified user. Available only for admin.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#deactivate-user
func (s *UsersService) DeactivateUser(user int, options ...RequestOptionFunc) error {
	u := fmt.Sprintf("users/%d/deactivate", user)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return err
	}

	resp, err := s.client.Do(req, nil)
	if err != nil && resp == nil {
		return err
	}

	switch resp.StatusCode {
	case 201:
		return nil
	case 403:
		return ErrUserDeactivatePrevented
	case 404:
		return ErrUserNotFound
	default:
		return fmt.Errorf("Received unexpected result code: %d", resp.StatusCode)
	}
}

// ActivateUser activate the specified user. Available only for admin.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#activate-user
func (s *UsersService) ActivateUser(user int, options ...RequestOptionFunc) error {
	u := fmt.Sprintf("users/%d/activate", user)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return err
	}

	resp, err := s.client.Do(req, nil)
	if err != nil && resp == nil {
		return err
	}

	switch resp.StatusCode {
	case 201:
		return nil
	case 403:
		return ErrUserActivatePrevented
	case 404:
		return ErrUserNotFound
	default:
		return fmt.Errorf("Received unexpected result code: %d", resp.StatusCode)
	}
}

// Email represents an Email.
//
// GitLab API docs: https://doc.gitlab.com/ce/api/users.html#list-emails
type Email struct {
	ID    int    `json:"id"`
	Email string `json:"email"`
}

// ListEmails gets a list of currently authenticated user's Emails.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#list-emails
func (s *UsersService) ListEmails(options ...RequestOptionFunc) ([]*Email, *Response, error) {
	req, err := s.client.NewRequest("GET", "user/emails", nil, options)
	if err != nil {
		return nil, nil, err
	}

	var e []*Email
	resp, err := s.client.Do(req, &e)
	if err != nil {
		return nil, resp, err
	}

	return e, resp, err
}

// ListEmailsForUserOptions represents the available ListEmailsForUser() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#list-emails-for-user
type ListEmailsForUserOptions ListOptions

// ListEmailsForUser gets a list of a specified user's Emails. Available
// only for admin
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#list-emails-for-user
func (s *UsersService) ListEmailsForUser(user int, opt *ListEmailsForUserOptions, options ...RequestOptionFunc) ([]*Email, *Response, error) {
	u := fmt.Sprintf("users/%d/emails", user)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var e []*Email
	resp, err := s.client.Do(req, &e)
	if err != nil {
		return nil, resp, err
	}

	return e, resp, err
}

// GetEmail gets a single email.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#single-email
func (s *UsersService) GetEmail(email int, options ...RequestOptionFunc) (*Email, *Response, error) {
	u := fmt.Sprintf("user/emails/%d", email)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	e := new(Email)
	resp, err := s.client.Do(req, e)
	if err != nil {
		return nil, resp, err
	}

	return e, resp, err
}

// AddEmailOptions represents the available AddEmail() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#add-email
type AddEmailOptions struct {
	Email *string `url:"email,omitempty" json:"email,omitempty"`
}

// AddEmail creates a new email owned by the currently authenticated user.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#add-email
func (s *UsersService) AddEmail(opt *AddEmailOptions, options ...RequestOptionFunc) (*Email, *Response, error) {
	req, err := s.client.NewRequest("POST", "user/emails", opt, options)
	if err != nil {
		return nil, nil, err
	}

	e := new(Email)
	resp, err := s.client.Do(req, e)
	if err != nil {
		return nil, resp, err
	}

	return e, resp, err
}

// AddEmailForUser creates new email owned by specified user. Available only for
// admin.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/users.html#add-email-for-user
func (s *UsersService) AddEmailForUser(user int, opt *AddEmailOptions, options ...RequestOptionFunc) (*Email, *Response, error) {
	u := fmt.Sprintf("users/%d/emails", user)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	e := new(Email)
	resp, err := s.client.Do(req, e)
	if err != nil {
		return nil, resp, err
	}

	return e, resp, err
}

// DeleteEmail deletes email owned by currently authenticated user. This is an
// idempotent function and calling it on a key that is already deleted or not
// available results in 200 OK.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#delete-email-for-current-owner
func (s *UsersService) DeleteEmail(email int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("user/emails/%d", email)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// DeleteEmailForUser deletes email owned by a specified user. Available only
// for admin.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#delete-email-for-given-user
func (s *UsersService) DeleteEmailForUser(user, email int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("users/%d/emails/%d", user, email)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ImpersonationToken represents an impersonation token.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#get-all-impersonation-tokens-of-a-user
type ImpersonationToken struct {
	ID        int        `json:"id"`
	Name      string     `json:"name"`
	Active    bool       `json:"active"`
	Token     string     `json:"token"`
	Scopes    []string   `json:"scopes"`
	Revoked   bool       `json:"revoked"`
	CreatedAt *time.Time `json:"created_at"`
	ExpiresAt *ISOTime   `json:"expires_at"`
}

// GetAllImpersonationTokensOptions represents the available
// GetAllImpersonationTokens() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#get-all-impersonation-tokens-of-a-user
type GetAllImpersonationTokensOptions struct {
	ListOptions
	State *string `url:"state,omitempty" json:"state,omitempty"`
}

// GetAllImpersonationTokens retrieves all impersonation tokens of a user.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#get-all-impersonation-tokens-of-a-user
func (s *UsersService) GetAllImpersonationTokens(user int, opt *GetAllImpersonationTokensOptions, options ...RequestOptionFunc) ([]*ImpersonationToken, *Response, error) {
	u := fmt.Sprintf("users/%d/impersonation_tokens", user)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var ts []*ImpersonationToken
	resp, err := s.client.Do(req, &ts)
	if err != nil {
		return nil, resp, err
	}

	return ts, resp, err
}

// GetImpersonationToken retrieves an impersonation token of a user.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#get-an-impersonation-token-of-a-user
func (s *UsersService) GetImpersonationToken(user, token int, options ...RequestOptionFunc) (*ImpersonationToken, *Response, error) {
	u := fmt.Sprintf("users/%d/impersonation_tokens/%d", user, token)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	t := new(ImpersonationToken)
	resp, err := s.client.Do(req, &t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// CreateImpersonationTokenOptions represents the available
// CreateImpersonationToken() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#create-an-impersonation-token
type CreateImpersonationTokenOptions struct {
	Name      *string    `url:"name,omitempty" json:"name,omitempty"`
	Scopes    *[]string  `url:"scopes,omitempty" json:"scopes,omitempty"`
	ExpiresAt *time.Time `url:"expires_at,omitempty" json:"expires_at,omitempty"`
}

// CreateImpersonationToken creates an impersonation token.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#create-an-impersonation-token
func (s *UsersService) CreateImpersonationToken(user int, opt *CreateImpersonationTokenOptions, options ...RequestOptionFunc) (*ImpersonationToken, *Response, error) {
	u := fmt.Sprintf("users/%d/impersonation_tokens", user)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	t := new(ImpersonationToken)
	resp, err := s.client.Do(req, &t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// RevokeImpersonationToken revokes an impersonation token.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#revoke-an-impersonation-token
func (s *UsersService) RevokeImpersonationToken(user, token int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("users/%d/impersonation_tokens/%d", user, token)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// UserActivity represents an entry in the user/activities response
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#get-user-activities-admin-only
type UserActivity struct {
	Username       string   `json:"username"`
	LastActivityOn *ISOTime `json:"last_activity_on"`
}

// GetUserActivitiesOptions represents the options for GetUserActivities
//
// GitLap API docs:
// https://docs.gitlab.com/ce/api/users.html#get-user-activities-admin-only
type GetUserActivitiesOptions struct {
	ListOptions
	From *ISOTime `url:"from,omitempty" json:"from,omitempty"`
}

// GetUserActivities retrieves user activities (admin only)
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#get-user-activities-admin-only
func (s *UsersService) GetUserActivities(opt *GetUserActivitiesOptions, options ...RequestOptionFunc) ([]*UserActivity, *Response, error) {
	req, err := s.client.NewRequest("GET", "user/activities", opt, options)
	if err != nil {
		return nil, nil, err
	}

	var t []*UserActivity
	resp, err := s.client.Do(req, &t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// UserStatus represents the current status of a user
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#user-status
type UserStatus struct {
	Emoji       string `json:"emoji"`
	Message     string `json:"message"`
	MessageHTML string `json:"message_html"`
}

// CurrentUserStatus retrieves the user status
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#user-status
func (s *UsersService) CurrentUserStatus(options ...RequestOptionFunc) (*UserStatus, *Response, error) {
	req, err := s.client.NewRequest("GET", "user/status", nil, options)
	if err != nil {
		return nil, nil, err
	}

	status := new(UserStatus)
	resp, err := s.client.Do(req, status)
	if err != nil {
		return nil, resp, err
	}

	return status, resp, err
}

// GetUserStatus retrieves a user's status
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#get-the-status-of-a-user
func (s *UsersService) GetUserStatus(user int, options ...RequestOptionFunc) (*UserStatus, *Response, error) {
	u := fmt.Sprintf("users/%d/status", user)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	status := new(UserStatus)
	resp, err := s.client.Do(req, status)
	if err != nil {
		return nil, resp, err
	}

	return status, resp, err
}

// UserStatusOptions represents the options required to set the status
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#set-user-status
type UserStatusOptions struct {
	Emoji   *string `url:"emoji,omitempty" json:"emoji,omitempty"`
	Message *string `url:"message,omitempty" json:"message,omitempty"`
}

// SetUserStatus sets the user's status
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/users.html#set-user-status
func (s *UsersService) SetUserStatus(opt *UserStatusOptions, options ...RequestOptionFunc) (*UserStatus, *Response, error) {
	req, err := s.client.NewRequest("PUT", "user/status", opt, options)
	if err != nil {
		return nil, nil, err
	}

	status := new(UserStatus)
	resp, err := s.client.Do(req, status)
	if err != nil {
		return nil, resp, err
	}

	return status, resp, err
}

// UserMembership represents a membership of the user in a namespace or project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/users.html#user-memberships-admin-only
type UserMembership struct {
	SourceID    int              `json:"source_id"`
	SourceName  string           `json:"source_name"`
	SourceType  string           `json:"source_type"`
	AccessLevel AccessLevelValue `json:"access_level"`
}

// GetUserMembershipOptions represents the options available to query user memberships.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/users.html#user-memberships-admin-only
type GetUserMembershipOptions struct {
	ListOptions
	Type *string `url:"type,omitempty" json:"type,omitempty"`
}

// GetUserMemberships retrieves a list of the user's memberships.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/users.html#user-memberships-admin-only
func (s *UsersService) GetUserMemberships(user int, opt *GetUserMembershipOptions, options ...RequestOptionFunc) ([]*UserMembership, *Response, error) {
	u := fmt.Sprintf("users/%d/memberships", user)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var m []*UserMembership
	resp, err := s.client.Do(req, &m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}
