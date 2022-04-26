// Copyright 2015 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// Scope models a GitHub authorization scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/oauth/#scopes
type Scope string

// This is the set of scopes for GitHub API V3
const (
	ScopeNone           Scope = "(no scope)" // REVISIT: is this actually returned, or just a documentation artifact?
	ScopeUser           Scope = "user"
	ScopeUserEmail      Scope = "user:email"
	ScopeUserFollow     Scope = "user:follow"
	ScopePublicRepo     Scope = "public_repo"
	ScopeRepo           Scope = "repo"
	ScopeRepoDeployment Scope = "repo_deployment"
	ScopeRepoStatus     Scope = "repo:status"
	ScopeDeleteRepo     Scope = "delete_repo"
	ScopeNotifications  Scope = "notifications"
	ScopeGist           Scope = "gist"
	ScopeReadRepoHook   Scope = "read:repo_hook"
	ScopeWriteRepoHook  Scope = "write:repo_hook"
	ScopeAdminRepoHook  Scope = "admin:repo_hook"
	ScopeAdminOrgHook   Scope = "admin:org_hook"
	ScopeReadOrg        Scope = "read:org"
	ScopeWriteOrg       Scope = "write:org"
	ScopeAdminOrg       Scope = "admin:org"
	ScopeReadPublicKey  Scope = "read:public_key"
	ScopeWritePublicKey Scope = "write:public_key"
	ScopeAdminPublicKey Scope = "admin:public_key"
	ScopeReadGPGKey     Scope = "read:gpg_key"
	ScopeWriteGPGKey    Scope = "write:gpg_key"
	ScopeAdminGPGKey    Scope = "admin:gpg_key"
	ScopeSecurityEvents Scope = "security_events"
)

// AuthorizationsService handles communication with the authorization related
// methods of the GitHub API.
//
// This service requires HTTP Basic Authentication; it cannot be accessed using
// an OAuth token.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/oauth_authorizations/
type AuthorizationsService service

// Authorization represents an individual GitHub authorization.
type Authorization struct {
	ID             *int64            `json:"id,omitempty"`
	URL            *string           `json:"url,omitempty"`
	Scopes         []Scope           `json:"scopes,omitempty"`
	Token          *string           `json:"token,omitempty"`
	TokenLastEight *string           `json:"token_last_eight,omitempty"`
	HashedToken    *string           `json:"hashed_token,omitempty"`
	App            *AuthorizationApp `json:"app,omitempty"`
	Note           *string           `json:"note,omitempty"`
	NoteURL        *string           `json:"note_url,omitempty"`
	UpdatedAt      *Timestamp        `json:"updated_at,omitempty"`
	CreatedAt      *Timestamp        `json:"created_at,omitempty"`
	Fingerprint    *string           `json:"fingerprint,omitempty"`

	// User is only populated by the Check and Reset methods.
	User *User `json:"user,omitempty"`
}

func (a Authorization) String() string {
	return Stringify(a)
}

// AuthorizationApp represents an individual GitHub app (in the context of authorization).
type AuthorizationApp struct {
	URL      *string `json:"url,omitempty"`
	Name     *string `json:"name,omitempty"`
	ClientID *string `json:"client_id,omitempty"`
}

func (a AuthorizationApp) String() string {
	return Stringify(a)
}

// Grant represents an OAuth application that has been granted access to an account.
type Grant struct {
	ID        *int64            `json:"id,omitempty"`
	URL       *string           `json:"url,omitempty"`
	App       *AuthorizationApp `json:"app,omitempty"`
	CreatedAt *Timestamp        `json:"created_at,omitempty"`
	UpdatedAt *Timestamp        `json:"updated_at,omitempty"`
	Scopes    []string          `json:"scopes,omitempty"`
}

func (g Grant) String() string {
	return Stringify(g)
}

// AuthorizationRequest represents a request to create an authorization.
type AuthorizationRequest struct {
	Scopes       []Scope `json:"scopes,omitempty"`
	Note         *string `json:"note,omitempty"`
	NoteURL      *string `json:"note_url,omitempty"`
	ClientID     *string `json:"client_id,omitempty"`
	ClientSecret *string `json:"client_secret,omitempty"`
	Fingerprint  *string `json:"fingerprint,omitempty"`
}

func (a AuthorizationRequest) String() string {
	return Stringify(a)
}

// AuthorizationUpdateRequest represents a request to update an authorization.
//
// Note that for any one update, you must only provide one of the "scopes"
// fields. That is, you may provide only one of "Scopes", or "AddScopes", or
// "RemoveScopes".
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/oauth_authorizations/#update-an-existing-authorization
type AuthorizationUpdateRequest struct {
	Scopes       []string `json:"scopes,omitempty"`
	AddScopes    []string `json:"add_scopes,omitempty"`
	RemoveScopes []string `json:"remove_scopes,omitempty"`
	Note         *string  `json:"note,omitempty"`
	NoteURL      *string  `json:"note_url,omitempty"`
	Fingerprint  *string  `json:"fingerprint,omitempty"`
}

func (a AuthorizationUpdateRequest) String() string {
	return Stringify(a)
}

// Check if an OAuth token is valid for a specific app.
//
// Note that this operation requires the use of BasicAuth, but where the
// username is the OAuth application clientID, and the password is its
// clientSecret. Invalid tokens will return a 404 Not Found.
//
// The returned Authorization.User field will be populated.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#check-a-token
func (s *AuthorizationsService) Check(ctx context.Context, clientID, accessToken string) (*Authorization, *Response, error) {
	u := fmt.Sprintf("applications/%v/token", clientID)

	reqBody := &struct {
		AccessToken string `json:"access_token"`
	}{AccessToken: accessToken}

	req, err := s.client.NewRequest("POST", u, reqBody)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Accept", mediaTypeOAuthAppPreview)

	a := new(Authorization)
	resp, err := s.client.Do(ctx, req, a)
	if err != nil {
		return nil, resp, err
	}

	return a, resp, nil
}

// Reset is used to reset a valid OAuth token without end user involvement.
// Applications must save the "token" property in the response, because changes
// take effect immediately.
//
// Note that this operation requires the use of BasicAuth, but where the
// username is the OAuth application clientID, and the password is its
// clientSecret. Invalid tokens will return a 404 Not Found.
//
// The returned Authorization.User field will be populated.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#reset-a-token
func (s *AuthorizationsService) Reset(ctx context.Context, clientID, accessToken string) (*Authorization, *Response, error) {
	u := fmt.Sprintf("applications/%v/token", clientID)

	reqBody := &struct {
		AccessToken string `json:"access_token"`
	}{AccessToken: accessToken}

	req, err := s.client.NewRequest("PATCH", u, reqBody)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Accept", mediaTypeOAuthAppPreview)

	a := new(Authorization)
	resp, err := s.client.Do(ctx, req, a)
	if err != nil {
		return nil, resp, err
	}

	return a, resp, nil
}

// Revoke an authorization for an application.
//
// Note that this operation requires the use of BasicAuth, but where the
// username is the OAuth application clientID, and the password is its
// clientSecret. Invalid tokens will return a 404 Not Found.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#delete-an-app-token
func (s *AuthorizationsService) Revoke(ctx context.Context, clientID, accessToken string) (*Response, error) {
	u := fmt.Sprintf("applications/%v/token", clientID)

	reqBody := &struct {
		AccessToken string `json:"access_token"`
	}{AccessToken: accessToken}

	req, err := s.client.NewRequest("DELETE", u, reqBody)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", mediaTypeOAuthAppPreview)

	return s.client.Do(ctx, req, nil)
}

// DeleteGrant deletes an OAuth application grant. Deleting an application's
// grant will also delete all OAuth tokens associated with the application for
// the user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#delete-an-app-authorization
func (s *AuthorizationsService) DeleteGrant(ctx context.Context, clientID, accessToken string) (*Response, error) {
	u := fmt.Sprintf("applications/%v/grant", clientID)

	reqBody := &struct {
		AccessToken string `json:"access_token"`
	}{AccessToken: accessToken}

	req, err := s.client.NewRequest("DELETE", u, reqBody)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", mediaTypeOAuthAppPreview)

	return s.client.Do(ctx, req, nil)
}

// CreateImpersonation creates an impersonation OAuth token.
//
// This requires admin permissions. With the returned Authorization.Token
// you can e.g. create or delete a user's public SSH key. NOTE: creating a
// new token automatically revokes an existing one.
//
// GitHub API docs: https://developer.github.com/enterprise/v3/enterprise-admin/users/#create-an-impersonation-oauth-token
func (s *AuthorizationsService) CreateImpersonation(ctx context.Context, username string, authReq *AuthorizationRequest) (*Authorization, *Response, error) {
	u := fmt.Sprintf("admin/users/%v/authorizations", username)
	req, err := s.client.NewRequest("POST", u, authReq)
	if err != nil {
		return nil, nil, err
	}

	a := new(Authorization)
	resp, err := s.client.Do(ctx, req, a)
	if err != nil {
		return nil, resp, err
	}
	return a, resp, nil
}

// DeleteImpersonation deletes an impersonation OAuth token.
//
// NOTE: there can be only one at a time.
//
// GitHub API docs: https://developer.github.com/enterprise/v3/enterprise-admin/users/#delete-an-impersonation-oauth-token
func (s *AuthorizationsService) DeleteImpersonation(ctx context.Context, username string) (*Response, error) {
	u := fmt.Sprintf("admin/users/%v/authorizations", username)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
