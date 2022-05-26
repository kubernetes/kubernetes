// Copyright 2019 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// createUserRequest is a subset of User and is used internally
// by CreateUser to pass only the known fields for the endpoint.
type createUserRequest struct {
	Login *string `json:"login,omitempty"`
	Email *string `json:"email,omitempty"`
}

// CreateUser creates a new user in GitHub Enterprise.
//
// GitHub Enterprise API docs: https://developer.github.com/enterprise/v3/enterprise-admin/users/#create-a-new-user
func (s *AdminService) CreateUser(ctx context.Context, login, email string) (*User, *Response, error) {
	u := "admin/users"

	userReq := &createUserRequest{
		Login: &login,
		Email: &email,
	}

	req, err := s.client.NewRequest("POST", u, userReq)
	if err != nil {
		return nil, nil, err
	}

	var user User
	resp, err := s.client.Do(ctx, req, &user)
	if err != nil {
		return nil, resp, err
	}

	return &user, resp, nil
}

// DeleteUser deletes a user in GitHub Enterprise.
//
// GitHub Enterprise API docs: https://developer.github.com/enterprise/v3/enterprise-admin/users/#delete-a-user
func (s *AdminService) DeleteUser(ctx context.Context, username string) (*Response, error) {
	u := "admin/users/" + username

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(ctx, req, nil)
	if err != nil {
		return resp, err
	}

	return resp, nil
}

// ImpersonateUserOptions represents the scoping for the OAuth token.
type ImpersonateUserOptions struct {
	Scopes []string `json:"scopes,omitempty"`
}

// OAuthAPP represents the GitHub Site Administrator OAuth app.
type OAuthAPP struct {
	URL      *string `json:"url,omitempty"`
	Name     *string `json:"name,omitempty"`
	ClientID *string `json:"client_id,omitempty"`
}

func (s OAuthAPP) String() string {
	return Stringify(s)
}

// UserAuthorization represents the impersonation response.
type UserAuthorization struct {
	ID             *int64     `json:"id,omitempty"`
	URL            *string    `json:"url,omitempty"`
	Scopes         []string   `json:"scopes,omitempty"`
	Token          *string    `json:"token,omitempty"`
	TokenLastEight *string    `json:"token_last_eight,omitempty"`
	HashedToken    *string    `json:"hashed_token,omitempty"`
	App            *OAuthAPP  `json:"app,omitempty"`
	Note           *string    `json:"note,omitempty"`
	NoteURL        *string    `json:"note_url,omitempty"`
	UpdatedAt      *Timestamp `json:"updated_at,omitempty"`
	CreatedAt      *Timestamp `json:"created_at,omitempty"`
	Fingerprint    *string    `json:"fingerprint,omitempty"`
}

// CreateUserImpersonation creates an impersonation OAuth token.
//
// GitHub Enterprise API docs: https://developer.github.com/enterprise/v3/enterprise-admin/users/#create-an-impersonation-oauth-token
func (s *AdminService) CreateUserImpersonation(ctx context.Context, username string, opts *ImpersonateUserOptions) (*UserAuthorization, *Response, error) {
	u := fmt.Sprintf("admin/users/%s/authorizations", username)

	req, err := s.client.NewRequest("POST", u, opts)
	if err != nil {
		return nil, nil, err
	}

	a := new(UserAuthorization)
	resp, err := s.client.Do(ctx, req, a)
	if err != nil {
		return nil, resp, err
	}

	return a, resp, nil
}

// DeleteUserImpersonation deletes an impersonation OAuth token.
//
// GitHub Enterprise API docs: https://developer.github.com/enterprise/v3/enterprise-admin/users/#delete-an-impersonation-oauth-token
func (s *AdminService) DeleteUserImpersonation(ctx context.Context, username string) (*Response, error) {
	u := fmt.Sprintf("admin/users/%s/authorizations", username)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(ctx, req, nil)
	if err != nil {
		return resp, err
	}

	return resp, nil
}
