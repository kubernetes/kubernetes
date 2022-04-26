// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "context"

// UserEmail represents user's email address
type UserEmail struct {
	Email      *string `json:"email,omitempty"`
	Primary    *bool   `json:"primary,omitempty"`
	Verified   *bool   `json:"verified,omitempty"`
	Visibility *string `json:"visibility,omitempty"`
}

// ListEmails lists all email addresses for the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#list-email-addresses-for-the-authenticated-user
func (s *UsersService) ListEmails(ctx context.Context, opts *ListOptions) ([]*UserEmail, *Response, error) {
	u := "user/emails"
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var emails []*UserEmail
	resp, err := s.client.Do(ctx, req, &emails)
	if err != nil {
		return nil, resp, err
	}

	return emails, resp, nil
}

// AddEmails adds email addresses of the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#add-an-email-address-for-the-authenticated-user
func (s *UsersService) AddEmails(ctx context.Context, emails []string) ([]*UserEmail, *Response, error) {
	u := "user/emails"
	req, err := s.client.NewRequest("POST", u, emails)
	if err != nil {
		return nil, nil, err
	}

	var e []*UserEmail
	resp, err := s.client.Do(ctx, req, &e)
	if err != nil {
		return nil, resp, err
	}

	return e, resp, nil
}

// DeleteEmails deletes email addresses from authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#delete-an-email-address-for-the-authenticated-user
func (s *UsersService) DeleteEmails(ctx context.Context, emails []string) (*Response, error) {
	u := "user/emails"
	req, err := s.client.NewRequest("DELETE", u, emails)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
