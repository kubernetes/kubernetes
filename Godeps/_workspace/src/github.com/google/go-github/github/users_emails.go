// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

// UserEmail represents user's email address
type UserEmail struct {
	Email    *string `json:"email,omitempty"`
	Primary  *bool   `json:"primary,omitempty"`
	Verified *bool   `json:"verified,omitempty"`
}

// ListEmails lists all email addresses for the authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/users/emails/#list-email-addresses-for-a-user
func (s *UsersService) ListEmails(opt *ListOptions) ([]UserEmail, *Response, error) {
	u := "user/emails"
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	emails := new([]UserEmail)
	resp, err := s.client.Do(req, emails)
	if err != nil {
		return nil, resp, err
	}

	return *emails, resp, err
}

// AddEmails adds email addresses of the authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/users/emails/#add-email-addresses
func (s *UsersService) AddEmails(emails []string) ([]UserEmail, *Response, error) {
	u := "user/emails"
	req, err := s.client.NewRequest("POST", u, emails)
	if err != nil {
		return nil, nil, err
	}

	e := new([]UserEmail)
	resp, err := s.client.Do(req, e)
	if err != nil {
		return nil, resp, err
	}

	return *e, resp, err
}

// DeleteEmails deletes email addresses from authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/users/emails/#delete-email-addresses
func (s *UsersService) DeleteEmails(emails []string) (*Response, error) {
	u := "user/emails"
	req, err := s.client.NewRequest("DELETE", u, emails)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
