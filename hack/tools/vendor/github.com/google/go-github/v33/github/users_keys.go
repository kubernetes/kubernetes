// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// Key represents a public SSH key used to authenticate a user or deploy script.
type Key struct {
	ID        *int64     `json:"id,omitempty"`
	Key       *string    `json:"key,omitempty"`
	URL       *string    `json:"url,omitempty"`
	Title     *string    `json:"title,omitempty"`
	ReadOnly  *bool      `json:"read_only,omitempty"`
	CreatedAt *Timestamp `json:"created_at,omitempty"`
}

func (k Key) String() string {
	return Stringify(k)
}

// ListKeys lists the verified public keys for a user. Passing the empty
// string will fetch keys for the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#list-public-ssh-keys-for-the-authenticated-user
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#list-public-keys-for-a-user
func (s *UsersService) ListKeys(ctx context.Context, user string, opts *ListOptions) ([]*Key, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("users/%v/keys", user)
	} else {
		u = "user/keys"
	}
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var keys []*Key
	resp, err := s.client.Do(ctx, req, &keys)
	if err != nil {
		return nil, resp, err
	}

	return keys, resp, nil
}

// GetKey fetches a single public key.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#get-a-public-ssh-key-for-the-authenticated-user
func (s *UsersService) GetKey(ctx context.Context, id int64) (*Key, *Response, error) {
	u := fmt.Sprintf("user/keys/%v", id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	key := new(Key)
	resp, err := s.client.Do(ctx, req, key)
	if err != nil {
		return nil, resp, err
	}

	return key, resp, nil
}

// CreateKey adds a public key for the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#create-a-public-ssh-key-for-the-authenticated-user
func (s *UsersService) CreateKey(ctx context.Context, key *Key) (*Key, *Response, error) {
	u := "user/keys"

	req, err := s.client.NewRequest("POST", u, key)
	if err != nil {
		return nil, nil, err
	}

	k := new(Key)
	resp, err := s.client.Do(ctx, req, k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, nil
}

// DeleteKey deletes a public key.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#delete-a-public-ssh-key-for-the-authenticated-user
func (s *UsersService) DeleteKey(ctx context.Context, id int64) (*Response, error) {
	u := fmt.Sprintf("user/keys/%v", id)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
