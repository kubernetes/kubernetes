// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// Key represents a public SSH key used to authenticate a user or deploy script.
type Key struct {
	ID    *int    `json:"id,omitempty"`
	Key   *string `json:"key,omitempty"`
	URL   *string `json:"url,omitempty"`
	Title *string `json:"title,omitempty"`
}

func (k Key) String() string {
	return Stringify(k)
}

// ListKeys lists the verified public keys for a user.  Passing the empty
// string will fetch keys for the authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/users/keys/#list-public-keys-for-a-user
func (s *UsersService) ListKeys(user string, opt *ListOptions) ([]Key, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("users/%v/keys", user)
	} else {
		u = "user/keys"
	}
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	keys := new([]Key)
	resp, err := s.client.Do(req, keys)
	if err != nil {
		return nil, resp, err
	}

	return *keys, resp, err
}

// GetKey fetches a single public key.
//
// GitHub API docs: http://developer.github.com/v3/users/keys/#get-a-single-public-key
func (s *UsersService) GetKey(id int) (*Key, *Response, error) {
	u := fmt.Sprintf("user/keys/%v", id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	key := new(Key)
	resp, err := s.client.Do(req, key)
	if err != nil {
		return nil, resp, err
	}

	return key, resp, err
}

// CreateKey adds a public key for the authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/users/keys/#create-a-public-key
func (s *UsersService) CreateKey(key *Key) (*Key, *Response, error) {
	u := "user/keys"

	req, err := s.client.NewRequest("POST", u, key)
	if err != nil {
		return nil, nil, err
	}

	k := new(Key)
	resp, err := s.client.Do(req, k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, err
}

// DeleteKey deletes a public key.
//
// GitHub API docs: http://developer.github.com/v3/users/keys/#delete-a-public-key
func (s *UsersService) DeleteKey(id int) (*Response, error) {
	u := fmt.Sprintf("user/keys/%v", id)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
