// Copyright 2016 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"time"
)

// GPGKey represents a GitHub user's public GPG key used to verify GPG signed commits and tags.
//
// https://developer.github.com/changes/2016-04-04-git-signing-api-preview/
type GPGKey struct {
	ID                *int64      `json:"id,omitempty"`
	PrimaryKeyID      *int64      `json:"primary_key_id,omitempty"`
	KeyID             *string     `json:"key_id,omitempty"`
	PublicKey         *string     `json:"public_key,omitempty"`
	Emails            []*GPGEmail `json:"emails,omitempty"`
	Subkeys           []*GPGKey   `json:"subkeys,omitempty"`
	CanSign           *bool       `json:"can_sign,omitempty"`
	CanEncryptComms   *bool       `json:"can_encrypt_comms,omitempty"`
	CanEncryptStorage *bool       `json:"can_encrypt_storage,omitempty"`
	CanCertify        *bool       `json:"can_certify,omitempty"`
	CreatedAt         *time.Time  `json:"created_at,omitempty"`
	ExpiresAt         *time.Time  `json:"expires_at,omitempty"`
}

// String stringifies a GPGKey.
func (k GPGKey) String() string {
	return Stringify(k)
}

// GPGEmail represents an email address associated to a GPG key.
type GPGEmail struct {
	Email    *string `json:"email,omitempty"`
	Verified *bool   `json:"verified,omitempty"`
}

// ListGPGKeys lists the public GPG keys for a user. Passing the empty
// string will fetch keys for the authenticated user. It requires authentication
// via Basic Auth or via OAuth with at least read:gpg_key scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#list-gpg-keys-for-the-authenticated-user
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#list-gpg-keys-for-a-user
func (s *UsersService) ListGPGKeys(ctx context.Context, user string, opts *ListOptions) ([]*GPGKey, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("users/%v/gpg_keys", user)
	} else {
		u = "user/gpg_keys"
	}
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var keys []*GPGKey
	resp, err := s.client.Do(ctx, req, &keys)
	if err != nil {
		return nil, resp, err
	}

	return keys, resp, nil
}

// GetGPGKey gets extended details for a single GPG key. It requires authentication
// via Basic Auth or via OAuth with at least read:gpg_key scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#get-a-gpg-key-for-the-authenticated-user
func (s *UsersService) GetGPGKey(ctx context.Context, id int64) (*GPGKey, *Response, error) {
	u := fmt.Sprintf("user/gpg_keys/%v", id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	key := &GPGKey{}
	resp, err := s.client.Do(ctx, req, key)
	if err != nil {
		return nil, resp, err
	}

	return key, resp, nil
}

// CreateGPGKey creates a GPG key. It requires authenticatation via Basic Auth
// or OAuth with at least write:gpg_key scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#create-a-gpg-key
func (s *UsersService) CreateGPGKey(ctx context.Context, armoredPublicKey string) (*GPGKey, *Response, error) {
	gpgKey := &struct {
		ArmoredPublicKey string `json:"armored_public_key"`
	}{ArmoredPublicKey: armoredPublicKey}
	req, err := s.client.NewRequest("POST", "user/gpg_keys", gpgKey)
	if err != nil {
		return nil, nil, err
	}

	key := &GPGKey{}
	resp, err := s.client.Do(ctx, req, key)
	if err != nil {
		return nil, resp, err
	}

	return key, resp, nil
}

// DeleteGPGKey deletes a GPG key. It requires authentication via Basic Auth or
// via OAuth with at least admin:gpg_key scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/users/#delete-a-gpg-key-for-the-authenticated-user
func (s *UsersService) DeleteGPGKey(ctx context.Context, id int64) (*Response, error) {
	u := fmt.Sprintf("user/gpg_keys/%v", id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
