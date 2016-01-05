// Copyright 2015 CoreOS, Inc.
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

package client

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/url"
	"path"

	"github.com/coreos/etcd/Godeps/_workspace/src/golang.org/x/net/context"
)

var (
	defaultV2AuthPrefix = "/v2/auth"
)

type User struct {
	User     string   `json:"user"`
	Password string   `json:"password,omitempty"`
	Roles    []string `json:"roles"`
	Grant    []string `json:"grant,omitempty"`
	Revoke   []string `json:"revoke,omitempty"`
}

func v2AuthURL(ep url.URL, action string, name string) *url.URL {
	if name != "" {
		ep.Path = path.Join(ep.Path, defaultV2AuthPrefix, action, name)
		return &ep
	}
	ep.Path = path.Join(ep.Path, defaultV2AuthPrefix, action)
	return &ep
}

// NewAuthAPI constructs a new AuthAPI that uses HTTP to
// interact with etcd's general auth features.
func NewAuthAPI(c Client) AuthAPI {
	return &httpAuthAPI{
		client: c,
	}
}

type AuthAPI interface {
	// Enable auth.
	Enable(ctx context.Context) error

	// Disable auth.
	Disable(ctx context.Context) error
}

type httpAuthAPI struct {
	client httpClient
}

func (s *httpAuthAPI) Enable(ctx context.Context) error {
	return s.enableDisable(ctx, &authAPIAction{"PUT"})
}

func (s *httpAuthAPI) Disable(ctx context.Context) error {
	return s.enableDisable(ctx, &authAPIAction{"DELETE"})
}

func (s *httpAuthAPI) enableDisable(ctx context.Context, req httpAction) error {
	resp, body, err := s.client.Do(ctx, req)
	if err != nil {
		return err
	}
	if err := assertStatusCode(resp.StatusCode, http.StatusOK, http.StatusCreated); err != nil {
		var sec authError
		err := json.Unmarshal(body, &sec)
		if err != nil {
			return err
		}
		return sec
	}
	return nil
}

type authAPIAction struct {
	verb string
}

func (l *authAPIAction) HTTPRequest(ep url.URL) *http.Request {
	u := v2AuthURL(ep, "enable", "")
	req, _ := http.NewRequest(l.verb, u.String(), nil)
	return req
}

type authError struct {
	Message string `json:"message"`
	Code    int    `json:"-"`
}

func (e authError) Error() string {
	return e.Message
}

// NewAuthUserAPI constructs a new AuthUserAPI that uses HTTP to
// interact with etcd's user creation and modification features.
func NewAuthUserAPI(c Client) AuthUserAPI {
	return &httpAuthUserAPI{
		client: c,
	}
}

type AuthUserAPI interface {
	// Add a user.
	AddUser(ctx context.Context, username string, password string) error

	// Remove a user.
	RemoveUser(ctx context.Context, username string) error

	// Get user details.
	GetUser(ctx context.Context, username string) (*User, error)

	// Grant a user some permission roles.
	GrantUser(ctx context.Context, username string, roles []string) (*User, error)

	// Revoke some permission roles from a user.
	RevokeUser(ctx context.Context, username string, roles []string) (*User, error)

	// Change the user's password.
	ChangePassword(ctx context.Context, username string, password string) (*User, error)

	// List users.
	ListUsers(ctx context.Context) ([]string, error)
}

type httpAuthUserAPI struct {
	client httpClient
}

type authUserAPIAction struct {
	verb     string
	username string
	user     *User
}

type authUserAPIList struct{}

func (list *authUserAPIList) HTTPRequest(ep url.URL) *http.Request {
	u := v2AuthURL(ep, "users", "")
	req, _ := http.NewRequest("GET", u.String(), nil)
	req.Header.Set("Content-Type", "application/json")
	return req
}

func (l *authUserAPIAction) HTTPRequest(ep url.URL) *http.Request {
	u := v2AuthURL(ep, "users", l.username)
	if l.user == nil {
		req, _ := http.NewRequest(l.verb, u.String(), nil)
		return req
	}
	b, err := json.Marshal(l.user)
	if err != nil {
		panic(err)
	}
	body := bytes.NewReader(b)
	req, _ := http.NewRequest(l.verb, u.String(), body)
	req.Header.Set("Content-Type", "application/json")
	return req
}

func (u *httpAuthUserAPI) ListUsers(ctx context.Context) ([]string, error) {
	resp, body, err := u.client.Do(ctx, &authUserAPIList{})
	if err != nil {
		return nil, err
	}
	if err := assertStatusCode(resp.StatusCode, http.StatusOK); err != nil {
		var sec authError
		err := json.Unmarshal(body, &sec)
		if err != nil {
			return nil, err
		}
		return nil, sec
	}
	var userList struct {
		Users []string `json:"users"`
	}
	err = json.Unmarshal(body, &userList)
	if err != nil {
		return nil, err
	}
	return userList.Users, nil
}

func (u *httpAuthUserAPI) AddUser(ctx context.Context, username string, password string) error {
	user := &User{
		User:     username,
		Password: password,
	}
	return u.addRemoveUser(ctx, &authUserAPIAction{
		verb:     "PUT",
		username: username,
		user:     user,
	})
}

func (u *httpAuthUserAPI) RemoveUser(ctx context.Context, username string) error {
	return u.addRemoveUser(ctx, &authUserAPIAction{
		verb:     "DELETE",
		username: username,
	})
}

func (u *httpAuthUserAPI) addRemoveUser(ctx context.Context, req *authUserAPIAction) error {
	resp, body, err := u.client.Do(ctx, req)
	if err != nil {
		return err
	}
	if err := assertStatusCode(resp.StatusCode, http.StatusOK, http.StatusCreated); err != nil {
		var sec authError
		err := json.Unmarshal(body, &sec)
		if err != nil {
			return err
		}
		return sec
	}
	return nil
}

func (u *httpAuthUserAPI) GetUser(ctx context.Context, username string) (*User, error) {
	return u.modUser(ctx, &authUserAPIAction{
		verb:     "GET",
		username: username,
	})
}

func (u *httpAuthUserAPI) GrantUser(ctx context.Context, username string, roles []string) (*User, error) {
	user := &User{
		User:  username,
		Grant: roles,
	}
	return u.modUser(ctx, &authUserAPIAction{
		verb:     "PUT",
		username: username,
		user:     user,
	})
}

func (u *httpAuthUserAPI) RevokeUser(ctx context.Context, username string, roles []string) (*User, error) {
	user := &User{
		User:   username,
		Revoke: roles,
	}
	return u.modUser(ctx, &authUserAPIAction{
		verb:     "PUT",
		username: username,
		user:     user,
	})
}

func (u *httpAuthUserAPI) ChangePassword(ctx context.Context, username string, password string) (*User, error) {
	user := &User{
		User:     username,
		Password: password,
	}
	return u.modUser(ctx, &authUserAPIAction{
		verb:     "PUT",
		username: username,
		user:     user,
	})
}

func (u *httpAuthUserAPI) modUser(ctx context.Context, req *authUserAPIAction) (*User, error) {
	resp, body, err := u.client.Do(ctx, req)
	if err != nil {
		return nil, err
	}
	if err := assertStatusCode(resp.StatusCode, http.StatusOK); err != nil {
		var sec authError
		err := json.Unmarshal(body, &sec)
		if err != nil {
			return nil, err
		}
		return nil, sec
	}
	var user User
	err = json.Unmarshal(body, &user)
	if err != nil {
		return nil, err
	}
	return &user, nil
}
