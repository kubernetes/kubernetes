// Copyright 2015 The etcd Authors
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
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"path"
)

var defaultV2AuthPrefix = "/v2/auth"

type User struct {
	User     string   `json:"user"`
	Password string   `json:"password,omitempty"`
	Roles    []string `json:"roles"`
	Grant    []string `json:"grant,omitempty"`
	Revoke   []string `json:"revoke,omitempty"`
}

// userListEntry is the user representation given by the server for ListUsers
type userListEntry struct {
	User  string `json:"user"`
	Roles []Role `json:"roles"`
}

type UserRoles struct {
	User  string `json:"user"`
	Roles []Role `json:"roles"`
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
	return s.enableDisable(ctx, &authAPIAction{http.MethodPut})
}

func (s *httpAuthAPI) Disable(ctx context.Context) error {
	return s.enableDisable(ctx, &authAPIAction{http.MethodDelete})
}

func (s *httpAuthAPI) enableDisable(ctx context.Context, req httpAction) error {
	resp, body, err := s.client.Do(ctx, req)
	if err != nil {
		return err
	}
	if err = assertStatusCode(resp.StatusCode, http.StatusOK, http.StatusCreated); err != nil {
		var sec authError
		err = json.Unmarshal(body, &sec)
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
	// AddUser adds a user.
	AddUser(ctx context.Context, username string, password string) error

	// RemoveUser removes a user.
	RemoveUser(ctx context.Context, username string) error

	// GetUser retrieves user details.
	GetUser(ctx context.Context, username string) (*User, error)

	// GrantUser grants a user some permission roles.
	GrantUser(ctx context.Context, username string, roles []string) (*User, error)

	// RevokeUser revokes some permission roles from a user.
	RevokeUser(ctx context.Context, username string, roles []string) (*User, error)

	// ChangePassword changes the user's password.
	ChangePassword(ctx context.Context, username string, password string) (*User, error)

	// ListUsers lists the users.
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
	req, _ := http.NewRequest(http.MethodGet, u.String(), nil)
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
	if err = assertStatusCode(resp.StatusCode, http.StatusOK); err != nil {
		var sec authError
		err = json.Unmarshal(body, &sec)
		if err != nil {
			return nil, err
		}
		return nil, sec
	}

	var userList struct {
		Users []userListEntry `json:"users"`
	}

	if err = json.Unmarshal(body, &userList); err != nil {
		return nil, err
	}

	ret := make([]string, 0, len(userList.Users))
	for _, u := range userList.Users {
		ret = append(ret, u.User)
	}
	return ret, nil
}

func (u *httpAuthUserAPI) AddUser(ctx context.Context, username string, password string) error {
	user := &User{
		User:     username,
		Password: password,
	}
	return u.addRemoveUser(ctx, &authUserAPIAction{
		verb:     http.MethodPut,
		username: username,
		user:     user,
	})
}

func (u *httpAuthUserAPI) RemoveUser(ctx context.Context, username string) error {
	return u.addRemoveUser(ctx, &authUserAPIAction{
		verb:     http.MethodDelete,
		username: username,
	})
}

func (u *httpAuthUserAPI) addRemoveUser(ctx context.Context, req *authUserAPIAction) error {
	resp, body, err := u.client.Do(ctx, req)
	if err != nil {
		return err
	}
	if err = assertStatusCode(resp.StatusCode, http.StatusOK, http.StatusCreated); err != nil {
		var sec authError
		err = json.Unmarshal(body, &sec)
		if err != nil {
			return err
		}
		return sec
	}
	return nil
}

func (u *httpAuthUserAPI) GetUser(ctx context.Context, username string) (*User, error) {
	return u.modUser(ctx, &authUserAPIAction{
		verb:     http.MethodGet,
		username: username,
	})
}

func (u *httpAuthUserAPI) GrantUser(ctx context.Context, username string, roles []string) (*User, error) {
	user := &User{
		User:  username,
		Grant: roles,
	}
	return u.modUser(ctx, &authUserAPIAction{
		verb:     http.MethodPut,
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
		verb:     http.MethodPut,
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
		verb:     http.MethodPut,
		username: username,
		user:     user,
	})
}

func (u *httpAuthUserAPI) modUser(ctx context.Context, req *authUserAPIAction) (*User, error) {
	resp, body, err := u.client.Do(ctx, req)
	if err != nil {
		return nil, err
	}
	if err = assertStatusCode(resp.StatusCode, http.StatusOK); err != nil {
		var sec authError
		err = json.Unmarshal(body, &sec)
		if err != nil {
			return nil, err
		}
		return nil, sec
	}
	var user User
	if err = json.Unmarshal(body, &user); err != nil {
		var userR UserRoles
		if urerr := json.Unmarshal(body, &userR); urerr != nil {
			return nil, err
		}
		user.User = userR.User
		for _, r := range userR.Roles {
			user.Roles = append(user.Roles, r.Role)
		}
	}
	return &user, nil
}
