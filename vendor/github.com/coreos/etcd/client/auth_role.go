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

	"github.com/coreos/etcd/Godeps/_workspace/src/golang.org/x/net/context"
)

type Role struct {
	Role        string       `json:"role"`
	Permissions Permissions  `json:"permissions"`
	Grant       *Permissions `json:"grant,omitempty"`
	Revoke      *Permissions `json:"revoke,omitempty"`
}

type Permissions struct {
	KV rwPermission `json:"kv"`
}

type rwPermission struct {
	Read  []string `json:"read"`
	Write []string `json:"write"`
}

type PermissionType int

const (
	ReadPermission PermissionType = iota
	WritePermission
	ReadWritePermission
)

// NewAuthRoleAPI constructs a new AuthRoleAPI that uses HTTP to
// interact with etcd's role creation and modification features.
func NewAuthRoleAPI(c Client) AuthRoleAPI {
	return &httpAuthRoleAPI{
		client: c,
	}
}

type AuthRoleAPI interface {
	// Add a role.
	AddRole(ctx context.Context, role string) error

	// Remove a role.
	RemoveRole(ctx context.Context, role string) error

	// Get role details.
	GetRole(ctx context.Context, role string) (*Role, error)

	// Grant a role some permission prefixes for the KV store.
	GrantRoleKV(ctx context.Context, role string, prefixes []string, permType PermissionType) (*Role, error)

	// Revoke some some permission prefixes for a role on the KV store.
	RevokeRoleKV(ctx context.Context, role string, prefixes []string, permType PermissionType) (*Role, error)

	// List roles.
	ListRoles(ctx context.Context) ([]string, error)
}

type httpAuthRoleAPI struct {
	client httpClient
}

type authRoleAPIAction struct {
	verb string
	name string
	role *Role
}

type authRoleAPIList struct{}

func (list *authRoleAPIList) HTTPRequest(ep url.URL) *http.Request {
	u := v2AuthURL(ep, "roles", "")
	req, _ := http.NewRequest("GET", u.String(), nil)
	req.Header.Set("Content-Type", "application/json")
	return req
}

func (l *authRoleAPIAction) HTTPRequest(ep url.URL) *http.Request {
	u := v2AuthURL(ep, "roles", l.name)
	if l.role == nil {
		req, _ := http.NewRequest(l.verb, u.String(), nil)
		return req
	}
	b, err := json.Marshal(l.role)
	if err != nil {
		panic(err)
	}
	body := bytes.NewReader(b)
	req, _ := http.NewRequest(l.verb, u.String(), body)
	req.Header.Set("Content-Type", "application/json")
	return req
}

func (r *httpAuthRoleAPI) ListRoles(ctx context.Context) ([]string, error) {
	resp, body, err := r.client.Do(ctx, &authRoleAPIList{})
	if err != nil {
		return nil, err
	}
	if err := assertStatusCode(resp.StatusCode, http.StatusOK); err != nil {
		return nil, err
	}
	var userList struct {
		Roles []string `json:"roles"`
	}
	err = json.Unmarshal(body, &userList)
	if err != nil {
		return nil, err
	}
	return userList.Roles, nil
}

func (r *httpAuthRoleAPI) AddRole(ctx context.Context, rolename string) error {
	role := &Role{
		Role: rolename,
	}
	return r.addRemoveRole(ctx, &authRoleAPIAction{
		verb: "PUT",
		name: rolename,
		role: role,
	})
}

func (r *httpAuthRoleAPI) RemoveRole(ctx context.Context, rolename string) error {
	return r.addRemoveRole(ctx, &authRoleAPIAction{
		verb: "DELETE",
		name: rolename,
	})
}

func (r *httpAuthRoleAPI) addRemoveRole(ctx context.Context, req *authRoleAPIAction) error {
	resp, body, err := r.client.Do(ctx, req)
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

func (r *httpAuthRoleAPI) GetRole(ctx context.Context, rolename string) (*Role, error) {
	return r.modRole(ctx, &authRoleAPIAction{
		verb: "GET",
		name: rolename,
	})
}

func buildRWPermission(prefixes []string, permType PermissionType) rwPermission {
	var out rwPermission
	switch permType {
	case ReadPermission:
		out.Read = prefixes
	case WritePermission:
		out.Write = prefixes
	case ReadWritePermission:
		out.Read = prefixes
		out.Write = prefixes
	}
	return out
}

func (r *httpAuthRoleAPI) GrantRoleKV(ctx context.Context, rolename string, prefixes []string, permType PermissionType) (*Role, error) {
	rwp := buildRWPermission(prefixes, permType)
	role := &Role{
		Role: rolename,
		Grant: &Permissions{
			KV: rwp,
		},
	}
	return r.modRole(ctx, &authRoleAPIAction{
		verb: "PUT",
		name: rolename,
		role: role,
	})
}

func (r *httpAuthRoleAPI) RevokeRoleKV(ctx context.Context, rolename string, prefixes []string, permType PermissionType) (*Role, error) {
	rwp := buildRWPermission(prefixes, permType)
	role := &Role{
		Role: rolename,
		Revoke: &Permissions{
			KV: rwp,
		},
	}
	return r.modRole(ctx, &authRoleAPIAction{
		verb: "PUT",
		name: rolename,
		role: role,
	})
}

func (r *httpAuthRoleAPI) modRole(ctx context.Context, req *authRoleAPIAction) (*Role, error) {
	resp, body, err := r.client.Do(ctx, req)
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
	var role Role
	err = json.Unmarshal(body, &role)
	if err != nil {
		return nil, err
	}
	return &role, nil
}
