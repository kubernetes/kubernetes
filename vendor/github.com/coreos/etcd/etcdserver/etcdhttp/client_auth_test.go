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

package etcdhttp

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"path"
	"strings"
	"testing"

	"github.com/coreos/etcd/etcdserver/auth"
)

const goodPassword = "good"

func mustJSONRequest(t *testing.T, method string, p string, body string) *http.Request {
	req, err := http.NewRequest(method, path.Join(authPrefix, p), strings.NewReader(body))
	if err != nil {
		t.Fatalf("Error making JSON request: %s %s %s\n", method, p, body)
	}
	req.Header.Set("Content-Type", "application/json")
	return req
}

type mockAuthStore struct {
	users   map[string]*auth.User
	roles   map[string]*auth.Role
	err     error
	enabled bool
}

func (s *mockAuthStore) AllUsers() ([]string, error) { return []string{"alice", "bob", "root"}, s.err }
func (s *mockAuthStore) GetUser(name string) (auth.User, error) {
	u, ok := s.users[name]
	if !ok {
		return auth.User{}, s.err
	}
	return *u, s.err
}
func (s *mockAuthStore) CreateOrUpdateUser(user auth.User) (out auth.User, created bool, err error) {
	if s.users == nil {
		u, err := s.CreateUser(user)
		return u, true, err
	}
	u, err := s.UpdateUser(user)
	return u, false, err
}
func (s *mockAuthStore) CreateUser(user auth.User) (auth.User, error) { return user, s.err }
func (s *mockAuthStore) DeleteUser(name string) error                 { return s.err }
func (s *mockAuthStore) UpdateUser(user auth.User) (auth.User, error) {
	return *s.users[user.User], s.err
}
func (s *mockAuthStore) AllRoles() ([]string, error) {
	return []string{"awesome", "guest", "root"}, s.err
}
func (s *mockAuthStore) GetRole(name string) (auth.Role, error) { return *s.roles[name], s.err }
func (s *mockAuthStore) CreateRole(role auth.Role) error        { return s.err }
func (s *mockAuthStore) DeleteRole(name string) error           { return s.err }
func (s *mockAuthStore) UpdateRole(role auth.Role) (auth.Role, error) {
	return *s.roles[role.Role], s.err
}
func (s *mockAuthStore) AuthEnabled() bool  { return s.enabled }
func (s *mockAuthStore) EnableAuth() error  { return s.err }
func (s *mockAuthStore) DisableAuth() error { return s.err }

func (s *mockAuthStore) CheckPassword(user auth.User, password string) bool {
	return user.Password == password
}

func (s *mockAuthStore) HashPassword(password string) (string, error) {
	return password, nil
}

func TestAuthFlow(t *testing.T) {
	enableMapMu.Lock()
	enabledMap = make(map[capability]bool)
	enabledMap[authCapability] = true
	enableMapMu.Unlock()
	var testCases = []struct {
		req   *http.Request
		store mockAuthStore

		wcode int
		wbody string
	}{
		{
			req:   mustJSONRequest(t, "PUT", "users/alice", `{{{{{{{`),
			store: mockAuthStore{},
			wcode: http.StatusBadRequest,
			wbody: `{"message":"Invalid JSON in request body."}`,
		},
		{
			req:   mustJSONRequest(t, "PUT", "users/alice", `{"user": "alice", "password": "goodpassword"}`),
			store: mockAuthStore{enabled: true},
			wcode: http.StatusUnauthorized,
			wbody: `{"message":"Insufficient credentials"}`,
		},
		// Users
		{
			req: mustJSONRequest(t, "GET", "users", ""),
			store: mockAuthStore{
				users: map[string]*auth.User{
					"alice": {
						User:     "alice",
						Roles:    []string{"alicerole", "guest"},
						Password: "wheeee",
					},
					"bob": {
						User:     "bob",
						Roles:    []string{"guest"},
						Password: "wheeee",
					},
					"root": {
						User:     "root",
						Roles:    []string{"root"},
						Password: "wheeee",
					},
				},
				roles: map[string]*auth.Role{
					"alicerole": {
						Role: "alicerole",
					},
					"guest": {
						Role: "guest",
					},
					"root": {
						Role: "root",
					},
				},
			},
			wcode: http.StatusOK,
			wbody: `{"users":[` +
				`{"user":"alice","roles":[` +
				`{"role":"alicerole","permissions":{"kv":{"read":null,"write":null}}},` +
				`{"role":"guest","permissions":{"kv":{"read":null,"write":null}}}` +
				`]},` +
				`{"user":"bob","roles":[{"role":"guest","permissions":{"kv":{"read":null,"write":null}}}]},` +
				`{"user":"root","roles":[{"role":"root","permissions":{"kv":{"read":null,"write":null}}}]}]}`,
		},
		{
			req: mustJSONRequest(t, "GET", "users/alice", ""),
			store: mockAuthStore{
				users: map[string]*auth.User{
					"alice": {
						User:     "alice",
						Roles:    []string{"alicerole"},
						Password: "wheeee",
					},
				},
				roles: map[string]*auth.Role{
					"alicerole": {
						Role: "alicerole",
					},
				},
			},
			wcode: http.StatusOK,
			wbody: `{"user":"alice","roles":[{"role":"alicerole","permissions":{"kv":{"read":null,"write":null}}}]}`,
		},
		{
			req:   mustJSONRequest(t, "PUT", "users/alice", `{"user": "alice", "password": "goodpassword"}`),
			store: mockAuthStore{},
			wcode: http.StatusCreated,
			wbody: `{"user":"alice","roles":null}`,
		},
		{
			req:   mustJSONRequest(t, "DELETE", "users/alice", ``),
			store: mockAuthStore{},
			wcode: http.StatusOK,
			wbody: ``,
		},
		{
			req: mustJSONRequest(t, "PUT", "users/alice", `{"user": "alice", "password": "goodpassword"}`),
			store: mockAuthStore{
				users: map[string]*auth.User{
					"alice": {
						User:     "alice",
						Roles:    []string{"alicerole", "guest"},
						Password: "wheeee",
					},
				},
			},
			wcode: http.StatusOK,
			wbody: `{"user":"alice","roles":["alicerole","guest"]}`,
		},
		{
			req: mustJSONRequest(t, "PUT", "users/alice", `{"user": "alice", "grant": ["alicerole"]}`),
			store: mockAuthStore{
				users: map[string]*auth.User{
					"alice": {
						User:     "alice",
						Roles:    []string{"alicerole", "guest"},
						Password: "wheeee",
					},
				},
			},
			wcode: http.StatusOK,
			wbody: `{"user":"alice","roles":["alicerole","guest"]}`,
		},
		{
			req: mustJSONRequest(t, "GET", "users/alice", ``),
			store: mockAuthStore{
				users: map[string]*auth.User{},
				err:   auth.Error{Status: http.StatusNotFound, Errmsg: "auth: User alice doesn't exist."},
			},
			wcode: http.StatusNotFound,
			wbody: `{"message":"auth: User alice doesn't exist."}`,
		},
		{
			req: mustJSONRequest(t, "GET", "roles/manager", ""),
			store: mockAuthStore{
				roles: map[string]*auth.Role{
					"manager": {
						Role: "manager",
					},
				},
			},
			wcode: http.StatusOK,
			wbody: `{"role":"manager","permissions":{"kv":{"read":null,"write":null}}}`,
		},
		{
			req:   mustJSONRequest(t, "DELETE", "roles/manager", ``),
			store: mockAuthStore{},
			wcode: http.StatusOK,
			wbody: ``,
		},
		{
			req:   mustJSONRequest(t, "PUT", "roles/manager", `{"role":"manager","permissions":{"kv":{"read":[],"write":[]}}}`),
			store: mockAuthStore{},
			wcode: http.StatusCreated,
			wbody: `{"role":"manager","permissions":{"kv":{"read":[],"write":[]}}}`,
		},
		{
			req: mustJSONRequest(t, "PUT", "roles/manager", `{"role":"manager","revoke":{"kv":{"read":["foo"],"write":[]}}}`),
			store: mockAuthStore{
				roles: map[string]*auth.Role{
					"manager": {
						Role: "manager",
					},
				},
			},
			wcode: http.StatusOK,
			wbody: `{"role":"manager","permissions":{"kv":{"read":null,"write":null}}}`,
		},
		{
			req: mustJSONRequest(t, "GET", "roles", ""),
			store: mockAuthStore{
				roles: map[string]*auth.Role{
					"awesome": {
						Role: "awesome",
					},
					"guest": {
						Role: "guest",
					},
					"root": {
						Role: "root",
					},
				},
			},
			wcode: http.StatusOK,
			wbody: `{"roles":[{"role":"awesome","permissions":{"kv":{"read":null,"write":null}}},` +
				`{"role":"guest","permissions":{"kv":{"read":null,"write":null}}},` +
				`{"role":"root","permissions":{"kv":{"read":null,"write":null}}}]}`,
		},
		{
			req: mustJSONRequest(t, "GET", "enable", ""),
			store: mockAuthStore{
				enabled: true,
			},
			wcode: http.StatusOK,
			wbody: `{"enabled":true}`,
		},
		{
			req: mustJSONRequest(t, "PUT", "enable", ""),
			store: mockAuthStore{
				enabled: false,
			},
			wcode: http.StatusOK,
			wbody: ``,
		},
		{
			req: (func() *http.Request {
				req := mustJSONRequest(t, "DELETE", "enable", "")
				req.SetBasicAuth("root", "good")
				return req
			})(),
			store: mockAuthStore{
				enabled: true,
				users: map[string]*auth.User{
					"root": {
						User:     "root",
						Password: goodPassword,
						Roles:    []string{"root"},
					},
				},
				roles: map[string]*auth.Role{
					"root": {
						Role: "root",
					},
				},
			},
			wcode: http.StatusOK,
			wbody: ``,
		},
		{
			req: (func() *http.Request {
				req := mustJSONRequest(t, "DELETE", "enable", "")
				req.SetBasicAuth("root", "bad")
				return req
			})(),
			store: mockAuthStore{
				enabled: true,
				users: map[string]*auth.User{
					"root": {
						User:     "root",
						Password: goodPassword,
						Roles:    []string{"root"},
					},
				},
				roles: map[string]*auth.Role{
					"root": {
						Role: "guest",
					},
				},
			},
			wcode: http.StatusUnauthorized,
			wbody: `{"message":"Insufficient credentials"}`,
		},
	}

	for i, tt := range testCases {
		mux := http.NewServeMux()
		h := &authHandler{
			sec:     &tt.store,
			cluster: &fakeCluster{id: 1},
		}
		handleAuth(mux, h)
		rw := httptest.NewRecorder()
		mux.ServeHTTP(rw, tt.req)
		if rw.Code != tt.wcode {
			t.Errorf("#%d: got code=%d, want %d", i, rw.Code, tt.wcode)
		}
		g := rw.Body.String()
		g = strings.TrimSpace(g)
		if g != tt.wbody {
			t.Errorf("#%d: got body=%s, want %s", i, g, tt.wbody)
		}
	}
}

func mustAuthRequest(method, username, password string) *http.Request {
	req, err := http.NewRequest(method, "path", strings.NewReader(""))
	if err != nil {
		panic("Cannot make auth request: " + err.Error())
	}
	req.SetBasicAuth(username, password)
	return req
}

func TestPrefixAccess(t *testing.T) {
	var table = []struct {
		key                string
		req                *http.Request
		store              *mockAuthStore
		hasRoot            bool
		hasKeyPrefixAccess bool
		hasRecursiveAccess bool
	}{
		{
			key: "/foo",
			req: mustAuthRequest("GET", "root", "good"),
			store: &mockAuthStore{
				users: map[string]*auth.User{
					"root": {
						User:     "root",
						Password: goodPassword,
						Roles:    []string{"root"},
					},
				},
				roles: map[string]*auth.Role{
					"root": {
						Role: "root",
					},
				},
				enabled: true,
			},
			hasRoot:            true,
			hasKeyPrefixAccess: true,
			hasRecursiveAccess: true,
		},
		{
			key: "/foo",
			req: mustAuthRequest("GET", "user", "good"),
			store: &mockAuthStore{
				users: map[string]*auth.User{
					"user": {
						User:     "user",
						Password: goodPassword,
						Roles:    []string{"foorole"},
					},
				},
				roles: map[string]*auth.Role{
					"foorole": {
						Role: "foorole",
						Permissions: auth.Permissions{
							KV: auth.RWPermission{
								Read:  []string{"/foo"},
								Write: []string{"/foo"},
							},
						},
					},
				},
				enabled: true,
			},
			hasRoot:            false,
			hasKeyPrefixAccess: true,
			hasRecursiveAccess: false,
		},
		{
			key: "/foo",
			req: mustAuthRequest("GET", "user", "good"),
			store: &mockAuthStore{
				users: map[string]*auth.User{
					"user": {
						User:     "user",
						Password: goodPassword,
						Roles:    []string{"foorole"},
					},
				},
				roles: map[string]*auth.Role{
					"foorole": {
						Role: "foorole",
						Permissions: auth.Permissions{
							KV: auth.RWPermission{
								Read:  []string{"/foo*"},
								Write: []string{"/foo*"},
							},
						},
					},
				},
				enabled: true,
			},
			hasRoot:            false,
			hasKeyPrefixAccess: true,
			hasRecursiveAccess: true,
		},
		{
			key: "/foo",
			req: mustAuthRequest("GET", "user", "bad"),
			store: &mockAuthStore{
				users: map[string]*auth.User{
					"user": {
						User:     "user",
						Password: goodPassword,
						Roles:    []string{"foorole"},
					},
				},
				roles: map[string]*auth.Role{
					"foorole": {
						Role: "foorole",
						Permissions: auth.Permissions{
							KV: auth.RWPermission{
								Read:  []string{"/foo*"},
								Write: []string{"/foo*"},
							},
						},
					},
				},
				enabled: true,
			},
			hasRoot:            false,
			hasKeyPrefixAccess: false,
			hasRecursiveAccess: false,
		},
		{
			key: "/foo",
			req: mustAuthRequest("GET", "user", "good"),
			store: &mockAuthStore{
				users:   map[string]*auth.User{},
				err:     errors.New("Not the user"),
				enabled: true,
			},
			hasRoot:            false,
			hasKeyPrefixAccess: false,
			hasRecursiveAccess: false,
		},
		{
			key: "/foo",
			req: mustJSONRequest(t, "GET", "somepath", ""),
			store: &mockAuthStore{
				users: map[string]*auth.User{
					"user": {
						User:     "user",
						Password: goodPassword,
						Roles:    []string{"foorole"},
					},
				},
				roles: map[string]*auth.Role{
					"guest": {
						Role: "guest",
						Permissions: auth.Permissions{
							KV: auth.RWPermission{
								Read:  []string{"/foo*"},
								Write: []string{"/foo*"},
							},
						},
					},
				},
				enabled: true,
			},
			hasRoot:            false,
			hasKeyPrefixAccess: true,
			hasRecursiveAccess: true,
		},
		{
			key: "/bar",
			req: mustJSONRequest(t, "GET", "somepath", ""),
			store: &mockAuthStore{
				users: map[string]*auth.User{
					"user": {
						User:     "user",
						Password: goodPassword,
						Roles:    []string{"foorole"},
					},
				},
				roles: map[string]*auth.Role{
					"guest": {
						Role: "guest",
						Permissions: auth.Permissions{
							KV: auth.RWPermission{
								Read:  []string{"/foo*"},
								Write: []string{"/foo*"},
							},
						},
					},
				},
				enabled: true,
			},
			hasRoot:            false,
			hasKeyPrefixAccess: false,
			hasRecursiveAccess: false,
		},
		// check access for multiple roles
		{
			key: "/foo",
			req: mustAuthRequest("GET", "user", "good"),
			store: &mockAuthStore{
				users: map[string]*auth.User{
					"user": {
						User:     "user",
						Password: goodPassword,
						Roles:    []string{"role1", "role2"},
					},
				},
				roles: map[string]*auth.Role{
					"role1": {
						Role: "role1",
					},
					"role2": {
						Role: "role2",
						Permissions: auth.Permissions{
							KV: auth.RWPermission{
								Read:  []string{"/foo"},
								Write: []string{"/foo"},
							},
						},
					},
				},
				enabled: true,
			},
			hasRoot:            false,
			hasKeyPrefixAccess: true,
			hasRecursiveAccess: false,
		},
		{
			key: "/foo",
			req: (func() *http.Request {
				req := mustJSONRequest(t, "GET", "somepath", "")
				req.Header.Set("Authorization", "malformedencoding")
				return req
			})(),
			store: &mockAuthStore{
				enabled: true,
				users: map[string]*auth.User{
					"root": {
						User:     "root",
						Password: goodPassword,
						Roles:    []string{"root"},
					},
				},
				roles: map[string]*auth.Role{
					"guest": {
						Role: "guest",
						Permissions: auth.Permissions{
							KV: auth.RWPermission{
								Read:  []string{"/foo*"},
								Write: []string{"/foo*"},
							},
						},
					},
				},
			},
			hasRoot:            false,
			hasKeyPrefixAccess: false,
			hasRecursiveAccess: false,
		},
	}

	for i, tt := range table {
		if tt.hasRoot != hasRootAccess(tt.store, tt.req) {
			t.Errorf("#%d: hasRoot doesn't match (expected %v)", i, tt.hasRoot)
		}
		if tt.hasKeyPrefixAccess != hasKeyPrefixAccess(tt.store, tt.req, tt.key, false) {
			t.Errorf("#%d: hasKeyPrefixAccess doesn't match (expected %v)", i, tt.hasRoot)
		}
		if tt.hasRecursiveAccess != hasKeyPrefixAccess(tt.store, tt.req, tt.key, true) {
			t.Errorf("#%d: hasRecursiveAccess doesn't match (expected %v)", i, tt.hasRoot)
		}
	}
}
