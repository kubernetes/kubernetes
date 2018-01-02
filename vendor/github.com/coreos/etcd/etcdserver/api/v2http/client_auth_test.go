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

package v2http

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path"
	"sort"
	"strings"
	"testing"

	"github.com/coreos/etcd/etcdserver/api"
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

func (s *mockAuthStore) AllUsers() ([]string, error) {
	var us []string
	for u := range s.users {
		us = append(us, u)
	}
	sort.Strings(us)
	return us, s.err
}
func (s *mockAuthStore) GetUser(name string) (auth.User, error) {
	u, ok := s.users[name]
	if !ok {
		return auth.User{}, s.err
	}
	return *u, s.err
}
func (s *mockAuthStore) CreateOrUpdateUser(user auth.User) (out auth.User, created bool, err error) {
	if s.users == nil {
		out, err = s.CreateUser(user)
		return out, true, err
	}
	out, err = s.UpdateUser(user)
	return out, false, err
}
func (s *mockAuthStore) CreateUser(user auth.User) (auth.User, error) { return user, s.err }
func (s *mockAuthStore) DeleteUser(name string) error                 { return s.err }
func (s *mockAuthStore) UpdateUser(user auth.User) (auth.User, error) {
	return *s.users[user.User], s.err
}
func (s *mockAuthStore) AllRoles() ([]string, error) {
	return []string{"awesome", "guest", "root"}, s.err
}
func (s *mockAuthStore) GetRole(name string) (auth.Role, error) {
	r, ok := s.roles[name]
	if ok {
		return *r, s.err
	}
	return auth.Role{}, fmt.Errorf("%q does not exist (%v)", name, s.err)
}
func (s *mockAuthStore) CreateRole(role auth.Role) error { return s.err }
func (s *mockAuthStore) DeleteRole(name string) error    { return s.err }
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
	api.EnableCapability(api.AuthCapability)
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

func TestGetUserGrantedWithNonexistingRole(t *testing.T) {
	sh := &authHandler{
		sec: &mockAuthStore{
			users: map[string]*auth.User{
				"root": {
					User:  "root",
					Roles: []string{"root", "foo"},
				},
			},
			roles: map[string]*auth.Role{
				"root": {
					Role: "root",
				},
			},
		},
		cluster: &fakeCluster{id: 1},
	}
	srv := httptest.NewServer(http.HandlerFunc(sh.baseUsers))
	defer srv.Close()

	req, err := http.NewRequest("GET", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.URL, err = url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	cli := http.DefaultClient
	resp, err := cli.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var uc usersCollections
	if err := json.NewDecoder(resp.Body).Decode(&uc); err != nil {
		t.Fatal(err)
	}
	if len(uc.Users) != 1 {
		t.Fatalf("expected 1 user, got %+v", uc.Users)
	}
	if uc.Users[0].User != "root" {
		t.Fatalf("expected 'root', got %q", uc.Users[0].User)
	}
	if len(uc.Users[0].Roles) != 1 {
		t.Fatalf("expected 1 role, got %+v", uc.Users[0].Roles)
	}
	if uc.Users[0].Roles[0].Role != "root" {
		t.Fatalf("expected 'root', got %q", uc.Users[0].Roles[0].Role)
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

func unauthedRequest(method string) *http.Request {
	req, err := http.NewRequest(method, "path", strings.NewReader(""))
	if err != nil {
		panic("Cannot make request: " + err.Error())
	}
	return req
}

func tlsAuthedRequest(req *http.Request, certname string) *http.Request {
	bytes, err := ioutil.ReadFile(fmt.Sprintf("testdata/%s.pem", certname))
	if err != nil {
		panic(err)
	}

	block, _ := pem.Decode(bytes)
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		panic(err)
	}

	req.TLS = &tls.ConnectionState{
		VerifiedChains: [][]*x509.Certificate{{cert}},
	}
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
		{ // guest access in non-TLS mode
			key: "/foo",
			req: (func() *http.Request {
				return mustJSONRequest(t, "GET", "somepath", "")
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
			hasKeyPrefixAccess: true,
			hasRecursiveAccess: true,
		},
	}

	for i, tt := range table {
		if tt.hasRoot != hasRootAccess(tt.store, tt.req, true) {
			t.Errorf("#%d: hasRoot doesn't match (expected %v)", i, tt.hasRoot)
		}
		if tt.hasKeyPrefixAccess != hasKeyPrefixAccess(tt.store, tt.req, tt.key, false, true) {
			t.Errorf("#%d: hasKeyPrefixAccess doesn't match (expected %v)", i, tt.hasRoot)
		}
		if tt.hasRecursiveAccess != hasKeyPrefixAccess(tt.store, tt.req, tt.key, true, true) {
			t.Errorf("#%d: hasRecursiveAccess doesn't match (expected %v)", i, tt.hasRoot)
		}
	}
}

func TestUserFromClientCertificate(t *testing.T) {
	witherror := &mockAuthStore{
		users: map[string]*auth.User{
			"user": {
				User:     "user",
				Roles:    []string{"root"},
				Password: "password",
			},
			"basicauth": {
				User:     "basicauth",
				Roles:    []string{"root"},
				Password: "password",
			},
		},
		roles: map[string]*auth.Role{
			"root": {
				Role: "root",
			},
		},
		err: errors.New(""),
	}

	noerror := &mockAuthStore{
		users: map[string]*auth.User{
			"user": {
				User:     "user",
				Roles:    []string{"root"},
				Password: "password",
			},
			"basicauth": {
				User:     "basicauth",
				Roles:    []string{"root"},
				Password: "password",
			},
		},
		roles: map[string]*auth.Role{
			"root": {
				Role: "root",
			},
		},
	}

	var table = []struct {
		req        *http.Request
		userExists bool
		store      auth.Store
		username   string
	}{
		{
			// non tls request
			req:        unauthedRequest("GET"),
			userExists: false,
			store:      witherror,
		},
		{
			// cert with cn of existing user
			req:        tlsAuthedRequest(unauthedRequest("GET"), "user"),
			userExists: true,
			username:   "user",
			store:      noerror,
		},
		{
			// cert with cn of non-existing user
			req:        tlsAuthedRequest(unauthedRequest("GET"), "otheruser"),
			userExists: false,
			store:      witherror,
		},
	}

	for i, tt := range table {
		user := userFromClientCertificate(tt.store, tt.req)
		userExists := user != nil

		if tt.userExists != userExists {
			t.Errorf("#%d: userFromClientCertificate doesn't match (expected %v)", i, tt.userExists)
		}
		if user != nil && (tt.username != user.User) {
			t.Errorf("#%d: userFromClientCertificate username doesn't match (expected %s, got %s)", i, tt.username, user.User)
		}
	}
}

func TestUserFromBasicAuth(t *testing.T) {
	sec := &mockAuthStore{
		users: map[string]*auth.User{
			"user": {
				User:     "user",
				Roles:    []string{"root"},
				Password: "password",
			},
		},
		roles: map[string]*auth.Role{
			"root": {
				Role: "root",
			},
		},
	}

	var table = []struct {
		username   string
		req        *http.Request
		userExists bool
	}{
		{
			// valid user, valid pass
			username:   "user",
			req:        mustAuthRequest("GET", "user", "password"),
			userExists: true,
		},
		{
			// valid user, bad pass
			username:   "user",
			req:        mustAuthRequest("GET", "user", "badpass"),
			userExists: false,
		},
		{
			// valid user, no pass
			username:   "user",
			req:        mustAuthRequest("GET", "user", ""),
			userExists: false,
		},
		{
			// missing user
			username:   "missing",
			req:        mustAuthRequest("GET", "missing", "badpass"),
			userExists: false,
		},
		{
			// no basic auth
			req:        unauthedRequest("GET"),
			userExists: false,
		},
	}

	for i, tt := range table {
		user := userFromBasicAuth(sec, tt.req)
		userExists := user != nil

		if tt.userExists != userExists {
			t.Errorf("#%d: userFromBasicAuth doesn't match (expected %v)", i, tt.userExists)
		}
		if user != nil && (tt.username != user.User) {
			t.Errorf("#%d: userFromBasicAuth username doesn't match (expected %s, got %s)", i, tt.username, user.User)
		}
	}
}
