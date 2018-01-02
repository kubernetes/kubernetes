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

package auth

import (
	"reflect"
	"testing"
	"time"

	etcderr "github.com/coreos/etcd/error"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/etcdserverpb"
	etcdstore "github.com/coreos/etcd/store"
	"golang.org/x/net/context"
)

type fakeDoer struct{}

func (_ fakeDoer) Do(context.Context, etcdserverpb.Request) (etcdserver.Response, error) {
	return etcdserver.Response{}, nil
}

func TestCheckPassword(t *testing.T) {
	st := NewStore(fakeDoer{}, 5*time.Second)
	u := User{Password: "$2a$10$I3iddh1D..EIOXXQtsra4u8AjOtgEa2ERxVvYGfXFBJDo1omXwP.q"}
	matched := st.CheckPassword(u, "foo")
	if matched {
		t.Fatalf("expected false, got %v", matched)
	}
}

const testTimeout = time.Millisecond

func TestMergeUser(t *testing.T) {
	tbl := []struct {
		input  User
		merge  User
		expect User
		iserr  bool
	}{
		{
			User{User: "foo"},
			User{User: "bar"},
			User{},
			true,
		},
		{
			User{User: "foo"},
			User{User: "foo"},
			User{User: "foo", Roles: []string{}},
			false,
		},
		{
			User{User: "foo"},
			User{User: "foo", Grant: []string{"role1"}},
			User{User: "foo", Roles: []string{"role1"}},
			false,
		},
		{
			User{User: "foo", Roles: []string{"role1"}},
			User{User: "foo", Grant: []string{"role1"}},
			User{},
			true,
		},
		{
			User{User: "foo", Roles: []string{"role1"}},
			User{User: "foo", Revoke: []string{"role2"}},
			User{},
			true,
		},
		{
			User{User: "foo", Roles: []string{"role1"}},
			User{User: "foo", Grant: []string{"role2"}},
			User{User: "foo", Roles: []string{"role1", "role2"}},
			false,
		},
		{ // empty password will not overwrite the previous password
			User{User: "foo", Password: "foo", Roles: []string{}},
			User{User: "foo", Password: ""},
			User{User: "foo", Password: "foo", Roles: []string{}},
			false,
		},
	}

	for i, tt := range tbl {
		out, err := tt.input.merge(tt.merge, passwordStore{})
		if err != nil && !tt.iserr {
			t.Fatalf("Got unexpected error on item %d", i)
		}
		if !tt.iserr {
			if !reflect.DeepEqual(out, tt.expect) {
				t.Errorf("Unequal merge expectation on item %d: got: %#v, expect: %#v", i, out, tt.expect)
			}
		}
	}
}

func TestMergeRole(t *testing.T) {
	tbl := []struct {
		input  Role
		merge  Role
		expect Role
		iserr  bool
	}{
		{
			Role{Role: "foo"},
			Role{Role: "bar"},
			Role{},
			true,
		},
		{
			Role{Role: "foo"},
			Role{Role: "foo", Grant: &Permissions{KV: RWPermission{Read: []string{"/foodir"}, Write: []string{"/foodir"}}}},
			Role{Role: "foo", Permissions: Permissions{KV: RWPermission{Read: []string{"/foodir"}, Write: []string{"/foodir"}}}},
			false,
		},
		{
			Role{Role: "foo", Permissions: Permissions{KV: RWPermission{Read: []string{"/foodir"}, Write: []string{"/foodir"}}}},
			Role{Role: "foo", Revoke: &Permissions{KV: RWPermission{Read: []string{"/foodir"}, Write: []string{"/foodir"}}}},
			Role{Role: "foo", Permissions: Permissions{KV: RWPermission{Read: []string{}, Write: []string{}}}},
			false,
		},
		{
			Role{Role: "foo", Permissions: Permissions{KV: RWPermission{Read: []string{"/bardir"}}}},
			Role{Role: "foo", Revoke: &Permissions{KV: RWPermission{Read: []string{"/foodir"}}}},
			Role{},
			true,
		},
	}
	for i, tt := range tbl {
		out, err := tt.input.merge(tt.merge)
		if err != nil && !tt.iserr {
			t.Fatalf("Got unexpected error on item %d", i)
		}
		if !tt.iserr {
			if !reflect.DeepEqual(out, tt.expect) {
				t.Errorf("Unequal merge expectation on item %d: got: %#v, expect: %#v", i, out, tt.expect)
			}
		}
	}
}

type testDoer struct {
	get               []etcdserver.Response
	put               []etcdserver.Response
	getindex          int
	putindex          int
	explicitlyEnabled bool
}

func (td *testDoer) Do(_ context.Context, req etcdserverpb.Request) (etcdserver.Response, error) {
	if td.explicitlyEnabled && (req.Path == StorePermsPrefix+"/enabled") {
		t := "true"
		return etcdserver.Response{
			Event: &etcdstore.Event{
				Action: etcdstore.Get,
				Node: &etcdstore.NodeExtern{
					Key:   StorePermsPrefix + "/users/cat",
					Value: &t,
				},
			},
		}, nil
	}
	if (req.Method == "GET" || req.Method == "QGET") && td.get != nil {
		res := td.get[td.getindex]
		if res.Event == nil {
			td.getindex++
			return etcdserver.Response{}, &etcderr.Error{
				ErrorCode: etcderr.EcodeKeyNotFound,
			}
		}
		td.getindex++
		return res, nil
	}
	if req.Method == "PUT" && td.put != nil {
		res := td.put[td.putindex]
		if res.Event == nil {
			td.putindex++
			return etcdserver.Response{}, &etcderr.Error{
				ErrorCode: etcderr.EcodeNodeExist,
			}
		}
		td.putindex++
		return res, nil
	}
	return etcdserver.Response{}, nil
}

func TestAllUsers(t *testing.T) {
	d := &testDoer{
		get: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Nodes: etcdstore.NodeExterns([]*etcdstore.NodeExtern{
							{
								Key: StorePermsPrefix + "/users/cat",
							},
							{
								Key: StorePermsPrefix + "/users/dog",
							},
						}),
					},
				},
			},
		},
	}
	expected := []string{"cat", "dog"}

	s := store{server: d, timeout: testTimeout, ensuredOnce: false}
	users, err := s.AllUsers()
	if err != nil {
		t.Error("Unexpected error", err)
	}
	if !reflect.DeepEqual(users, expected) {
		t.Error("AllUsers doesn't match given store. Got", users, "expected", expected)
	}
}

func TestGetAndDeleteUser(t *testing.T) {
	data := `{"user": "cat", "roles" : ["animal"]}`
	d := &testDoer{
		get: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/users/cat",
						Value: &data,
					},
				},
			},
		},
		explicitlyEnabled: true,
	}
	expected := User{User: "cat", Roles: []string{"animal"}}

	s := store{server: d, timeout: testTimeout, ensuredOnce: false}
	out, err := s.GetUser("cat")
	if err != nil {
		t.Error("Unexpected error", err)
	}
	if !reflect.DeepEqual(out, expected) {
		t.Error("GetUser doesn't match given store. Got", out, "expected", expected)
	}
	err = s.DeleteUser("cat")
	if err != nil {
		t.Error("Unexpected error", err)
	}
}

func TestAllRoles(t *testing.T) {
	d := &testDoer{
		get: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Nodes: etcdstore.NodeExterns([]*etcdstore.NodeExtern{
							{
								Key: StorePermsPrefix + "/roles/animal",
							},
							{
								Key: StorePermsPrefix + "/roles/human",
							},
						}),
					},
				},
			},
		},
		explicitlyEnabled: true,
	}
	expected := []string{"animal", "human", "root"}

	s := store{server: d, timeout: testTimeout, ensuredOnce: false}
	out, err := s.AllRoles()
	if err != nil {
		t.Error("Unexpected error", err)
	}
	if !reflect.DeepEqual(out, expected) {
		t.Error("AllRoles doesn't match given store. Got", out, "expected", expected)
	}
}

func TestGetAndDeleteRole(t *testing.T) {
	data := `{"role": "animal"}`
	d := &testDoer{
		get: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/roles/animal",
						Value: &data,
					},
				},
			},
		},
		explicitlyEnabled: true,
	}
	expected := Role{Role: "animal"}

	s := store{server: d, timeout: testTimeout, ensuredOnce: false}
	out, err := s.GetRole("animal")
	if err != nil {
		t.Error("Unexpected error", err)
	}
	if !reflect.DeepEqual(out, expected) {
		t.Error("GetRole doesn't match given store. Got", out, "expected", expected)
	}
	err = s.DeleteRole("animal")
	if err != nil {
		t.Error("Unexpected error", err)
	}
}

func TestEnsure(t *testing.T) {
	d := &testDoer{
		get: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Set,
					Node: &etcdstore.NodeExtern{
						Key: StorePermsPrefix,
						Dir: true,
					},
				},
			},
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Set,
					Node: &etcdstore.NodeExtern{
						Key: StorePermsPrefix + "/users/",
						Dir: true,
					},
				},
			},
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Set,
					Node: &etcdstore.NodeExtern{
						Key: StorePermsPrefix + "/roles/",
						Dir: true,
					},
				},
			},
		},
	}

	s := store{server: d, timeout: testTimeout, ensuredOnce: false}
	err := s.ensureAuthDirectories()
	if err != nil {
		t.Error("Unexpected error", err)
	}
}

type fastPasswordStore struct {
}

func (_ fastPasswordStore) CheckPassword(user User, password string) bool {
	return user.Password == password
}

func (_ fastPasswordStore) HashPassword(password string) (string, error) { return password, nil }

func TestCreateAndUpdateUser(t *testing.T) {
	olduser := `{"user": "cat", "roles" : ["animal"]}`
	newuser := `{"user": "cat", "roles" : ["animal", "pet"]}`
	d := &testDoer{
		get: []etcdserver.Response{
			{
				Event: nil,
			},
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/users/cat",
						Value: &olduser,
					},
				},
			},
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/users/cat",
						Value: &olduser,
					},
				},
			},
		},
		put: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Update,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/users/cat",
						Value: &olduser,
					},
				},
			},
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Update,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/users/cat",
						Value: &newuser,
					},
				},
			},
		},
		explicitlyEnabled: true,
	}
	user := User{User: "cat", Password: "meow", Roles: []string{"animal"}}
	update := User{User: "cat", Grant: []string{"pet"}}
	expected := User{User: "cat", Roles: []string{"animal", "pet"}}

	s := store{server: d, timeout: testTimeout, ensuredOnce: true, PasswordStore: fastPasswordStore{}}
	out, created, err := s.CreateOrUpdateUser(user)
	if !created {
		t.Error("Should have created user, instead updated?")
	}
	if err != nil {
		t.Error("Unexpected error", err)
	}
	out.Password = "meow"
	if !reflect.DeepEqual(out, user) {
		t.Error("UpdateUser doesn't match given update. Got", out, "expected", expected)
	}
	out, created, err = s.CreateOrUpdateUser(update)
	if created {
		t.Error("Should have updated user, instead created?")
	}
	if err != nil {
		t.Error("Unexpected error", err)
	}
	if !reflect.DeepEqual(out, expected) {
		t.Error("UpdateUser doesn't match given update. Got", out, "expected", expected)
	}
}

func TestUpdateRole(t *testing.T) {
	oldrole := `{"role": "animal", "permissions" : {"kv": {"read": ["/animal"], "write": []}}}`
	newrole := `{"role": "animal", "permissions" : {"kv": {"read": ["/animal"], "write": ["/animal"]}}}`
	d := &testDoer{
		get: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/roles/animal",
						Value: &oldrole,
					},
				},
			},
		},
		put: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Update,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/roles/animal",
						Value: &newrole,
					},
				},
			},
		},
		explicitlyEnabled: true,
	}
	update := Role{Role: "animal", Grant: &Permissions{KV: RWPermission{Read: []string{}, Write: []string{"/animal"}}}}
	expected := Role{Role: "animal", Permissions: Permissions{KV: RWPermission{Read: []string{"/animal"}, Write: []string{"/animal"}}}}

	s := store{server: d, timeout: testTimeout, ensuredOnce: true}
	out, err := s.UpdateRole(update)
	if err != nil {
		t.Error("Unexpected error", err)
	}
	if !reflect.DeepEqual(out, expected) {
		t.Error("UpdateRole doesn't match given update. Got", out, "expected", expected)
	}
}

func TestCreateRole(t *testing.T) {
	role := `{"role": "animal", "permissions" : {"kv": {"read": ["/animal"], "write": []}}}`
	d := &testDoer{
		put: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Create,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/roles/animal",
						Value: &role,
					},
				},
			},
			{
				Event: nil,
			},
		},
		explicitlyEnabled: true,
	}
	r := Role{Role: "animal", Permissions: Permissions{KV: RWPermission{Read: []string{"/animal"}, Write: []string{}}}}

	s := store{server: d, timeout: testTimeout, ensuredOnce: true}
	err := s.CreateRole(Role{Role: "root"})
	if err == nil {
		t.Error("Should error creating root role")
	}
	err = s.CreateRole(r)
	if err != nil {
		t.Error("Unexpected error", err)
	}
	err = s.CreateRole(r)
	if err == nil {
		t.Error("Creating duplicate role, should error")
	}
}

func TestEnableAuth(t *testing.T) {
	rootUser := `{"user": "root", "password": ""}`
	guestRole := `{"role": "guest", "permissions" : {"kv": {"read": ["*"], "write": ["*"]}}}`
	trueval := "true"
	falseval := "false"
	d := &testDoer{
		get: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/enabled",
						Value: &falseval,
					},
				},
			},
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/user/root",
						Value: &rootUser,
					},
				},
			},
			{
				Event: nil,
			},
		},
		put: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Create,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/roles/guest",
						Value: &guestRole,
					},
				},
			},
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Update,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/enabled",
						Value: &trueval,
					},
				},
			},
		},
		explicitlyEnabled: false,
	}
	s := store{server: d, timeout: testTimeout, ensuredOnce: true}
	err := s.EnableAuth()
	if err != nil {
		t.Error("Unexpected error", err)
	}
}

func TestDisableAuth(t *testing.T) {
	trueval := "true"
	falseval := "false"
	d := &testDoer{
		get: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/enabled",
						Value: &falseval,
					},
				},
			},
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Get,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/enabled",
						Value: &trueval,
					},
				},
			},
		},
		put: []etcdserver.Response{
			{
				Event: &etcdstore.Event{
					Action: etcdstore.Update,
					Node: &etcdstore.NodeExtern{
						Key:   StorePermsPrefix + "/enabled",
						Value: &falseval,
					},
				},
			},
		},
		explicitlyEnabled: false,
	}
	s := store{server: d, timeout: testTimeout, ensuredOnce: true}
	err := s.DisableAuth()
	if err == nil {
		t.Error("Expected error; already disabled")
	}

	err = s.DisableAuth()
	if err != nil {
		t.Error("Unexpected error", err)
	}
}

func TestSimpleMatch(t *testing.T) {
	role := Role{Role: "foo", Permissions: Permissions{KV: RWPermission{Read: []string{"/foodir/*", "/fookey"}, Write: []string{"/bardir/*", "/barkey"}}}}
	if !role.HasKeyAccess("/foodir/foo/bar", false) {
		t.Fatal("role lacks expected access")
	}
	if !role.HasKeyAccess("/fookey", false) {
		t.Fatal("role lacks expected access")
	}
	if !role.HasRecursiveAccess("/foodir/*", false) {
		t.Fatal("role lacks expected access")
	}
	if !role.HasRecursiveAccess("/foodir/foo*", false) {
		t.Fatal("role lacks expected access")
	}
	if !role.HasRecursiveAccess("/bardir/*", true) {
		t.Fatal("role lacks expected access")
	}
	if !role.HasKeyAccess("/bardir/bar/foo", true) {
		t.Fatal("role lacks expected access")
	}
	if !role.HasKeyAccess("/barkey", true) {
		t.Fatal("role lacks expected access")
	}

	if role.HasKeyAccess("/bardir/bar/foo", false) {
		t.Fatal("role has unexpected access")
	}
	if role.HasKeyAccess("/barkey", false) {
		t.Fatal("role has unexpected access")
	}
	if role.HasKeyAccess("/foodir/foo/bar", true) {
		t.Fatal("role has unexpected access")
	}
	if role.HasKeyAccess("/fookey", true) {
		t.Fatal("role has unexpected access")
	}
}
