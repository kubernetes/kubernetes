// Copyright 2016 The etcd Authors
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
	"os"
	"testing"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/mvcc/backend"
	"golang.org/x/crypto/bcrypt"
	"golang.org/x/net/context"
)

func init() { BcryptCost = bcrypt.MinCost }

func dummyIndexWaiter(index uint64) <-chan struct{} {
	ch := make(chan struct{})
	go func() {
		ch <- struct{}{}
	}()
	return ch
}

// TestNewAuthStoreRevision ensures newly auth store
// keeps the old revision when there are no changes.
func TestNewAuthStoreRevision(t *testing.T) {
	b, tPath := backend.NewDefaultTmpBackend()
	defer os.Remove(tPath)

	as := NewAuthStore(b, dummyIndexWaiter)
	err := enableAuthAndCreateRoot(as)
	if err != nil {
		t.Fatal(err)
	}
	old := as.Revision()
	b.Close()
	as.Close()

	// no changes to commit
	b2 := backend.NewDefaultBackend(tPath)
	as = NewAuthStore(b2, dummyIndexWaiter)
	new := as.Revision()
	b2.Close()
	as.Close()

	if old != new {
		t.Fatalf("expected revision %d, got %d", old, new)
	}
}

func enableAuthAndCreateRoot(as *authStore) error {
	_, err := as.UserAdd(&pb.AuthUserAddRequest{Name: "root", Password: "root"})
	if err != nil {
		return err
	}

	_, err = as.RoleAdd(&pb.AuthRoleAddRequest{Name: "root"})
	if err != nil {
		return err
	}

	_, err = as.UserGrantRole(&pb.AuthUserGrantRoleRequest{User: "root", Role: "root"})
	if err != nil {
		return err
	}

	return as.AuthEnable()
}

func TestCheckPassword(t *testing.T) {
	b, tPath := backend.NewDefaultTmpBackend()
	defer func() {
		b.Close()
		os.Remove(tPath)
	}()

	as := NewAuthStore(b, dummyIndexWaiter)
	defer as.Close()
	err := enableAuthAndCreateRoot(as)
	if err != nil {
		t.Fatal(err)
	}

	ua := &pb.AuthUserAddRequest{Name: "foo", Password: "bar"}
	_, err = as.UserAdd(ua)
	if err != nil {
		t.Fatal(err)
	}

	// auth a non-existing user
	_, err = as.CheckPassword("foo-test", "bar")
	if err == nil {
		t.Fatalf("expected %v, got %v", ErrAuthFailed, err)
	}
	if err != ErrAuthFailed {
		t.Fatalf("expected %v, got %v", ErrAuthFailed, err)
	}

	// auth an existing user with correct password
	_, err = as.CheckPassword("foo", "bar")
	if err != nil {
		t.Fatal(err)
	}

	// auth an existing user but with wrong password
	_, err = as.CheckPassword("foo", "")
	if err == nil {
		t.Fatalf("expected %v, got %v", ErrAuthFailed, err)
	}
	if err != ErrAuthFailed {
		t.Fatalf("expected %v, got %v", ErrAuthFailed, err)
	}
}

func TestUserDelete(t *testing.T) {
	b, tPath := backend.NewDefaultTmpBackend()
	defer func() {
		b.Close()
		os.Remove(tPath)
	}()

	as := NewAuthStore(b, dummyIndexWaiter)
	defer as.Close()
	err := enableAuthAndCreateRoot(as)
	if err != nil {
		t.Fatal(err)
	}

	ua := &pb.AuthUserAddRequest{Name: "foo"}
	_, err = as.UserAdd(ua)
	if err != nil {
		t.Fatal(err)
	}

	// delete an existing user
	ud := &pb.AuthUserDeleteRequest{Name: "foo"}
	_, err = as.UserDelete(ud)
	if err != nil {
		t.Fatal(err)
	}

	// delete a non-existing user
	_, err = as.UserDelete(ud)
	if err == nil {
		t.Fatalf("expected %v, got %v", ErrUserNotFound, err)
	}
	if err != ErrUserNotFound {
		t.Fatalf("expected %v, got %v", ErrUserNotFound, err)
	}
}

func TestUserChangePassword(t *testing.T) {
	b, tPath := backend.NewDefaultTmpBackend()
	defer func() {
		b.Close()
		os.Remove(tPath)
	}()

	as := NewAuthStore(b, dummyIndexWaiter)
	defer as.Close()
	err := enableAuthAndCreateRoot(as)
	if err != nil {
		t.Fatal(err)
	}

	_, err = as.UserAdd(&pb.AuthUserAddRequest{Name: "foo"})
	if err != nil {
		t.Fatal(err)
	}

	ctx1 := context.WithValue(context.WithValue(context.TODO(), "index", uint64(1)), "simpleToken", "dummy")
	_, err = as.Authenticate(ctx1, "foo", "")
	if err != nil {
		t.Fatal(err)
	}

	_, err = as.UserChangePassword(&pb.AuthUserChangePasswordRequest{Name: "foo", Password: "bar"})
	if err != nil {
		t.Fatal(err)
	}

	ctx2 := context.WithValue(context.WithValue(context.TODO(), "index", uint64(2)), "simpleToken", "dummy")
	_, err = as.Authenticate(ctx2, "foo", "bar")
	if err != nil {
		t.Fatal(err)
	}

	// change a non-existing user
	_, err = as.UserChangePassword(&pb.AuthUserChangePasswordRequest{Name: "foo-test", Password: "bar"})
	if err == nil {
		t.Fatalf("expected %v, got %v", ErrUserNotFound, err)
	}
	if err != ErrUserNotFound {
		t.Fatalf("expected %v, got %v", ErrUserNotFound, err)
	}
}

func TestRoleAdd(t *testing.T) {
	b, tPath := backend.NewDefaultTmpBackend()
	defer func() {
		b.Close()
		os.Remove(tPath)
	}()

	as := NewAuthStore(b, dummyIndexWaiter)
	defer as.Close()
	err := enableAuthAndCreateRoot(as)
	if err != nil {
		t.Fatal(err)
	}

	// adds a new role
	_, err = as.RoleAdd(&pb.AuthRoleAddRequest{Name: "role-test"})
	if err != nil {
		t.Fatal(err)
	}
}

func TestUserGrant(t *testing.T) {
	b, tPath := backend.NewDefaultTmpBackend()
	defer func() {
		b.Close()
		os.Remove(tPath)
	}()

	as := NewAuthStore(b, dummyIndexWaiter)
	defer as.Close()
	err := enableAuthAndCreateRoot(as)
	if err != nil {
		t.Fatal(err)
	}

	_, err = as.UserAdd(&pb.AuthUserAddRequest{Name: "foo"})
	if err != nil {
		t.Fatal(err)
	}

	// adds a new role
	_, err = as.RoleAdd(&pb.AuthRoleAddRequest{Name: "role-test"})
	if err != nil {
		t.Fatal(err)
	}

	// grants a role to the user
	_, err = as.UserGrantRole(&pb.AuthUserGrantRoleRequest{User: "foo", Role: "role-test"})
	if err != nil {
		t.Fatal(err)
	}

	// grants a role to a non-existing user
	_, err = as.UserGrantRole(&pb.AuthUserGrantRoleRequest{User: "foo-test", Role: "role-test"})
	if err == nil {
		t.Fatalf("expected %v, got %v", ErrUserNotFound, err)
	}
	if err != ErrUserNotFound {
		t.Fatalf("expected %v, got %v", ErrUserNotFound, err)
	}

	// non-admin user
	err = as.IsAdminPermitted(&AuthInfo{Username: "foo", Revision: 1})
	if err != ErrPermissionDenied {
		t.Errorf("expected %v, got %v", ErrPermissionDenied, err)
	}

	// disabled auth should return nil
	as.AuthDisable()
	err = as.IsAdminPermitted(&AuthInfo{Username: "root", Revision: 1})
	if err != nil {
		t.Errorf("expected nil, got %v", err)
	}
}

func TestRecoverFromSnapshot(t *testing.T) {
	as, _ := setupAuthStore(t)

	ua := &pb.AuthUserAddRequest{Name: "foo"}
	_, err := as.UserAdd(ua) // add an existing user
	if err == nil {
		t.Fatalf("expected %v, got %v", ErrUserAlreadyExist, err)
	}
	if err != ErrUserAlreadyExist {
		t.Fatalf("expected %v, got %v", ErrUserAlreadyExist, err)
	}

	ua = &pb.AuthUserAddRequest{Name: ""}
	_, err = as.UserAdd(ua) // add a user with empty name
	if err != ErrUserEmpty {
		t.Fatal(err)
	}

	as.Close()

	as2 := NewAuthStore(as.be, dummyIndexWaiter)
	defer func(a *authStore) {
		a.Close()
	}(as2)

	if !as2.isAuthEnabled() {
		t.Fatal("recovering authStore from existing backend failed")
	}

	ul, err := as.UserList(&pb.AuthUserListRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if !contains(ul.Users, "root") {
		t.Errorf("expected %v in %v", "root", ul.Users)
	}
}

func contains(array []string, str string) bool {
	for _, s := range array {
		if s == str {
			return true
		}
	}
	return false
}

func setupAuthStore(t *testing.T) (store *authStore, teardownfunc func(t *testing.T)) {
	b, tPath := backend.NewDefaultTmpBackend()

	as := NewAuthStore(b, dummyIndexWaiter)
	err := enableAuthAndCreateRoot(as)
	if err != nil {
		t.Fatal(err)
	}

	// adds a new role
	_, err = as.RoleAdd(&pb.AuthRoleAddRequest{Name: "role-test"})
	if err != nil {
		t.Fatal(err)
	}

	ua := &pb.AuthUserAddRequest{Name: "foo", Password: "bar"}
	_, err = as.UserAdd(ua) // add a non-existing user
	if err != nil {
		t.Fatal(err)
	}

	tearDown := func(t *testing.T) {
		b.Close()
		os.Remove(tPath)
		as.Close()
	}
	return as, tearDown
}
