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
	"golang.org/x/net/context"
)

func TestUserAdd(t *testing.T) {
	b, tPath := backend.NewDefaultTmpBackend()
	defer func() {
		b.Close()
		os.Remove(tPath)
	}()

	as := NewAuthStore(b)
	ua := &pb.AuthUserAddRequest{Name: "foo"}
	_, err := as.UserAdd(ua) // add a non-existing user
	if err != nil {
		t.Fatal(err)
	}
	_, err = as.UserAdd(ua) // add an existing user
	if err == nil {
		t.Fatalf("expected %v, got %v", ErrUserAlreadyExist, err)
	}
	if err != ErrUserAlreadyExist {
		t.Fatalf("expected %v, got %v", ErrUserAlreadyExist, err)
	}
}

func TestAuthenticate(t *testing.T) {
	b, tPath := backend.NewDefaultTmpBackend()
	defer func() {
		b.Close()
		os.Remove(tPath)
	}()

	as := NewAuthStore(b)

	ua := &pb.AuthUserAddRequest{Name: "foo", Password: "bar"}
	_, err := as.UserAdd(ua)
	if err != nil {
		t.Fatal(err)
	}

	// auth a non-existing user
	ctx1 := context.WithValue(context.WithValue(context.TODO(), "index", uint64(1)), "simpleToken", "dummy")
	_, err = as.Authenticate(ctx1, "foo-test", "bar")
	if err == nil {
		t.Fatalf("expected %v, got %v", ErrAuthFailed, err)
	}
	if err != ErrAuthFailed {
		t.Fatalf("expected %v, got %v", ErrAuthFailed, err)
	}

	// auth an existing user with correct password
	ctx2 := context.WithValue(context.WithValue(context.TODO(), "index", uint64(2)), "simpleToken", "dummy")
	_, err = as.Authenticate(ctx2, "foo", "bar")
	if err != nil {
		t.Fatal(err)
	}

	// auth an existing user but with wrong password
	ctx3 := context.WithValue(context.WithValue(context.TODO(), "index", uint64(3)), "simpleToken", "dummy")
	_, err = as.Authenticate(ctx3, "foo", "")
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

	as := NewAuthStore(b)

	ua := &pb.AuthUserAddRequest{Name: "foo"}
	_, err := as.UserAdd(ua)
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

	as := NewAuthStore(b)

	_, err := as.UserAdd(&pb.AuthUserAddRequest{Name: "foo"})
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

	as := NewAuthStore(b)

	// adds a new role
	_, err := as.RoleAdd(&pb.AuthRoleAddRequest{Name: "role-test"})
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

	as := NewAuthStore(b)

	_, err := as.UserAdd(&pb.AuthUserAddRequest{Name: "foo"})
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
}
