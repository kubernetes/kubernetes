// Copyright 2016 Nippon Telegraph and Telephone Corporation.
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
	"errors"

	"github.com/coreos/etcd/auth/authpb"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/storage/backend"
	"github.com/coreos/pkg/capnslog"
	"golang.org/x/crypto/bcrypt"
)

var (
	enableFlagKey = []byte("authEnabled")

	authBucketName      = []byte("auth")
	authUsersBucketName = []byte("authUsers")
	authRolesBucketName = []byte("authRoles")

	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "auth")

	ErrUserAlreadyExist = errors.New("auth: user already exists")
	ErrUserNotFound     = errors.New("auth: user not found")
	ErrRoleAlreadyExist = errors.New("auth: role already exists")
)

type AuthStore interface {
	// AuthEnable() turns on the authentication feature
	AuthEnable()

	// Recover recovers the state of auth store from the given backend
	Recover(b backend.Backend)

	// UserAdd adds a new user
	UserAdd(r *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error)

	// UserDelete deletes a user
	UserDelete(r *pb.AuthUserDeleteRequest) (*pb.AuthUserDeleteResponse, error)

	// UserChangePassword changes a password of a user
	UserChangePassword(r *pb.AuthUserChangePasswordRequest) (*pb.AuthUserChangePasswordResponse, error)

	// RoleAdd adds a new role
	RoleAdd(r *pb.AuthRoleAddRequest) (*pb.AuthRoleAddResponse, error)
}

type authStore struct {
	be backend.Backend
}

func (as *authStore) AuthEnable() {
	value := []byte{1}

	b := as.be
	tx := b.BatchTx()
	tx.Lock()
	tx.UnsafePut(authBucketName, enableFlagKey, value)
	tx.Unlock()
	b.ForceCommit()

	plog.Noticef("Authentication enabled")
}

func (as *authStore) Recover(be backend.Backend) {
	as.be = be
	// TODO(mitake): recovery process
}

func (as *authStore) UserAdd(r *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error) {
	plog.Noticef("adding a new user: %s", r.Name)

	hashed, err := bcrypt.GenerateFromPassword([]byte(r.Password), bcrypt.DefaultCost)
	if err != nil {
		plog.Errorf("failed to hash password: %s", err)
		return nil, err
	}

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	_, vs := tx.UnsafeRange(authUsersBucketName, []byte(r.Name), nil, 0)
	if len(vs) != 0 {
		return &pb.AuthUserAddResponse{}, ErrUserAlreadyExist
	}

	newUser := authpb.User{
		Name:     []byte(r.Name),
		Password: hashed,
	}

	marshaledUser, merr := newUser.Marshal()
	if merr != nil {
		plog.Errorf("failed to marshal a new user data: %s", merr)
		return nil, merr
	}

	tx.UnsafePut(authUsersBucketName, []byte(r.Name), marshaledUser)

	plog.Noticef("added a new user: %s", r.Name)

	return &pb.AuthUserAddResponse{}, nil
}

func (as *authStore) UserDelete(r *pb.AuthUserDeleteRequest) (*pb.AuthUserDeleteResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	_, vs := tx.UnsafeRange(authUsersBucketName, []byte(r.Name), nil, 0)
	if len(vs) != 1 {
		return &pb.AuthUserDeleteResponse{}, ErrUserNotFound
	}

	tx.UnsafeDelete(authUsersBucketName, []byte(r.Name))

	plog.Noticef("deleted a user: %s", r.Name)

	return &pb.AuthUserDeleteResponse{}, nil
}

func (as *authStore) UserChangePassword(r *pb.AuthUserChangePasswordRequest) (*pb.AuthUserChangePasswordResponse, error) {
	// TODO(mitake): measure the cost of bcrypt.GenerateFromPassword()
	// If the cost is too high, we should move the encryption to outside of the raft
	hashed, err := bcrypt.GenerateFromPassword([]byte(r.Password), bcrypt.DefaultCost)
	if err != nil {
		plog.Errorf("failed to hash password: %s", err)
		return nil, err
	}

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	_, vs := tx.UnsafeRange(authUsersBucketName, []byte(r.Name), nil, 0)
	if len(vs) != 1 {
		return &pb.AuthUserChangePasswordResponse{}, ErrUserNotFound
	}

	updatedUser := authpb.User{
		Name:     []byte(r.Name),
		Password: hashed,
	}

	marshaledUser, merr := updatedUser.Marshal()
	if merr != nil {
		plog.Errorf("failed to marshal a new user data: %s", merr)
		return nil, merr
	}

	tx.UnsafePut(authUsersBucketName, []byte(r.Name), marshaledUser)

	plog.Noticef("changed a password of a user: %s", r.Name)

	return &pb.AuthUserChangePasswordResponse{}, nil
}

func (as *authStore) RoleAdd(r *pb.AuthRoleAddRequest) (*pb.AuthRoleAddResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	_, vs := tx.UnsafeRange(authRolesBucketName, []byte(r.Name), nil, 0)
	if len(vs) != 0 {
		return nil, ErrRoleAlreadyExist
	}

	newRole := &authpb.Role{
		Name: []byte(r.Name),
	}

	marshaledRole, err := newRole.Marshal()
	if err != nil {
		return nil, err
	}

	tx.UnsafePut(authRolesBucketName, []byte(r.Name), marshaledRole)

	plog.Noticef("Role %s is created", r.Name)

	return &pb.AuthRoleAddResponse{}, nil
}

func NewAuthStore(be backend.Backend) *authStore {
	tx := be.BatchTx()
	tx.Lock()

	tx.UnsafeCreateBucket(authBucketName)
	tx.UnsafeCreateBucket(authUsersBucketName)
	tx.UnsafeCreateBucket(authRolesBucketName)

	tx.Unlock()
	be.ForceCommit()

	return &authStore{
		be: be,
	}
}
