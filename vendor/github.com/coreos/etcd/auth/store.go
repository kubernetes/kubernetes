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
	"bytes"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/coreos/etcd/auth/authpb"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/mvcc/backend"
	"github.com/coreos/pkg/capnslog"
	"golang.org/x/crypto/bcrypt"
	"golang.org/x/net/context"
)

var (
	enableFlagKey = []byte("authEnabled")
	authEnabled   = []byte{1}
	authDisabled  = []byte{0}

	authBucketName      = []byte("auth")
	authUsersBucketName = []byte("authUsers")
	authRolesBucketName = []byte("authRoles")

	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "auth")

	ErrRootUserNotExist     = errors.New("auth: root user does not exist")
	ErrRootRoleNotExist     = errors.New("auth: root user does not have root role")
	ErrUserAlreadyExist     = errors.New("auth: user already exists")
	ErrUserNotFound         = errors.New("auth: user not found")
	ErrRoleAlreadyExist     = errors.New("auth: role already exists")
	ErrRoleNotFound         = errors.New("auth: role not found")
	ErrAuthFailed           = errors.New("auth: authentication failed, invalid user ID or password")
	ErrPermissionDenied     = errors.New("auth: permission denied")
	ErrRoleNotGranted       = errors.New("auth: role is not granted to the user")
	ErrPermissionNotGranted = errors.New("auth: permission is not granted to the role")
)

const (
	rootUser = "root"
	rootRole = "root"
)

type AuthStore interface {
	// AuthEnable turns on the authentication feature
	AuthEnable() error

	// AuthDisable turns off the authentication feature
	AuthDisable()

	// Authenticate does authentication based on given user name and password
	Authenticate(ctx context.Context, username, password string) (*pb.AuthenticateResponse, error)

	// Recover recovers the state of auth store from the given backend
	Recover(b backend.Backend)

	// UserAdd adds a new user
	UserAdd(r *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error)

	// UserDelete deletes a user
	UserDelete(r *pb.AuthUserDeleteRequest) (*pb.AuthUserDeleteResponse, error)

	// UserChangePassword changes a password of a user
	UserChangePassword(r *pb.AuthUserChangePasswordRequest) (*pb.AuthUserChangePasswordResponse, error)

	// UserGrantRole grants a role to the user
	UserGrantRole(r *pb.AuthUserGrantRoleRequest) (*pb.AuthUserGrantRoleResponse, error)

	// UserGet gets the detailed information of a users
	UserGet(r *pb.AuthUserGetRequest) (*pb.AuthUserGetResponse, error)

	// UserRevokeRole revokes a role of a user
	UserRevokeRole(r *pb.AuthUserRevokeRoleRequest) (*pb.AuthUserRevokeRoleResponse, error)

	// RoleAdd adds a new role
	RoleAdd(r *pb.AuthRoleAddRequest) (*pb.AuthRoleAddResponse, error)

	// RoleGrantPermission grants a permission to a role
	RoleGrantPermission(r *pb.AuthRoleGrantPermissionRequest) (*pb.AuthRoleGrantPermissionResponse, error)

	// RoleGet gets the detailed information of a role
	RoleGet(r *pb.AuthRoleGetRequest) (*pb.AuthRoleGetResponse, error)

	// RoleRevokePermission gets the detailed information of a role
	RoleRevokePermission(r *pb.AuthRoleRevokePermissionRequest) (*pb.AuthRoleRevokePermissionResponse, error)

	// RoleDelete gets the detailed information of a role
	RoleDelete(r *pb.AuthRoleDeleteRequest) (*pb.AuthRoleDeleteResponse, error)

	// UserList gets a list of all users
	UserList(r *pb.AuthUserListRequest) (*pb.AuthUserListResponse, error)

	// RoleList gets a list of all roles
	RoleList(r *pb.AuthRoleListRequest) (*pb.AuthRoleListResponse, error)

	// UsernameFromToken gets a username from the given Token
	UsernameFromToken(token string) (string, bool)

	// IsPutPermitted checks put permission of the user
	IsPutPermitted(username string, key []byte) bool

	// IsRangePermitted checks range permission of the user
	IsRangePermitted(username string, key, rangeEnd []byte) bool

	// IsDeleteRangePermitted checks delete-range permission of the user
	IsDeleteRangePermitted(username string, key, rangeEnd []byte) bool

	// IsAdminPermitted checks admin permission of the user
	IsAdminPermitted(username string) bool

	// GenSimpleToken produces a simple random string
	GenSimpleToken() (string, error)
}

type authStore struct {
	be        backend.Backend
	enabled   bool
	enabledMu sync.RWMutex

	rangePermCache map[string]*unifiedRangePermissions // username -> unifiedRangePermissions

	simpleTokensMu sync.RWMutex
	simpleTokens   map[string]string // token -> username
}

func (as *authStore) AuthEnable() error {
	b := as.be
	tx := b.BatchTx()
	tx.Lock()
	defer func() {
		tx.Unlock()
		b.ForceCommit()
	}()

	u := getUser(tx, rootUser)
	if u == nil {
		return ErrRootUserNotExist
	}

	if !hasRootRole(u) {
		return ErrRootRoleNotExist
	}

	tx.UnsafePut(authBucketName, enableFlagKey, authEnabled)

	as.enabledMu.Lock()
	as.enabled = true
	as.enabledMu.Unlock()

	as.rangePermCache = make(map[string]*unifiedRangePermissions)

	plog.Noticef("Authentication enabled")

	return nil
}

func (as *authStore) AuthDisable() {
	b := as.be
	tx := b.BatchTx()
	tx.Lock()
	tx.UnsafePut(authBucketName, enableFlagKey, authDisabled)
	tx.Unlock()
	b.ForceCommit()

	as.enabledMu.Lock()
	as.enabled = false
	as.enabledMu.Unlock()

	plog.Noticef("Authentication disabled")
}

func (as *authStore) Authenticate(ctx context.Context, username, password string) (*pb.AuthenticateResponse, error) {
	// TODO(mitake): after adding jwt support, branching based on values of ctx is required
	index := ctx.Value("index").(uint64)
	simpleToken := ctx.Value("simpleToken").(string)

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	user := getUser(tx, username)
	if user == nil {
		return nil, ErrAuthFailed
	}

	if bcrypt.CompareHashAndPassword(user.Password, []byte(password)) != nil {
		plog.Noticef("authentication failed, invalid password for user %s", username)
		return &pb.AuthenticateResponse{}, ErrAuthFailed
	}

	token := fmt.Sprintf("%s.%d", simpleToken, index)
	as.assignSimpleTokenToUser(username, token)

	plog.Infof("authorized %s, token is %s", username, token)
	return &pb.AuthenticateResponse{Token: token}, nil
}

func (as *authStore) Recover(be backend.Backend) {
	enabled := false
	as.be = be
	tx := be.BatchTx()
	tx.Lock()
	_, vs := tx.UnsafeRange(authBucketName, enableFlagKey, nil, 0)
	if len(vs) == 1 {
		if bytes.Equal(vs[0], authEnabled) {
			enabled = true
		}
	}
	tx.Unlock()

	as.enabledMu.Lock()
	as.enabled = enabled
	as.enabledMu.Unlock()
}

func (as *authStore) UserAdd(r *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error) {
	hashed, err := bcrypt.GenerateFromPassword([]byte(r.Password), bcrypt.DefaultCost)
	if err != nil {
		plog.Errorf("failed to hash password: %s", err)
		return nil, err
	}

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	user := getUser(tx, r.Name)
	if user != nil {
		return nil, ErrUserAlreadyExist
	}

	newUser := &authpb.User{
		Name:     []byte(r.Name),
		Password: hashed,
	}

	putUser(tx, newUser)

	plog.Noticef("added a new user: %s", r.Name)

	return &pb.AuthUserAddResponse{}, nil
}

func (as *authStore) UserDelete(r *pb.AuthUserDeleteRequest) (*pb.AuthUserDeleteResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	user := getUser(tx, r.Name)
	if user == nil {
		return nil, ErrUserNotFound
	}

	delUser(tx, r.Name)

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

	user := getUser(tx, r.Name)
	if user == nil {
		return nil, ErrUserNotFound
	}

	updatedUser := &authpb.User{
		Name:     []byte(r.Name),
		Roles:    user.Roles,
		Password: hashed,
	}

	putUser(tx, updatedUser)

	plog.Noticef("changed a password of a user: %s", r.Name)

	return &pb.AuthUserChangePasswordResponse{}, nil
}

func (as *authStore) UserGrantRole(r *pb.AuthUserGrantRoleRequest) (*pb.AuthUserGrantRoleResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	user := getUser(tx, r.User)
	if user == nil {
		return nil, ErrUserNotFound
	}

	if r.Role != rootRole {
		role := getRole(tx, r.Role)
		if role == nil {
			return nil, ErrRoleNotFound
		}
	}

	idx := sort.SearchStrings(user.Roles, r.Role)
	if idx < len(user.Roles) && strings.Compare(user.Roles[idx], r.Role) == 0 {
		plog.Warningf("user %s is already granted role %s", r.User, r.Role)
		return &pb.AuthUserGrantRoleResponse{}, nil
	}

	user.Roles = append(user.Roles, r.Role)
	sort.Sort(sort.StringSlice(user.Roles))

	putUser(tx, user)

	as.invalidateCachedPerm(r.User)

	plog.Noticef("granted role %s to user %s", r.Role, r.User)
	return &pb.AuthUserGrantRoleResponse{}, nil
}

func (as *authStore) UserGet(r *pb.AuthUserGetRequest) (*pb.AuthUserGetResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	var resp pb.AuthUserGetResponse

	user := getUser(tx, r.Name)
	if user == nil {
		return nil, ErrUserNotFound
	}

	for _, role := range user.Roles {
		resp.Roles = append(resp.Roles, role)
	}

	return &resp, nil
}

func (as *authStore) UserList(r *pb.AuthUserListRequest) (*pb.AuthUserListResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	var resp pb.AuthUserListResponse

	users := getAllUsers(tx)

	for _, u := range users {
		resp.Users = append(resp.Users, string(u.Name))
	}

	return &resp, nil
}

func (as *authStore) UserRevokeRole(r *pb.AuthUserRevokeRoleRequest) (*pb.AuthUserRevokeRoleResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	user := getUser(tx, r.Name)
	if user == nil {
		return nil, ErrUserNotFound
	}

	updatedUser := &authpb.User{
		Name:     user.Name,
		Password: user.Password,
	}

	for _, role := range user.Roles {
		if strings.Compare(role, r.Role) != 0 {
			updatedUser.Roles = append(updatedUser.Roles, role)
		}
	}

	if len(updatedUser.Roles) == len(user.Roles) {
		return nil, ErrRoleNotGranted
	}

	putUser(tx, updatedUser)

	as.invalidateCachedPerm(r.Name)

	plog.Noticef("revoked role %s from user %s", r.Role, r.Name)
	return &pb.AuthUserRevokeRoleResponse{}, nil
}

func (as *authStore) RoleGet(r *pb.AuthRoleGetRequest) (*pb.AuthRoleGetResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	var resp pb.AuthRoleGetResponse

	role := getRole(tx, r.Role)
	if role == nil {
		return nil, ErrRoleNotFound
	}

	for _, perm := range role.KeyPermission {
		resp.Perm = append(resp.Perm, perm)
	}

	return &resp, nil
}

func (as *authStore) RoleList(r *pb.AuthRoleListRequest) (*pb.AuthRoleListResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	var resp pb.AuthRoleListResponse

	roles := getAllRoles(tx)

	for _, r := range roles {
		resp.Roles = append(resp.Roles, string(r.Name))
	}

	return &resp, nil
}

func (as *authStore) RoleRevokePermission(r *pb.AuthRoleRevokePermissionRequest) (*pb.AuthRoleRevokePermissionResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	role := getRole(tx, r.Role)
	if role == nil {
		return nil, ErrRoleNotFound
	}

	updatedRole := &authpb.Role{
		Name: role.Name,
	}

	for _, perm := range role.KeyPermission {
		if !bytes.Equal(perm.Key, []byte(r.Key)) || !bytes.Equal(perm.RangeEnd, []byte(r.RangeEnd)) {
			updatedRole.KeyPermission = append(updatedRole.KeyPermission, perm)
		}
	}

	if len(role.KeyPermission) == len(updatedRole.KeyPermission) {
		return nil, ErrPermissionNotGranted
	}

	putRole(tx, updatedRole)

	// TODO(mitake): currently single role update invalidates every cache
	// It should be optimized.
	as.clearCachedPerm()

	plog.Noticef("revoked key %s from role %s", r.Key, r.Role)
	return &pb.AuthRoleRevokePermissionResponse{}, nil
}

func (as *authStore) RoleDelete(r *pb.AuthRoleDeleteRequest) (*pb.AuthRoleDeleteResponse, error) {
	// TODO(mitake): current scheme of role deletion allows existing users to have the deleted roles
	//
	// Assume a case like below:
	// create a role r1
	// create a user u1 and grant r1 to u1
	// delete r1
	//
	// After this sequence, u1 is still granted the role r1. So if admin create a new role with the name r1,
	// the new r1 is automatically granted u1.
	// In some cases, it would be confusing. So we need to provide an option for deleting the grant relation
	// from all users.

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	role := getRole(tx, r.Role)
	if role == nil {
		return nil, ErrRoleNotFound
	}

	delRole(tx, r.Role)

	plog.Noticef("deleted role %s", r.Role)
	return &pb.AuthRoleDeleteResponse{}, nil
}

func (as *authStore) RoleAdd(r *pb.AuthRoleAddRequest) (*pb.AuthRoleAddResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	role := getRole(tx, r.Name)
	if role != nil {
		return nil, ErrRoleAlreadyExist
	}

	newRole := &authpb.Role{
		Name: []byte(r.Name),
	}

	putRole(tx, newRole)

	plog.Noticef("Role %s is created", r.Name)

	return &pb.AuthRoleAddResponse{}, nil
}

func (as *authStore) UsernameFromToken(token string) (string, bool) {
	as.simpleTokensMu.RLock()
	defer as.simpleTokensMu.RUnlock()
	t, ok := as.simpleTokens[token]
	return t, ok
}

type permSlice []*authpb.Permission

func (perms permSlice) Len() int {
	return len(perms)
}

func (perms permSlice) Less(i, j int) bool {
	return bytes.Compare(perms[i].Key, perms[j].Key) < 0
}

func (perms permSlice) Swap(i, j int) {
	perms[i], perms[j] = perms[j], perms[i]
}

func (as *authStore) RoleGrantPermission(r *pb.AuthRoleGrantPermissionRequest) (*pb.AuthRoleGrantPermissionResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	role := getRole(tx, r.Name)
	if role == nil {
		return nil, ErrRoleNotFound
	}

	idx := sort.Search(len(role.KeyPermission), func(i int) bool {
		return bytes.Compare(role.KeyPermission[i].Key, []byte(r.Perm.Key)) >= 0
	})

	if idx < len(role.KeyPermission) && bytes.Equal(role.KeyPermission[idx].Key, r.Perm.Key) && bytes.Equal(role.KeyPermission[idx].RangeEnd, r.Perm.RangeEnd) {
		// update existing permission
		role.KeyPermission[idx].PermType = r.Perm.PermType
	} else {
		// append new permission to the role
		newPerm := &authpb.Permission{
			Key:      []byte(r.Perm.Key),
			RangeEnd: []byte(r.Perm.RangeEnd),
			PermType: r.Perm.PermType,
		}

		role.KeyPermission = append(role.KeyPermission, newPerm)
		sort.Sort(permSlice(role.KeyPermission))
	}

	putRole(tx, role)

	// TODO(mitake): currently single role update invalidates every cache
	// It should be optimized.
	as.clearCachedPerm()

	plog.Noticef("role %s's permission of key %s is updated as %s", r.Name, r.Perm.Key, authpb.Permission_Type_name[int32(r.Perm.PermType)])

	return &pb.AuthRoleGrantPermissionResponse{}, nil
}

func (as *authStore) isOpPermitted(userName string, key, rangeEnd []byte, permTyp authpb.Permission_Type) bool {
	// TODO(mitake): this function would be costly so we need a caching mechanism
	if !as.isAuthEnabled() {
		return true
	}

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	user := getUser(tx, userName)
	if user == nil {
		plog.Errorf("invalid user name %s for permission checking", userName)
		return false
	}

	// root role should have permission on all ranges
	if hasRootRole(user) {
		return true
	}

	if as.isRangeOpPermitted(tx, userName, key, rangeEnd, permTyp) {
		return true
	}

	return false
}

func (as *authStore) IsPutPermitted(username string, key []byte) bool {
	return as.isOpPermitted(username, key, nil, authpb.WRITE)
}

func (as *authStore) IsRangePermitted(username string, key, rangeEnd []byte) bool {
	return as.isOpPermitted(username, key, rangeEnd, authpb.READ)
}

func (as *authStore) IsDeleteRangePermitted(username string, key, rangeEnd []byte) bool {
	return as.isOpPermitted(username, key, rangeEnd, authpb.WRITE)
}

func (as *authStore) IsAdminPermitted(username string) bool {
	if !as.isAuthEnabled() {
		return true
	}

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	u := getUser(tx, username)
	if u == nil {
		return false
	}

	return hasRootRole(u)
}

func getUser(tx backend.BatchTx, username string) *authpb.User {
	_, vs := tx.UnsafeRange(authUsersBucketName, []byte(username), nil, 0)
	if len(vs) == 0 {
		return nil
	}

	user := &authpb.User{}
	err := user.Unmarshal(vs[0])
	if err != nil {
		plog.Panicf("failed to unmarshal user struct (name: %s): %s", username, err)
	}
	return user
}

func getAllUsers(tx backend.BatchTx) []*authpb.User {
	_, vs := tx.UnsafeRange(authUsersBucketName, []byte{0}, []byte{0xff}, -1)
	if len(vs) == 0 {
		return nil
	}

	var users []*authpb.User

	for _, v := range vs {
		user := &authpb.User{}
		err := user.Unmarshal(v)
		if err != nil {
			plog.Panicf("failed to unmarshal user struct: %s", err)
		}

		users = append(users, user)
	}

	return users
}

func putUser(tx backend.BatchTx, user *authpb.User) {
	b, err := user.Marshal()
	if err != nil {
		plog.Panicf("failed to marshal user struct (name: %s): %s", user.Name, err)
	}
	tx.UnsafePut(authUsersBucketName, user.Name, b)
}

func delUser(tx backend.BatchTx, username string) {
	tx.UnsafeDelete(authUsersBucketName, []byte(username))
}

func getRole(tx backend.BatchTx, rolename string) *authpb.Role {
	_, vs := tx.UnsafeRange(authRolesBucketName, []byte(rolename), nil, 0)
	if len(vs) == 0 {
		return nil
	}

	role := &authpb.Role{}
	err := role.Unmarshal(vs[0])
	if err != nil {
		plog.Panicf("failed to unmarshal role struct (name: %s): %s", rolename, err)
	}
	return role
}

func getAllRoles(tx backend.BatchTx) []*authpb.Role {
	_, vs := tx.UnsafeRange(authRolesBucketName, []byte{0}, []byte{0xff}, -1)
	if len(vs) == 0 {
		return nil
	}

	var roles []*authpb.Role

	for _, v := range vs {
		role := &authpb.Role{}
		err := role.Unmarshal(v)
		if err != nil {
			plog.Panicf("failed to unmarshal role struct: %s", err)
		}

		roles = append(roles, role)
	}

	return roles
}

func putRole(tx backend.BatchTx, role *authpb.Role) {
	b, err := role.Marshal()
	if err != nil {
		plog.Panicf("failed to marshal role struct (name: %s): %s", role.Name, err)
	}

	tx.UnsafePut(authRolesBucketName, []byte(role.Name), b)
}

func delRole(tx backend.BatchTx, rolename string) {
	tx.UnsafeDelete(authRolesBucketName, []byte(rolename))
}

func (as *authStore) isAuthEnabled() bool {
	as.enabledMu.RLock()
	defer as.enabledMu.RUnlock()
	return as.enabled
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
		be:           be,
		simpleTokens: make(map[string]string),
	}
}

func hasRootRole(u *authpb.User) bool {
	for _, r := range u.Roles {
		if r == rootRole {
			return true
		}
	}
	return false
}
