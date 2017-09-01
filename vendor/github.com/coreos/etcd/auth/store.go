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
	"encoding/binary"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/coreos/etcd/auth/authpb"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/mvcc/backend"
	"github.com/coreos/pkg/capnslog"
	"golang.org/x/crypto/bcrypt"
	"golang.org/x/net/context"
	"google.golang.org/grpc/metadata"
)

var (
	enableFlagKey = []byte("authEnabled")
	authEnabled   = []byte{1}
	authDisabled  = []byte{0}

	revisionKey = []byte("authRevision")

	authBucketName      = []byte("auth")
	authUsersBucketName = []byte("authUsers")
	authRolesBucketName = []byte("authRoles")

	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "auth")

	ErrRootUserNotExist     = errors.New("auth: root user does not exist")
	ErrRootRoleNotExist     = errors.New("auth: root user does not have root role")
	ErrUserAlreadyExist     = errors.New("auth: user already exists")
	ErrUserEmpty            = errors.New("auth: user name is empty")
	ErrUserNotFound         = errors.New("auth: user not found")
	ErrRoleAlreadyExist     = errors.New("auth: role already exists")
	ErrRoleNotFound         = errors.New("auth: role not found")
	ErrAuthFailed           = errors.New("auth: authentication failed, invalid user ID or password")
	ErrPermissionDenied     = errors.New("auth: permission denied")
	ErrRoleNotGranted       = errors.New("auth: role is not granted to the user")
	ErrPermissionNotGranted = errors.New("auth: permission is not granted to the role")
	ErrAuthNotEnabled       = errors.New("auth: authentication is not enabled")
	ErrAuthOldRevision      = errors.New("auth: revision in header is old")
	ErrInvalidAuthToken     = errors.New("auth: invalid auth token")

	// BcryptCost is the algorithm cost / strength for hashing auth passwords
	BcryptCost = bcrypt.DefaultCost
)

const (
	rootUser = "root"
	rootRole = "root"

	revBytesLen = 8
)

type AuthInfo struct {
	Username string
	Revision uint64
}

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

	// AuthInfoFromToken gets a username from the given Token and current revision number
	// (The revision number is used for preventing the TOCTOU problem)
	AuthInfoFromToken(token string) (*AuthInfo, bool)

	// IsPutPermitted checks put permission of the user
	IsPutPermitted(authInfo *AuthInfo, key []byte) error

	// IsRangePermitted checks range permission of the user
	IsRangePermitted(authInfo *AuthInfo, key, rangeEnd []byte) error

	// IsDeleteRangePermitted checks delete-range permission of the user
	IsDeleteRangePermitted(authInfo *AuthInfo, key, rangeEnd []byte) error

	// IsAdminPermitted checks admin permission of the user
	IsAdminPermitted(authInfo *AuthInfo) error

	// GenSimpleToken produces a simple random string
	GenSimpleToken() (string, error)

	// Revision gets current revision of authStore
	Revision() uint64

	// CheckPassword checks a given pair of username and password is correct
	CheckPassword(username, password string) (uint64, error)

	// Close does cleanup of AuthStore
	Close() error

	// AuthInfoFromCtx gets AuthInfo from gRPC's context
	AuthInfoFromCtx(ctx context.Context) (*AuthInfo, error)
}

type authStore struct {
	be        backend.Backend
	enabled   bool
	enabledMu sync.RWMutex

	rangePermCache map[string]*unifiedRangePermissions // username -> unifiedRangePermissions

	revision uint64

	// tokenSimple in v3.2+
	indexWaiter       func(uint64) <-chan struct{}
	simpleTokenKeeper *simpleTokenTTLKeeper
	simpleTokensMu    sync.Mutex
	simpleTokens      map[string]string // token -> username
}

func newDeleterFunc(as *authStore) func(string) {
	return func(t string) {
		as.simpleTokensMu.Lock()
		defer as.simpleTokensMu.Unlock()
		if username, ok := as.simpleTokens[t]; ok {
			plog.Infof("deleting token %s for user %s", t, username)
			delete(as.simpleTokens, t)
		}
	}
}

func (as *authStore) AuthEnable() error {
	as.enabledMu.Lock()
	defer as.enabledMu.Unlock()
	if as.enabled {
		plog.Noticef("Authentication already enabled")
		return nil
	}
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

	as.enabled = true
	as.enable()

	as.rangePermCache = make(map[string]*unifiedRangePermissions)

	as.revision = getRevision(tx)

	plog.Noticef("Authentication enabled")

	return nil
}

func (as *authStore) AuthDisable() {
	as.enabledMu.Lock()
	defer as.enabledMu.Unlock()
	if !as.enabled {
		return
	}
	b := as.be
	tx := b.BatchTx()
	tx.Lock()
	tx.UnsafePut(authBucketName, enableFlagKey, authDisabled)
	as.commitRevision(tx)
	tx.Unlock()
	b.ForceCommit()

	as.enabled = false

	as.simpleTokensMu.Lock()
	tk := as.simpleTokenKeeper
	as.simpleTokenKeeper = nil
	as.simpleTokens = make(map[string]string) // invalidate all tokens
	as.simpleTokensMu.Unlock()
	if tk != nil {
		tk.stop()
	}

	plog.Noticef("Authentication disabled")
}

func (as *authStore) Close() error {
	as.enabledMu.Lock()
	defer as.enabledMu.Unlock()
	if !as.enabled {
		return nil
	}
	if as.simpleTokenKeeper != nil {
		as.simpleTokenKeeper.stop()
		as.simpleTokenKeeper = nil
	}
	return nil
}

func (as *authStore) Authenticate(ctx context.Context, username, password string) (*pb.AuthenticateResponse, error) {
	if !as.isAuthEnabled() {
		return nil, ErrAuthNotEnabled
	}

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

	token := fmt.Sprintf("%s.%d", simpleToken, index)
	as.assignSimpleTokenToUser(username, token)

	plog.Infof("authorized %s, token is %s", username, token)
	return &pb.AuthenticateResponse{Token: token}, nil
}

func (as *authStore) CheckPassword(username, password string) (uint64, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	user := getUser(tx, username)
	if user == nil {
		return 0, ErrAuthFailed
	}

	if bcrypt.CompareHashAndPassword(user.Password, []byte(password)) != nil {
		plog.Noticef("authentication failed, invalid password for user %s", username)
		return 0, ErrAuthFailed
	}

	return getRevision(tx), nil
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

	as.revision = getRevision(tx)

	tx.Unlock()

	as.enabledMu.Lock()
	as.enabled = enabled
	as.enabledMu.Unlock()
}

func (as *authStore) UserAdd(r *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error) {
	if len(r.Name) == 0 {
		return nil, ErrUserEmpty
	}

	hashed, err := bcrypt.GenerateFromPassword([]byte(r.Password), BcryptCost)
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

	as.commitRevision(tx)

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

	as.commitRevision(tx)

	as.invalidateCachedPerm(r.Name)
	as.invalidateUser(r.Name)

	plog.Noticef("deleted a user: %s", r.Name)

	return &pb.AuthUserDeleteResponse{}, nil
}

func (as *authStore) UserChangePassword(r *pb.AuthUserChangePasswordRequest) (*pb.AuthUserChangePasswordResponse, error) {
	// TODO(mitake): measure the cost of bcrypt.GenerateFromPassword()
	// If the cost is too high, we should move the encryption to outside of the raft
	hashed, err := bcrypt.GenerateFromPassword([]byte(r.Password), BcryptCost)
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

	as.commitRevision(tx)

	as.invalidateCachedPerm(r.Name)
	as.invalidateUser(r.Name)

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

	as.commitRevision(tx)

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
	resp.Roles = append(resp.Roles, user.Roles...)
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

	as.commitRevision(tx)

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
	resp.Perm = append(resp.Perm, role.KeyPermission...)
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

	as.commitRevision(tx)

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

	as.commitRevision(tx)

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

	as.commitRevision(tx)

	plog.Noticef("Role %s is created", r.Name)

	return &pb.AuthRoleAddResponse{}, nil
}

func (as *authStore) AuthInfoFromToken(token string) (*AuthInfo, bool) {
	// same as '(t *tokenSimple) info' in v3.2+
	as.simpleTokensMu.Lock()
	username, ok := as.simpleTokens[token]
	if ok && as.simpleTokenKeeper != nil {
		as.simpleTokenKeeper.resetSimpleToken(token)
	}
	as.simpleTokensMu.Unlock()
	return &AuthInfo{Username: username, Revision: as.revision}, ok
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

	as.commitRevision(tx)

	plog.Noticef("role %s's permission of key %s is updated as %s", r.Name, r.Perm.Key, authpb.Permission_Type_name[int32(r.Perm.PermType)])

	return &pb.AuthRoleGrantPermissionResponse{}, nil
}

func (as *authStore) isOpPermitted(userName string, revision uint64, key, rangeEnd []byte, permTyp authpb.Permission_Type) error {
	// TODO(mitake): this function would be costly so we need a caching mechanism
	if !as.isAuthEnabled() {
		return nil
	}

	// only gets rev == 0 when passed AuthInfo{}; no user given
	if revision == 0 {
		return ErrUserEmpty
	}

	if revision < as.revision {
		return ErrAuthOldRevision
	}

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	user := getUser(tx, userName)
	if user == nil {
		plog.Errorf("invalid user name %s for permission checking", userName)
		return ErrPermissionDenied
	}

	// root role should have permission on all ranges
	if hasRootRole(user) {
		return nil
	}

	if as.isRangeOpPermitted(tx, userName, key, rangeEnd, permTyp) {
		return nil
	}

	return ErrPermissionDenied
}

func (as *authStore) IsPutPermitted(authInfo *AuthInfo, key []byte) error {
	return as.isOpPermitted(authInfo.Username, authInfo.Revision, key, nil, authpb.WRITE)
}

func (as *authStore) IsRangePermitted(authInfo *AuthInfo, key, rangeEnd []byte) error {
	return as.isOpPermitted(authInfo.Username, authInfo.Revision, key, rangeEnd, authpb.READ)
}

func (as *authStore) IsDeleteRangePermitted(authInfo *AuthInfo, key, rangeEnd []byte) error {
	return as.isOpPermitted(authInfo.Username, authInfo.Revision, key, rangeEnd, authpb.WRITE)
}

func (as *authStore) IsAdminPermitted(authInfo *AuthInfo) error {
	if !as.isAuthEnabled() {
		return nil
	}
	if authInfo == nil {
		return ErrUserEmpty
	}

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	u := getUser(tx, authInfo.Username)
	if u == nil {
		return ErrUserNotFound
	}

	if !hasRootRole(u) {
		return ErrPermissionDenied
	}

	return nil
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

func NewAuthStore(be backend.Backend, indexWaiter func(uint64) <-chan struct{}) *authStore {
	tx := be.BatchTx()
	tx.Lock()

	tx.UnsafeCreateBucket(authBucketName)
	tx.UnsafeCreateBucket(authUsersBucketName)
	tx.UnsafeCreateBucket(authRolesBucketName)

	enabled := false
	_, vs := tx.UnsafeRange(authBucketName, enableFlagKey, nil, 0)
	if len(vs) == 1 {
		if bytes.Equal(vs[0], authEnabled) {
			enabled = true
		}
	}

	as := &authStore{
		be:             be,
		simpleTokens:   make(map[string]string),
		revision:       getRevision(tx),
		indexWaiter:    indexWaiter,
		enabled:        enabled,
		rangePermCache: make(map[string]*unifiedRangePermissions),
	}

	if enabled {
		as.enable()
	}

	if as.revision == 0 {
		as.commitRevision(tx)
	}

	tx.Unlock()
	be.ForceCommit()

	return as
}

func hasRootRole(u *authpb.User) bool {
	for _, r := range u.Roles {
		if r == rootRole {
			return true
		}
	}
	return false
}

func (as *authStore) commitRevision(tx backend.BatchTx) {
	as.revision++
	revBytes := make([]byte, revBytesLen)
	binary.BigEndian.PutUint64(revBytes, as.revision)
	tx.UnsafePut(authBucketName, revisionKey, revBytes)
}

func getRevision(tx backend.BatchTx) uint64 {
	_, vs := tx.UnsafeRange(authBucketName, []byte(revisionKey), nil, 0)
	if len(vs) != 1 {
		// this can happen in the initialization phase
		return 0
	}

	return binary.BigEndian.Uint64(vs[0])
}

func (as *authStore) Revision() uint64 {
	return as.revision
}

func (as *authStore) isValidSimpleToken(token string, ctx context.Context) bool {
	splitted := strings.Split(token, ".")
	if len(splitted) != 2 {
		return false
	}
	index, err := strconv.Atoi(splitted[1])
	if err != nil {
		return false
	}

	select {
	case <-as.indexWaiter(uint64(index)):
		return true
	case <-ctx.Done():
	}

	return false
}

func (as *authStore) AuthInfoFromCtx(ctx context.Context) (*AuthInfo, error) {
	md, ok := metadata.FromContext(ctx)
	if !ok {
		return nil, nil
	}

	ts, tok := md["token"]
	if !tok {
		return nil, nil
	}

	token := ts[0]
	if !as.isValidSimpleToken(token, ctx) {
		return nil, ErrInvalidAuthToken
	}

	authInfo, uok := as.AuthInfoFromToken(token)
	if !uok {
		plog.Warningf("invalid auth token: %s", token)
		return nil, ErrInvalidAuthToken
	}
	return authInfo, nil
}
