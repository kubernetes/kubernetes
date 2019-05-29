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
	"context"
	"encoding/binary"
	"errors"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/coreos/etcd/auth/authpb"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/mvcc/backend"

	"github.com/coreos/pkg/capnslog"
	"golang.org/x/crypto/bcrypt"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
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
	ErrInvalidAuthOpts      = errors.New("auth: invalid auth options")
	ErrInvalidAuthMgmt      = errors.New("auth: invalid auth management")

	// BcryptCost is the algorithm cost / strength for hashing auth passwords
	BcryptCost = bcrypt.DefaultCost
)

const (
	rootUser = "root"
	rootRole = "root"

	tokenTypeSimple = "simple"
	tokenTypeJWT    = "jwt"

	revBytesLen = 8
)

type AuthInfo struct {
	Username string
	Revision uint64
}

// AuthenticateParamIndex is used for a key of context in the parameters of Authenticate()
type AuthenticateParamIndex struct{}

// AuthenticateParamSimpleTokenPrefix is used for a key of context in the parameters of Authenticate()
type AuthenticateParamSimpleTokenPrefix struct{}

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

	// IsPutPermitted checks put permission of the user
	IsPutPermitted(authInfo *AuthInfo, key []byte) error

	// IsRangePermitted checks range permission of the user
	IsRangePermitted(authInfo *AuthInfo, key, rangeEnd []byte) error

	// IsDeleteRangePermitted checks delete-range permission of the user
	IsDeleteRangePermitted(authInfo *AuthInfo, key, rangeEnd []byte) error

	// IsAdminPermitted checks admin permission of the user
	IsAdminPermitted(authInfo *AuthInfo) error

	// GenTokenPrefix produces a random string in a case of simple token
	// in a case of JWT, it produces an empty string
	GenTokenPrefix() (string, error)

	// Revision gets current revision of authStore
	Revision() uint64

	// CheckPassword checks a given pair of username and password is correct
	CheckPassword(username, password string) (uint64, error)

	// Close does cleanup of AuthStore
	Close() error

	// AuthInfoFromCtx gets AuthInfo from gRPC's context
	AuthInfoFromCtx(ctx context.Context) (*AuthInfo, error)

	// AuthInfoFromTLS gets AuthInfo from TLS info of gRPC's context
	AuthInfoFromTLS(ctx context.Context) *AuthInfo

	// WithRoot generates and installs a token that can be used as a root credential
	WithRoot(ctx context.Context) context.Context

	// HasRole checks that user has role
	HasRole(user, role string) bool
}

type TokenProvider interface {
	info(ctx context.Context, token string, revision uint64) (*AuthInfo, bool)
	assign(ctx context.Context, username string, revision uint64) (string, error)
	enable()
	disable()

	invalidateUser(string)
	genTokenPrefix() (string, error)
}

type authStore struct {
	// atomic operations; need 64-bit align, or 32-bit tests will crash
	revision uint64

	be        backend.Backend
	enabled   bool
	enabledMu sync.RWMutex

	rangePermCache map[string]*unifiedRangePermissions // username -> unifiedRangePermissions

	tokenProvider TokenProvider
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
	as.tokenProvider.enable()

	as.rangePermCache = make(map[string]*unifiedRangePermissions)

	as.setRevision(getRevision(tx))

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
	as.tokenProvider.disable()

	plog.Noticef("Authentication disabled")
}

func (as *authStore) Close() error {
	as.enabledMu.Lock()
	defer as.enabledMu.Unlock()
	if !as.enabled {
		return nil
	}
	as.tokenProvider.disable()
	return nil
}

func (as *authStore) Authenticate(ctx context.Context, username, password string) (*pb.AuthenticateResponse, error) {
	if !as.isAuthEnabled() {
		return nil, ErrAuthNotEnabled
	}

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	user := getUser(tx, username)
	if user == nil {
		return nil, ErrAuthFailed
	}

	// Password checking is already performed in the API layer, so we don't need to check for now.
	// Staleness of password can be detected with OCC in the API layer, too.

	token, err := as.tokenProvider.assign(ctx, username, as.Revision())
	if err != nil {
		return nil, err
	}

	plog.Debugf("authorized %s, token is %s", username, token)
	return &pb.AuthenticateResponse{Token: token}, nil
}

func (as *authStore) CheckPassword(username, password string) (uint64, error) {
	if !as.isAuthEnabled() {
		return 0, ErrAuthNotEnabled
	}

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

	as.setRevision(getRevision(tx))

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
	if as.enabled && strings.Compare(r.Name, rootUser) == 0 {
		plog.Errorf("the user root must not be deleted")
		return nil, ErrInvalidAuthMgmt
	}

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
	as.tokenProvider.invalidateUser(r.Name)

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
	as.tokenProvider.invalidateUser(r.Name)

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
	sort.Strings(user.Roles)

	putUser(tx, user)

	as.invalidateCachedPerm(r.User)

	as.commitRevision(tx)

	plog.Noticef("granted role %s to user %s", r.Role, r.User)
	return &pb.AuthUserGrantRoleResponse{}, nil
}

func (as *authStore) UserGet(r *pb.AuthUserGetRequest) (*pb.AuthUserGetResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	user := getUser(tx, r.Name)
	tx.Unlock()

	if user == nil {
		return nil, ErrUserNotFound
	}

	var resp pb.AuthUserGetResponse
	resp.Roles = append(resp.Roles, user.Roles...)
	return &resp, nil
}

func (as *authStore) UserList(r *pb.AuthUserListRequest) (*pb.AuthUserListResponse, error) {
	tx := as.be.BatchTx()
	tx.Lock()
	users := getAllUsers(tx)
	tx.Unlock()

	resp := &pb.AuthUserListResponse{Users: make([]string, len(users))}
	for i := range users {
		resp.Users[i] = string(users[i].Name)
	}
	return resp, nil
}

func (as *authStore) UserRevokeRole(r *pb.AuthUserRevokeRoleRequest) (*pb.AuthUserRevokeRoleResponse, error) {
	if as.enabled && strings.Compare(r.Name, rootUser) == 0 && strings.Compare(r.Role, rootRole) == 0 {
		plog.Errorf("the role root must not be revoked from the user root")
		return nil, ErrInvalidAuthMgmt
	}

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
	roles := getAllRoles(tx)
	tx.Unlock()

	resp := &pb.AuthRoleListResponse{Roles: make([]string, len(roles))}
	for i := range roles {
		resp.Roles[i] = string(roles[i].Name)
	}
	return resp, nil
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
	if as.enabled && strings.Compare(r.Role, rootRole) == 0 {
		plog.Errorf("the role root must not be deleted")
		return nil, ErrInvalidAuthMgmt
	}

	tx := as.be.BatchTx()
	tx.Lock()
	defer tx.Unlock()

	role := getRole(tx, r.Role)
	if role == nil {
		return nil, ErrRoleNotFound
	}

	delRole(tx, r.Role)

	users := getAllUsers(tx)
	for _, user := range users {
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
			continue
		}

		putUser(tx, updatedUser)

		as.invalidateCachedPerm(string(user.Name))
	}

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

func (as *authStore) authInfoFromToken(ctx context.Context, token string) (*AuthInfo, bool) {
	return as.tokenProvider.info(ctx, token, as.Revision())
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

	if revision < as.Revision() {
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
	u := getUser(tx, authInfo.Username)
	tx.Unlock()

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

	users := make([]*authpb.User, len(vs))
	for i := range vs {
		user := &authpb.User{}
		err := user.Unmarshal(vs[i])
		if err != nil {
			plog.Panicf("failed to unmarshal user struct: %s", err)
		}
		users[i] = user
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

	roles := make([]*authpb.Role, len(vs))
	for i := range vs {
		role := &authpb.Role{}
		err := role.Unmarshal(vs[i])
		if err != nil {
			plog.Panicf("failed to unmarshal role struct: %s", err)
		}
		roles[i] = role
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

func NewAuthStore(be backend.Backend, tp TokenProvider) *authStore {
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
		revision:       getRevision(tx),
		enabled:        enabled,
		rangePermCache: make(map[string]*unifiedRangePermissions),
		tokenProvider:  tp,
	}

	if enabled {
		as.tokenProvider.enable()
	}

	if as.Revision() == 0 {
		as.commitRevision(tx)
	}

	tx.Unlock()
	be.ForceCommit()

	return as
}

func hasRootRole(u *authpb.User) bool {
	// u.Roles is sorted in UserGrantRole(), so we can use binary search.
	idx := sort.SearchStrings(u.Roles, rootRole)
	return idx != len(u.Roles) && u.Roles[idx] == rootRole
}

func (as *authStore) commitRevision(tx backend.BatchTx) {
	atomic.AddUint64(&as.revision, 1)
	revBytes := make([]byte, revBytesLen)
	binary.BigEndian.PutUint64(revBytes, as.Revision())
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

func (as *authStore) setRevision(rev uint64) {
	atomic.StoreUint64(&as.revision, rev)
}

func (as *authStore) Revision() uint64 {
	return atomic.LoadUint64(&as.revision)
}

func (as *authStore) AuthInfoFromTLS(ctx context.Context) *AuthInfo {
	peer, ok := peer.FromContext(ctx)
	if !ok || peer == nil || peer.AuthInfo == nil {
		return nil
	}

	tlsInfo := peer.AuthInfo.(credentials.TLSInfo)
	for _, chains := range tlsInfo.State.VerifiedChains {
		for _, chain := range chains {
			cn := chain.Subject.CommonName
			plog.Debugf("found common name %s", cn)

			ai := &AuthInfo{
				Username: cn,
				Revision: as.Revision(),
			}
			md, ok := metadata.FromIncomingContext(ctx)
			if !ok {
				return nil
			}

			// gRPC-gateway proxy request to etcd server includes Grpcgateway-Accept
			// header. The proxy uses etcd client server certificate. If the certificate
			// has a CommonName we should never use this for authentication.
			if gw := md["grpcgateway-accept"]; len(gw) > 0 {
				plog.Warningf("ignoring common name in gRPC-gateway proxy request %s", ai.Username)
				return nil
			}
			return ai
		}
	}

	return nil
}

func (as *authStore) AuthInfoFromCtx(ctx context.Context) (*AuthInfo, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, nil
	}

	//TODO(mitake|hexfusion) review unifying key names
	ts, ok := md["token"]
	if !ok {
		ts, ok = md["authorization"]
	}
	if !ok {
		return nil, nil
	}

	token := ts[0]
	authInfo, uok := as.authInfoFromToken(ctx, token)
	if !uok {
		plog.Warningf("invalid auth token: %s", token)
		return nil, ErrInvalidAuthToken
	}

	return authInfo, nil
}

func (as *authStore) GenTokenPrefix() (string, error) {
	return as.tokenProvider.genTokenPrefix()
}

func decomposeOpts(optstr string) (string, map[string]string, error) {
	opts := strings.Split(optstr, ",")
	tokenType := opts[0]

	typeSpecificOpts := make(map[string]string)
	for i := 1; i < len(opts); i++ {
		pair := strings.Split(opts[i], "=")

		if len(pair) != 2 {
			plog.Errorf("invalid token specific option: %s", optstr)
			return "", nil, ErrInvalidAuthOpts
		}

		if _, ok := typeSpecificOpts[pair[0]]; ok {
			plog.Errorf("invalid token specific option, duplicated parameters (%s): %s", pair[0], optstr)
			return "", nil, ErrInvalidAuthOpts
		}

		typeSpecificOpts[pair[0]] = pair[1]
	}

	return tokenType, typeSpecificOpts, nil

}

func NewTokenProvider(tokenOpts string, indexWaiter func(uint64) <-chan struct{}) (TokenProvider, error) {
	tokenType, typeSpecificOpts, err := decomposeOpts(tokenOpts)
	if err != nil {
		return nil, ErrInvalidAuthOpts
	}

	switch tokenType {
	case tokenTypeSimple:
		plog.Warningf("simple token is not cryptographically signed")
		return newTokenProviderSimple(indexWaiter), nil

	case tokenTypeJWT:
		return newTokenProviderJWT(typeSpecificOpts)

	case "":
		return newTokenProviderNop()
	default:
		plog.Errorf("unknown token type: %s", tokenType)
		return nil, ErrInvalidAuthOpts
	}
}

func (as *authStore) WithRoot(ctx context.Context) context.Context {
	if !as.isAuthEnabled() {
		return ctx
	}

	var ctxForAssign context.Context
	if ts, ok := as.tokenProvider.(*tokenSimple); ok && ts != nil {
		ctx1 := context.WithValue(ctx, AuthenticateParamIndex{}, uint64(0))
		prefix, err := ts.genTokenPrefix()
		if err != nil {
			plog.Errorf("failed to generate prefix of internally used token")
			return ctx
		}
		ctxForAssign = context.WithValue(ctx1, AuthenticateParamSimpleTokenPrefix{}, prefix)
	} else {
		ctxForAssign = ctx
	}

	token, err := as.tokenProvider.assign(ctxForAssign, "root", as.Revision())
	if err != nil {
		// this must not happen
		plog.Errorf("failed to assign token for lease revoking: %s", err)
		return ctx
	}

	mdMap := map[string]string{
		"token": token,
	}
	tokenMD := metadata.New(mdMap)

	// use "mdIncomingKey{}" since it's called from local etcdserver
	return metadata.NewIncomingContext(ctx, tokenMD)
}

func (as *authStore) HasRole(user, role string) bool {
	tx := as.be.BatchTx()
	tx.Lock()
	u := getUser(tx, user)
	tx.Unlock()

	if u == nil {
		plog.Warningf("tried to check user %s has role %s, but user %s doesn't exist", user, role, user)
		return false
	}

	for _, r := range u.Roles {
		if role == r {
			return true
		}
	}

	return false
}
