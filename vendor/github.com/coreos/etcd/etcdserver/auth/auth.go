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

// Package auth implements etcd authentication.
package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path"
	"reflect"
	"sort"
	"strings"
	"time"

	etcderr "github.com/coreos/etcd/error"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/pkg/capnslog"

	"golang.org/x/crypto/bcrypt"
)

const (
	// StorePermsPrefix is the internal prefix of the storage layer dedicated to storing user data.
	StorePermsPrefix = "/2"

	// RootRoleName is the name of the ROOT role, with privileges to manage the cluster.
	RootRoleName = "root"

	// GuestRoleName is the name of the role that defines the privileges of an unauthenticated user.
	GuestRoleName = "guest"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "etcdserver/auth")
)

var rootRole = Role{
	Role: RootRoleName,
	Permissions: Permissions{
		KV: RWPermission{
			Read:  []string{"/*"},
			Write: []string{"/*"},
		},
	},
}

var guestRole = Role{
	Role: GuestRoleName,
	Permissions: Permissions{
		KV: RWPermission{
			Read:  []string{"/*"},
			Write: []string{"/*"},
		},
	},
}

type doer interface {
	Do(context.Context, etcdserverpb.Request) (etcdserver.Response, error)
}

type Store interface {
	AllUsers() ([]string, error)
	GetUser(name string) (User, error)
	CreateOrUpdateUser(user User) (out User, created bool, err error)
	CreateUser(user User) (User, error)
	DeleteUser(name string) error
	UpdateUser(user User) (User, error)
	AllRoles() ([]string, error)
	GetRole(name string) (Role, error)
	CreateRole(role Role) error
	DeleteRole(name string) error
	UpdateRole(role Role) (Role, error)
	AuthEnabled() bool
	EnableAuth() error
	DisableAuth() error
	PasswordStore
}

type PasswordStore interface {
	CheckPassword(user User, password string) bool
	HashPassword(password string) (string, error)
}

type store struct {
	server      doer
	timeout     time.Duration
	ensuredOnce bool

	PasswordStore
}

type User struct {
	User     string   `json:"user"`
	Password string   `json:"password,omitempty"`
	Roles    []string `json:"roles"`
	Grant    []string `json:"grant,omitempty"`
	Revoke   []string `json:"revoke,omitempty"`
}

type Role struct {
	Role        string       `json:"role"`
	Permissions Permissions  `json:"permissions"`
	Grant       *Permissions `json:"grant,omitempty"`
	Revoke      *Permissions `json:"revoke,omitempty"`
}

type Permissions struct {
	KV RWPermission `json:"kv"`
}

func (p *Permissions) IsEmpty() bool {
	return p == nil || (len(p.KV.Read) == 0 && len(p.KV.Write) == 0)
}

type RWPermission struct {
	Read  []string `json:"read"`
	Write []string `json:"write"`
}

type Error struct {
	Status int
	Errmsg string
}

func (ae Error) Error() string   { return ae.Errmsg }
func (ae Error) HTTPStatus() int { return ae.Status }

func authErr(hs int, s string, v ...interface{}) Error {
	return Error{Status: hs, Errmsg: fmt.Sprintf("auth: "+s, v...)}
}

func NewStore(server doer, timeout time.Duration) Store {
	s := &store{
		server:        server,
		timeout:       timeout,
		PasswordStore: passwordStore{},
	}
	return s
}

// passwordStore implements PasswordStore using bcrypt to hash user passwords
type passwordStore struct{}

func (_ passwordStore) CheckPassword(user User, password string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password))
	return err == nil
}

func (_ passwordStore) HashPassword(password string) (string, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	return string(hash), err
}

func (s *store) AllUsers() ([]string, error) {
	resp, err := s.requestResource("/users/", false, false)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeKeyNotFound {
				return []string{}, nil
			}
		}
		return nil, err
	}
	var nodes []string
	for _, n := range resp.Event.Node.Nodes {
		_, user := path.Split(n.Key)
		nodes = append(nodes, user)
	}
	sort.Strings(nodes)
	return nodes, nil
}

func (s *store) GetUser(name string) (User, error) { return s.getUser(name, false) }

// CreateOrUpdateUser should be only used for creating the new user or when you are not
// sure if it is a create or update. (When only password is passed in, we are not sure
// if it is a update or create)
func (s *store) CreateOrUpdateUser(user User) (out User, created bool, err error) {
	_, err = s.getUser(user.User, true)
	if err == nil {
		out, err = s.UpdateUser(user)
		return out, false, err
	}
	u, err := s.CreateUser(user)
	return u, true, err
}

func (s *store) CreateUser(user User) (User, error) {
	// Attach root role to root user.
	if user.User == "root" {
		user = attachRootRole(user)
	}
	u, err := s.createUserInternal(user)
	if err == nil {
		plog.Noticef("created user %s", user.User)
	}
	return u, err
}

func (s *store) createUserInternal(user User) (User, error) {
	if user.Password == "" {
		return user, authErr(http.StatusBadRequest, "Cannot create user %s with an empty password", user.User)
	}
	hash, err := s.HashPassword(user.Password)
	if err != nil {
		return user, err
	}
	user.Password = hash

	_, err = s.createResource("/users/"+user.User, user)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeNodeExist {
				return user, authErr(http.StatusConflict, "User %s already exists.", user.User)
			}
		}
	}
	return user, err
}

func (s *store) DeleteUser(name string) error {
	if s.AuthEnabled() && name == "root" {
		return authErr(http.StatusForbidden, "Cannot delete root user while auth is enabled.")
	}
	_, err := s.deleteResource("/users/" + name)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeKeyNotFound {
				return authErr(http.StatusNotFound, "User %s does not exist", name)
			}
		}
		return err
	}
	plog.Noticef("deleted user %s", name)
	return nil
}

func (s *store) UpdateUser(user User) (User, error) {
	old, err := s.getUser(user.User, true)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeKeyNotFound {
				return user, authErr(http.StatusNotFound, "User %s doesn't exist.", user.User)
			}
		}
		return old, err
	}

	newUser, err := old.merge(user, s.PasswordStore)
	if err != nil {
		return old, err
	}
	if reflect.DeepEqual(old, newUser) {
		return old, authErr(http.StatusBadRequest, "User not updated. Use grant/revoke/password to update the user.")
	}
	_, err = s.updateResource("/users/"+user.User, newUser)
	if err == nil {
		plog.Noticef("updated user %s", user.User)
	}
	return newUser, err
}

func (s *store) AllRoles() ([]string, error) {
	nodes := []string{RootRoleName}
	resp, err := s.requestResource("/roles/", false, false)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeKeyNotFound {
				return nodes, nil
			}
		}
		return nil, err
	}
	for _, n := range resp.Event.Node.Nodes {
		_, role := path.Split(n.Key)
		nodes = append(nodes, role)
	}
	sort.Strings(nodes)
	return nodes, nil
}

func (s *store) GetRole(name string) (Role, error) { return s.getRole(name, false) }

func (s *store) CreateRole(role Role) error {
	if role.Role == RootRoleName {
		return authErr(http.StatusForbidden, "Cannot modify role %s: is root role.", role.Role)
	}
	_, err := s.createResource("/roles/"+role.Role, role)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeNodeExist {
				return authErr(http.StatusConflict, "Role %s already exists.", role.Role)
			}
		}
	}
	if err == nil {
		plog.Noticef("created new role %s", role.Role)
	}
	return err
}

func (s *store) DeleteRole(name string) error {
	if name == RootRoleName {
		return authErr(http.StatusForbidden, "Cannot modify role %s: is root role.", name)
	}
	_, err := s.deleteResource("/roles/" + name)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeKeyNotFound {
				return authErr(http.StatusNotFound, "Role %s doesn't exist.", name)
			}
		}
	}
	if err == nil {
		plog.Noticef("deleted role %s", name)
	}
	return err
}

func (s *store) UpdateRole(role Role) (Role, error) {
	if role.Role == RootRoleName {
		return Role{}, authErr(http.StatusForbidden, "Cannot modify role %s: is root role.", role.Role)
	}
	old, err := s.getRole(role.Role, true)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeKeyNotFound {
				return role, authErr(http.StatusNotFound, "Role %s doesn't exist.", role.Role)
			}
		}
		return old, err
	}
	newRole, err := old.merge(role)
	if err != nil {
		return old, err
	}
	if reflect.DeepEqual(old, newRole) {
		return old, authErr(http.StatusBadRequest, "Role not updated. Use grant/revoke to update the role.")
	}
	_, err = s.updateResource("/roles/"+role.Role, newRole)
	if err == nil {
		plog.Noticef("updated role %s", role.Role)
	}
	return newRole, err
}

func (s *store) AuthEnabled() bool {
	return s.detectAuth()
}

func (s *store) EnableAuth() error {
	if s.AuthEnabled() {
		return authErr(http.StatusConflict, "already enabled")
	}

	if _, err := s.getUser("root", true); err != nil {
		return authErr(http.StatusConflict, "No root user available, please create one")
	}
	if _, err := s.getRole(GuestRoleName, true); err != nil {
		plog.Printf("no guest role access found, creating default")
		if err := s.CreateRole(guestRole); err != nil {
			plog.Errorf("error creating guest role. aborting auth enable.")
			return err
		}
	}

	if err := s.enableAuth(); err != nil {
		plog.Errorf("error enabling auth (%v)", err)
		return err
	}

	plog.Noticef("auth: enabled auth")
	return nil
}

func (s *store) DisableAuth() error {
	if !s.AuthEnabled() {
		return authErr(http.StatusConflict, "already disabled")
	}

	err := s.disableAuth()
	if err == nil {
		plog.Noticef("auth: disabled auth")
	} else {
		plog.Errorf("error disabling auth (%v)", err)
	}
	return err
}

// merge applies the properties of the passed-in User to the User on which it
// is called and returns a new User with these modifications applied. Think of
// all Users as immutable sets of data. Merge allows you to perform the set
// operations (desired grants and revokes) atomically
func (ou User) merge(nu User, s PasswordStore) (User, error) {
	var out User
	if ou.User != nu.User {
		return out, authErr(http.StatusConflict, "Merging user data with conflicting usernames: %s %s", ou.User, nu.User)
	}
	out.User = ou.User
	if nu.Password != "" {
		hash, err := s.HashPassword(nu.Password)
		if err != nil {
			return ou, err
		}
		out.Password = hash
	} else {
		out.Password = ou.Password
	}
	currentRoles := types.NewUnsafeSet(ou.Roles...)
	for _, g := range nu.Grant {
		if currentRoles.Contains(g) {
			plog.Noticef("granting duplicate role %s for user %s", g, nu.User)
			return User{}, authErr(http.StatusConflict, fmt.Sprintf("Granting duplicate role %s for user %s", g, nu.User))
		}
		currentRoles.Add(g)
	}
	for _, r := range nu.Revoke {
		if !currentRoles.Contains(r) {
			plog.Noticef("revoking ungranted role %s for user %s", r, nu.User)
			return User{}, authErr(http.StatusConflict, fmt.Sprintf("Revoking ungranted role %s for user %s", r, nu.User))
		}
		currentRoles.Remove(r)
	}
	out.Roles = currentRoles.Values()
	sort.Strings(out.Roles)
	return out, nil
}

// merge for a role works the same as User above -- atomic Role application to
// each of the substructures.
func (r Role) merge(n Role) (Role, error) {
	var out Role
	var err error
	if r.Role != n.Role {
		return out, authErr(http.StatusConflict, "Merging role with conflicting names: %s %s", r.Role, n.Role)
	}
	out.Role = r.Role
	out.Permissions, err = r.Permissions.Grant(n.Grant)
	if err != nil {
		return out, err
	}
	out.Permissions, err = out.Permissions.Revoke(n.Revoke)
	return out, err
}

func (r Role) HasKeyAccess(key string, write bool) bool {
	if r.Role == RootRoleName {
		return true
	}
	return r.Permissions.KV.HasAccess(key, write)
}

func (r Role) HasRecursiveAccess(key string, write bool) bool {
	if r.Role == RootRoleName {
		return true
	}
	return r.Permissions.KV.HasRecursiveAccess(key, write)
}

// Grant adds a set of permissions to the permission object on which it is called,
// returning a new permission object.
func (p Permissions) Grant(n *Permissions) (Permissions, error) {
	var out Permissions
	var err error
	if n == nil {
		return p, nil
	}
	out.KV, err = p.KV.Grant(n.KV)
	return out, err
}

// Revoke removes a set of permissions to the permission object on which it is called,
// returning a new permission object.
func (p Permissions) Revoke(n *Permissions) (Permissions, error) {
	var out Permissions
	var err error
	if n == nil {
		return p, nil
	}
	out.KV, err = p.KV.Revoke(n.KV)
	return out, err
}

// Grant adds a set of permissions to the permission object on which it is called,
// returning a new permission object.
func (rw RWPermission) Grant(n RWPermission) (RWPermission, error) {
	var out RWPermission
	currentRead := types.NewUnsafeSet(rw.Read...)
	for _, r := range n.Read {
		if currentRead.Contains(r) {
			return out, authErr(http.StatusConflict, "Granting duplicate read permission %s", r)
		}
		currentRead.Add(r)
	}
	currentWrite := types.NewUnsafeSet(rw.Write...)
	for _, w := range n.Write {
		if currentWrite.Contains(w) {
			return out, authErr(http.StatusConflict, "Granting duplicate write permission %s", w)
		}
		currentWrite.Add(w)
	}
	out.Read = currentRead.Values()
	out.Write = currentWrite.Values()
	sort.Strings(out.Read)
	sort.Strings(out.Write)
	return out, nil
}

// Revoke removes a set of permissions to the permission object on which it is called,
// returning a new permission object.
func (rw RWPermission) Revoke(n RWPermission) (RWPermission, error) {
	var out RWPermission
	currentRead := types.NewUnsafeSet(rw.Read...)
	for _, r := range n.Read {
		if !currentRead.Contains(r) {
			plog.Noticef("revoking ungranted read permission %s", r)
			continue
		}
		currentRead.Remove(r)
	}
	currentWrite := types.NewUnsafeSet(rw.Write...)
	for _, w := range n.Write {
		if !currentWrite.Contains(w) {
			plog.Noticef("revoking ungranted write permission %s", w)
			continue
		}
		currentWrite.Remove(w)
	}
	out.Read = currentRead.Values()
	out.Write = currentWrite.Values()
	sort.Strings(out.Read)
	sort.Strings(out.Write)
	return out, nil
}

func (rw RWPermission) HasAccess(key string, write bool) bool {
	var list []string
	if write {
		list = rw.Write
	} else {
		list = rw.Read
	}
	for _, pat := range list {
		match, err := simpleMatch(pat, key)
		if err == nil && match {
			return true
		}
	}
	return false
}

func (rw RWPermission) HasRecursiveAccess(key string, write bool) bool {
	list := rw.Read
	if write {
		list = rw.Write
	}
	for _, pat := range list {
		match, err := prefixMatch(pat, key)
		if err == nil && match {
			return true
		}
	}
	return false
}

func simpleMatch(pattern string, key string) (match bool, err error) {
	if pattern[len(pattern)-1] == '*' {
		return strings.HasPrefix(key, pattern[:len(pattern)-1]), nil
	}
	return key == pattern, nil
}

func prefixMatch(pattern string, key string) (match bool, err error) {
	if pattern[len(pattern)-1] != '*' {
		return false, nil
	}
	return strings.HasPrefix(key, pattern[:len(pattern)-1]), nil
}

func attachRootRole(u User) User {
	inRoles := false
	for _, r := range u.Roles {
		if r == RootRoleName {
			inRoles = true
			break
		}
	}
	if !inRoles {
		u.Roles = append(u.Roles, RootRoleName)
	}
	return u
}

func (s *store) getUser(name string, quorum bool) (User, error) {
	resp, err := s.requestResource("/users/"+name, false, quorum)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeKeyNotFound {
				return User{}, authErr(http.StatusNotFound, "User %s does not exist.", name)
			}
		}
		return User{}, err
	}
	var u User
	err = json.Unmarshal([]byte(*resp.Event.Node.Value), &u)
	if err != nil {
		return u, err
	}
	// Attach root role to root user.
	if u.User == "root" {
		u = attachRootRole(u)
	}
	return u, nil
}

func (s *store) getRole(name string, quorum bool) (Role, error) {
	if name == RootRoleName {
		return rootRole, nil
	}
	resp, err := s.requestResource("/roles/"+name, false, quorum)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeKeyNotFound {
				return Role{}, authErr(http.StatusNotFound, "Role %s does not exist.", name)
			}
		}
		return Role{}, err
	}
	var r Role
	err = json.Unmarshal([]byte(*resp.Event.Node.Value), &r)
	return r, err
}
