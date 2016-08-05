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

package v2http

import (
	"encoding/json"
	"net/http"
	"path"
	"strings"

	"github.com/coreos/etcd/etcdserver/api"
	"github.com/coreos/etcd/etcdserver/api/v2http/httptypes"
	"github.com/coreos/etcd/etcdserver/auth"
)

type authHandler struct {
	sec     auth.Store
	cluster api.Cluster
}

func hasWriteRootAccess(sec auth.Store, r *http.Request) bool {
	if r.Method == "GET" || r.Method == "HEAD" {
		return true
	}
	return hasRootAccess(sec, r)
}

func hasRootAccess(sec auth.Store, r *http.Request) bool {
	if sec == nil {
		// No store means no auth available, eg, tests.
		return true
	}
	if !sec.AuthEnabled() {
		return true
	}
	username, password, ok := r.BasicAuth()
	if !ok {
		return false
	}
	rootUser, err := sec.GetUser(username)
	if err != nil {
		return false
	}

	ok = sec.CheckPassword(rootUser, password)
	if !ok {
		plog.Warningf("auth: wrong password for user %s", username)
		return false
	}
	for _, role := range rootUser.Roles {
		if role == auth.RootRoleName {
			return true
		}
	}
	plog.Warningf("auth: user %s does not have the %s role for resource %s.", username, auth.RootRoleName, r.URL.Path)
	return false
}

func hasKeyPrefixAccess(sec auth.Store, r *http.Request, key string, recursive bool) bool {
	if sec == nil {
		// No store means no auth available, eg, tests.
		return true
	}
	if !sec.AuthEnabled() {
		return true
	}
	if r.Header.Get("Authorization") == "" {
		plog.Warningf("auth: no authorization provided, checking guest access")
		return hasGuestAccess(sec, r, key)
	}
	username, password, ok := r.BasicAuth()
	if !ok {
		plog.Warningf("auth: malformed basic auth encoding")
		return false
	}
	user, err := sec.GetUser(username)
	if err != nil {
		plog.Warningf("auth: no such user: %s.", username)
		return false
	}
	authAsUser := sec.CheckPassword(user, password)
	if !authAsUser {
		plog.Warningf("auth: incorrect password for user: %s.", username)
		return false
	}
	writeAccess := r.Method != "GET" && r.Method != "HEAD"
	for _, roleName := range user.Roles {
		role, err := sec.GetRole(roleName)
		if err != nil {
			continue
		}
		if recursive {
			if role.HasRecursiveAccess(key, writeAccess) {
				return true
			}
		} else if role.HasKeyAccess(key, writeAccess) {
			return true
		}
	}
	plog.Warningf("auth: invalid access for user %s on key %s.", username, key)
	return false
}

func hasGuestAccess(sec auth.Store, r *http.Request, key string) bool {
	writeAccess := r.Method != "GET" && r.Method != "HEAD"
	role, err := sec.GetRole(auth.GuestRoleName)
	if err != nil {
		return false
	}
	if role.HasKeyAccess(key, writeAccess) {
		return true
	}
	plog.Warningf("auth: invalid access for unauthenticated user on resource %s.", key)
	return false
}

func writeNoAuth(w http.ResponseWriter, r *http.Request) {
	herr := httptypes.NewHTTPError(http.StatusUnauthorized, "Insufficient credentials")
	if err := herr.WriteTo(w); err != nil {
		plog.Debugf("error writing HTTPError (%v) to %s", err, r.RemoteAddr)
	}
}

func handleAuth(mux *http.ServeMux, sh *authHandler) {
	mux.HandleFunc(authPrefix+"/roles", capabilityHandler(authCapability, sh.baseRoles))
	mux.HandleFunc(authPrefix+"/roles/", capabilityHandler(authCapability, sh.handleRoles))
	mux.HandleFunc(authPrefix+"/users", capabilityHandler(authCapability, sh.baseUsers))
	mux.HandleFunc(authPrefix+"/users/", capabilityHandler(authCapability, sh.handleUsers))
	mux.HandleFunc(authPrefix+"/enable", capabilityHandler(authCapability, sh.enableDisable))
}

func (sh *authHandler) baseRoles(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET") {
		return
	}
	if !hasRootAccess(sh.sec, r) {
		writeNoAuth(w, r)
		return
	}

	w.Header().Set("X-Etcd-Cluster-ID", sh.cluster.ID().String())
	w.Header().Set("Content-Type", "application/json")

	roles, err := sh.sec.AllRoles()
	if err != nil {
		writeError(w, r, err)
		return
	}
	if roles == nil {
		roles = make([]string, 0)
	}

	err = r.ParseForm()
	if err != nil {
		writeError(w, r, err)
		return
	}

	var rolesCollections struct {
		Roles []auth.Role `json:"roles"`
	}
	for _, roleName := range roles {
		var role auth.Role
		role, err = sh.sec.GetRole(roleName)
		if err != nil {
			writeError(w, r, err)
			return
		}
		rolesCollections.Roles = append(rolesCollections.Roles, role)
	}
	err = json.NewEncoder(w).Encode(rolesCollections)

	if err != nil {
		plog.Warningf("baseRoles error encoding on %s", r.URL)
		writeError(w, r, err)
		return
	}
}

func (sh *authHandler) handleRoles(w http.ResponseWriter, r *http.Request) {
	subpath := path.Clean(r.URL.Path[len(authPrefix):])
	// Split "/roles/rolename/command".
	// First item is an empty string, second is "roles"
	pieces := strings.Split(subpath, "/")
	if len(pieces) == 2 {
		sh.baseRoles(w, r)
		return
	}
	if len(pieces) != 3 {
		writeError(w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Invalid path"))
		return
	}
	sh.forRole(w, r, pieces[2])
}

func (sh *authHandler) forRole(w http.ResponseWriter, r *http.Request, role string) {
	if !allowMethod(w, r.Method, "GET", "PUT", "DELETE") {
		return
	}
	if !hasRootAccess(sh.sec, r) {
		writeNoAuth(w, r)
		return
	}
	w.Header().Set("X-Etcd-Cluster-ID", sh.cluster.ID().String())
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		data, err := sh.sec.GetRole(role)
		if err != nil {
			writeError(w, r, err)
			return
		}
		err = json.NewEncoder(w).Encode(data)
		if err != nil {
			plog.Warningf("forRole error encoding on %s", r.URL)
			return
		}
		return
	case "PUT":
		var in auth.Role
		err := json.NewDecoder(r.Body).Decode(&in)
		if err != nil {
			writeError(w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Invalid JSON in request body."))
			return
		}
		if in.Role != role {
			writeError(w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Role JSON name does not match the name in the URL"))
			return
		}

		var out auth.Role

		// create
		if in.Grant.IsEmpty() && in.Revoke.IsEmpty() {
			err = sh.sec.CreateRole(in)
			if err != nil {
				writeError(w, r, err)
				return
			}
			w.WriteHeader(http.StatusCreated)
			out = in
		} else {
			if !in.Permissions.IsEmpty() {
				writeError(w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Role JSON contains both permissions and grant/revoke"))
				return
			}
			out, err = sh.sec.UpdateRole(in)
			if err != nil {
				writeError(w, r, err)
				return
			}
			w.WriteHeader(http.StatusOK)
		}

		err = json.NewEncoder(w).Encode(out)
		if err != nil {
			plog.Warningf("forRole error encoding on %s", r.URL)
			return
		}
		return
	case "DELETE":
		err := sh.sec.DeleteRole(role)
		if err != nil {
			writeError(w, r, err)
			return
		}
	}
}

type userWithRoles struct {
	User  string      `json:"user"`
	Roles []auth.Role `json:"roles,omitempty"`
}

func (sh *authHandler) baseUsers(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET") {
		return
	}
	if !hasRootAccess(sh.sec, r) {
		writeNoAuth(w, r)
		return
	}
	w.Header().Set("X-Etcd-Cluster-ID", sh.cluster.ID().String())
	w.Header().Set("Content-Type", "application/json")

	users, err := sh.sec.AllUsers()
	if err != nil {
		writeError(w, r, err)
		return
	}
	if users == nil {
		users = make([]string, 0)
	}

	err = r.ParseForm()
	if err != nil {
		writeError(w, r, err)
		return
	}

	var usersCollections struct {
		Users []userWithRoles `json:"users"`
	}
	for _, userName := range users {
		var user auth.User
		user, err = sh.sec.GetUser(userName)
		if err != nil {
			writeError(w, r, err)
			return
		}

		uwr := userWithRoles{User: user.User}
		for _, roleName := range user.Roles {
			var role auth.Role
			role, err = sh.sec.GetRole(roleName)
			if err != nil {
				writeError(w, r, err)
				return
			}
			uwr.Roles = append(uwr.Roles, role)
		}

		usersCollections.Users = append(usersCollections.Users, uwr)
	}
	err = json.NewEncoder(w).Encode(usersCollections)

	if err != nil {
		plog.Warningf("baseUsers error encoding on %s", r.URL)
		writeError(w, r, err)
		return
	}
}

func (sh *authHandler) handleUsers(w http.ResponseWriter, r *http.Request) {
	subpath := path.Clean(r.URL.Path[len(authPrefix):])
	// Split "/users/username".
	// First item is an empty string, second is "users"
	pieces := strings.Split(subpath, "/")
	if len(pieces) == 2 {
		sh.baseUsers(w, r)
		return
	}
	if len(pieces) != 3 {
		writeError(w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Invalid path"))
		return
	}
	sh.forUser(w, r, pieces[2])
}

func (sh *authHandler) forUser(w http.ResponseWriter, r *http.Request, user string) {
	if !allowMethod(w, r.Method, "GET", "PUT", "DELETE") {
		return
	}
	if !hasRootAccess(sh.sec, r) {
		writeNoAuth(w, r)
		return
	}
	w.Header().Set("X-Etcd-Cluster-ID", sh.cluster.ID().String())
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		u, err := sh.sec.GetUser(user)
		if err != nil {
			writeError(w, r, err)
			return
		}

		err = r.ParseForm()
		if err != nil {
			writeError(w, r, err)
			return
		}

		uwr := userWithRoles{User: u.User}
		for _, roleName := range u.Roles {
			var role auth.Role
			role, err = sh.sec.GetRole(roleName)
			if err != nil {
				writeError(w, r, err)
				return
			}
			uwr.Roles = append(uwr.Roles, role)
		}
		err = json.NewEncoder(w).Encode(uwr)

		if err != nil {
			plog.Warningf("forUser error encoding on %s", r.URL)
			return
		}
		return
	case "PUT":
		var u auth.User
		err := json.NewDecoder(r.Body).Decode(&u)
		if err != nil {
			writeError(w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Invalid JSON in request body."))
			return
		}
		if u.User != user {
			writeError(w, r, httptypes.NewHTTPError(http.StatusBadRequest, "User JSON name does not match the name in the URL"))
			return
		}

		var (
			out     auth.User
			created bool
		)

		if len(u.Grant) == 0 && len(u.Revoke) == 0 {
			// create or update
			if len(u.Roles) != 0 {
				out, err = sh.sec.CreateUser(u)
			} else {
				// if user passes in both password and roles, we are unsure about his/her
				// intention.
				out, created, err = sh.sec.CreateOrUpdateUser(u)
			}

			if err != nil {
				writeError(w, r, err)
				return
			}
		} else {
			// update case
			if len(u.Roles) != 0 {
				writeError(w, r, httptypes.NewHTTPError(http.StatusBadRequest, "User JSON contains both roles and grant/revoke"))
				return
			}
			out, err = sh.sec.UpdateUser(u)
			if err != nil {
				writeError(w, r, err)
				return
			}
		}

		if created {
			w.WriteHeader(http.StatusCreated)
		} else {
			w.WriteHeader(http.StatusOK)
		}

		out.Password = ""

		err = json.NewEncoder(w).Encode(out)
		if err != nil {
			plog.Warningf("forUser error encoding on %s", r.URL)
			return
		}
		return
	case "DELETE":
		err := sh.sec.DeleteUser(user)
		if err != nil {
			writeError(w, r, err)
			return
		}
	}
}

type enabled struct {
	Enabled bool `json:"enabled"`
}

func (sh *authHandler) enableDisable(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET", "PUT", "DELETE") {
		return
	}
	if !hasWriteRootAccess(sh.sec, r) {
		writeNoAuth(w, r)
		return
	}
	w.Header().Set("X-Etcd-Cluster-ID", sh.cluster.ID().String())
	w.Header().Set("Content-Type", "application/json")
	isEnabled := sh.sec.AuthEnabled()
	switch r.Method {
	case "GET":
		jsonDict := enabled{isEnabled}
		err := json.NewEncoder(w).Encode(jsonDict)
		if err != nil {
			plog.Warningf("error encoding auth state on %s", r.URL)
		}
	case "PUT":
		err := sh.sec.EnableAuth()
		if err != nil {
			writeError(w, r, err)
			return
		}
	case "DELETE":
		err := sh.sec.DisableAuth()
		if err != nil {
			writeError(w, r, err)
			return
		}
	}
}
