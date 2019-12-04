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
	"encoding/json"
	"net/http"
	"path"
	"strings"

	"go.etcd.io/etcd/etcdserver/api"
	"go.etcd.io/etcd/etcdserver/api/v2auth"
	"go.etcd.io/etcd/etcdserver/api/v2http/httptypes"

	"go.uber.org/zap"
)

type authHandler struct {
	lg                    *zap.Logger
	sec                   v2auth.Store
	cluster               api.Cluster
	clientCertAuthEnabled bool
}

func hasWriteRootAccess(lg *zap.Logger, sec v2auth.Store, r *http.Request, clientCertAuthEnabled bool) bool {
	if r.Method == "GET" || r.Method == "HEAD" {
		return true
	}
	return hasRootAccess(lg, sec, r, clientCertAuthEnabled)
}

func userFromBasicAuth(lg *zap.Logger, sec v2auth.Store, r *http.Request) *v2auth.User {
	username, password, ok := r.BasicAuth()
	if !ok {
		if lg != nil {
			lg.Warn("malformed basic auth encoding")
		} else {
			plog.Warningf("auth: malformed basic auth encoding")
		}
		return nil
	}
	user, err := sec.GetUser(username)
	if err != nil {
		return nil
	}

	ok = sec.CheckPassword(user, password)
	if !ok {
		if lg != nil {
			lg.Warn("incorrect password", zap.String("user-name", username))
		} else {
			plog.Warningf("auth: incorrect password for user: %s", username)
		}
		return nil
	}
	return &user
}

func userFromClientCertificate(lg *zap.Logger, sec v2auth.Store, r *http.Request) *v2auth.User {
	if r.TLS == nil {
		return nil
	}

	for _, chains := range r.TLS.VerifiedChains {
		for _, chain := range chains {
			if lg != nil {
				lg.Debug("found common name", zap.String("common-name", chain.Subject.CommonName))
			} else {
				plog.Debugf("auth: found common name %s.\n", chain.Subject.CommonName)
			}
			user, err := sec.GetUser(chain.Subject.CommonName)
			if err == nil {
				if lg != nil {
					lg.Debug(
						"authenticated a user via common name",
						zap.String("user-name", user.User),
						zap.String("common-name", chain.Subject.CommonName),
					)
				} else {
					plog.Debugf("auth: authenticated user %s by cert common name.", user.User)
				}
				return &user
			}
		}
	}
	return nil
}

func hasRootAccess(lg *zap.Logger, sec v2auth.Store, r *http.Request, clientCertAuthEnabled bool) bool {
	if sec == nil {
		// No store means no auth available, eg, tests.
		return true
	}
	if !sec.AuthEnabled() {
		return true
	}

	var rootUser *v2auth.User
	if r.Header.Get("Authorization") == "" && clientCertAuthEnabled {
		rootUser = userFromClientCertificate(lg, sec, r)
		if rootUser == nil {
			return false
		}
	} else {
		rootUser = userFromBasicAuth(lg, sec, r)
		if rootUser == nil {
			return false
		}
	}

	for _, role := range rootUser.Roles {
		if role == v2auth.RootRoleName {
			return true
		}
	}

	if lg != nil {
		lg.Warn(
			"a user does not have root role for resource",
			zap.String("root-user", rootUser.User),
			zap.String("root-role-name", v2auth.RootRoleName),
			zap.String("resource-path", r.URL.Path),
		)
	} else {
		plog.Warningf("auth: user %s does not have the %s role for resource %s.", rootUser.User, v2auth.RootRoleName, r.URL.Path)
	}
	return false
}

func hasKeyPrefixAccess(lg *zap.Logger, sec v2auth.Store, r *http.Request, key string, recursive, clientCertAuthEnabled bool) bool {
	if sec == nil {
		// No store means no auth available, eg, tests.
		return true
	}
	if !sec.AuthEnabled() {
		return true
	}

	var user *v2auth.User
	if r.Header.Get("Authorization") == "" {
		if clientCertAuthEnabled {
			user = userFromClientCertificate(lg, sec, r)
		}
		if user == nil {
			return hasGuestAccess(lg, sec, r, key)
		}
	} else {
		user = userFromBasicAuth(lg, sec, r)
		if user == nil {
			return false
		}
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

	if lg != nil {
		lg.Warn(
			"invalid access for user on key",
			zap.String("user-name", user.User),
			zap.String("key", key),
		)
	} else {
		plog.Warningf("auth: invalid access for user %s on key %s.", user.User, key)
	}
	return false
}

func hasGuestAccess(lg *zap.Logger, sec v2auth.Store, r *http.Request, key string) bool {
	writeAccess := r.Method != "GET" && r.Method != "HEAD"
	role, err := sec.GetRole(v2auth.GuestRoleName)
	if err != nil {
		return false
	}
	if role.HasKeyAccess(key, writeAccess) {
		return true
	}

	if lg != nil {
		lg.Warn(
			"invalid access for a guest role on key",
			zap.String("role-name", v2auth.GuestRoleName),
			zap.String("key", key),
		)
	} else {
		plog.Warningf("auth: invalid access for unauthenticated user on resource %s.", key)
	}
	return false
}

func writeNoAuth(lg *zap.Logger, w http.ResponseWriter, r *http.Request) {
	herr := httptypes.NewHTTPError(http.StatusUnauthorized, "Insufficient credentials")
	if err := herr.WriteTo(w); err != nil {
		if lg != nil {
			lg.Debug(
				"failed to write v2 HTTP error",
				zap.String("remote-addr", r.RemoteAddr),
				zap.Error(err),
			)
		} else {
			plog.Debugf("error writing HTTPError (%v) to %s", err, r.RemoteAddr)
		}
	}
}

func handleAuth(mux *http.ServeMux, sh *authHandler) {
	mux.HandleFunc(authPrefix+"/roles", authCapabilityHandler(sh.baseRoles))
	mux.HandleFunc(authPrefix+"/roles/", authCapabilityHandler(sh.handleRoles))
	mux.HandleFunc(authPrefix+"/users", authCapabilityHandler(sh.baseUsers))
	mux.HandleFunc(authPrefix+"/users/", authCapabilityHandler(sh.handleUsers))
	mux.HandleFunc(authPrefix+"/enable", authCapabilityHandler(sh.enableDisable))
}

func (sh *authHandler) baseRoles(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET") {
		return
	}
	if !hasRootAccess(sh.lg, sh.sec, r, sh.clientCertAuthEnabled) {
		writeNoAuth(sh.lg, w, r)
		return
	}

	w.Header().Set("X-Etcd-Cluster-ID", sh.cluster.ID().String())
	w.Header().Set("Content-Type", "application/json")

	roles, err := sh.sec.AllRoles()
	if err != nil {
		writeError(sh.lg, w, r, err)
		return
	}
	if roles == nil {
		roles = make([]string, 0)
	}

	err = r.ParseForm()
	if err != nil {
		writeError(sh.lg, w, r, err)
		return
	}

	var rolesCollections struct {
		Roles []v2auth.Role `json:"roles"`
	}
	for _, roleName := range roles {
		var role v2auth.Role
		role, err = sh.sec.GetRole(roleName)
		if err != nil {
			writeError(sh.lg, w, r, err)
			return
		}
		rolesCollections.Roles = append(rolesCollections.Roles, role)
	}
	err = json.NewEncoder(w).Encode(rolesCollections)

	if err != nil {
		if sh.lg != nil {
			sh.lg.Warn(
				"failed to encode base roles",
				zap.String("url", r.URL.String()),
				zap.Error(err),
			)
		} else {
			plog.Warningf("baseRoles error encoding on %s", r.URL)
		}
		writeError(sh.lg, w, r, err)
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
		writeError(sh.lg, w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Invalid path"))
		return
	}
	sh.forRole(w, r, pieces[2])
}

func (sh *authHandler) forRole(w http.ResponseWriter, r *http.Request, role string) {
	if !allowMethod(w, r.Method, "GET", "PUT", "DELETE") {
		return
	}
	if !hasRootAccess(sh.lg, sh.sec, r, sh.clientCertAuthEnabled) {
		writeNoAuth(sh.lg, w, r)
		return
	}
	w.Header().Set("X-Etcd-Cluster-ID", sh.cluster.ID().String())
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		data, err := sh.sec.GetRole(role)
		if err != nil {
			writeError(sh.lg, w, r, err)
			return
		}
		err = json.NewEncoder(w).Encode(data)
		if err != nil {
			if sh.lg != nil {
				sh.lg.Warn(
					"failed to encode a role",
					zap.String("url", r.URL.String()),
					zap.Error(err),
				)
			} else {
				plog.Warningf("forRole error encoding on %s", r.URL)
			}
			return
		}
		return

	case "PUT":
		var in v2auth.Role
		err := json.NewDecoder(r.Body).Decode(&in)
		if err != nil {
			writeError(sh.lg, w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Invalid JSON in request body."))
			return
		}
		if in.Role != role {
			writeError(sh.lg, w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Role JSON name does not match the name in the URL"))
			return
		}

		var out v2auth.Role

		// create
		if in.Grant.IsEmpty() && in.Revoke.IsEmpty() {
			err = sh.sec.CreateRole(in)
			if err != nil {
				writeError(sh.lg, w, r, err)
				return
			}
			w.WriteHeader(http.StatusCreated)
			out = in
		} else {
			if !in.Permissions.IsEmpty() {
				writeError(sh.lg, w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Role JSON contains both permissions and grant/revoke"))
				return
			}
			out, err = sh.sec.UpdateRole(in)
			if err != nil {
				writeError(sh.lg, w, r, err)
				return
			}
			w.WriteHeader(http.StatusOK)
		}

		err = json.NewEncoder(w).Encode(out)
		if err != nil {
			if sh.lg != nil {
				sh.lg.Warn(
					"failed to encode a role",
					zap.String("url", r.URL.String()),
					zap.Error(err),
				)
			} else {
				plog.Warningf("forRole error encoding on %s", r.URL)
			}
			return
		}
		return

	case "DELETE":
		err := sh.sec.DeleteRole(role)
		if err != nil {
			writeError(sh.lg, w, r, err)
			return
		}
	}
}

type userWithRoles struct {
	User  string        `json:"user"`
	Roles []v2auth.Role `json:"roles,omitempty"`
}

type usersCollections struct {
	Users []userWithRoles `json:"users"`
}

func (sh *authHandler) baseUsers(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET") {
		return
	}
	if !hasRootAccess(sh.lg, sh.sec, r, sh.clientCertAuthEnabled) {
		writeNoAuth(sh.lg, w, r)
		return
	}
	w.Header().Set("X-Etcd-Cluster-ID", sh.cluster.ID().String())
	w.Header().Set("Content-Type", "application/json")

	users, err := sh.sec.AllUsers()
	if err != nil {
		writeError(sh.lg, w, r, err)
		return
	}
	if users == nil {
		users = make([]string, 0)
	}

	err = r.ParseForm()
	if err != nil {
		writeError(sh.lg, w, r, err)
		return
	}

	ucs := usersCollections{}
	for _, userName := range users {
		var user v2auth.User
		user, err = sh.sec.GetUser(userName)
		if err != nil {
			writeError(sh.lg, w, r, err)
			return
		}

		uwr := userWithRoles{User: user.User}
		for _, roleName := range user.Roles {
			var role v2auth.Role
			role, err = sh.sec.GetRole(roleName)
			if err != nil {
				continue
			}
			uwr.Roles = append(uwr.Roles, role)
		}

		ucs.Users = append(ucs.Users, uwr)
	}
	err = json.NewEncoder(w).Encode(ucs)

	if err != nil {
		if sh.lg != nil {
			sh.lg.Warn(
				"failed to encode users",
				zap.String("url", r.URL.String()),
				zap.Error(err),
			)
		} else {
			plog.Warningf("baseUsers error encoding on %s", r.URL)
		}
		writeError(sh.lg, w, r, err)
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
		writeError(sh.lg, w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Invalid path"))
		return
	}
	sh.forUser(w, r, pieces[2])
}

func (sh *authHandler) forUser(w http.ResponseWriter, r *http.Request, user string) {
	if !allowMethod(w, r.Method, "GET", "PUT", "DELETE") {
		return
	}
	if !hasRootAccess(sh.lg, sh.sec, r, sh.clientCertAuthEnabled) {
		writeNoAuth(sh.lg, w, r)
		return
	}
	w.Header().Set("X-Etcd-Cluster-ID", sh.cluster.ID().String())
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		u, err := sh.sec.GetUser(user)
		if err != nil {
			writeError(sh.lg, w, r, err)
			return
		}

		err = r.ParseForm()
		if err != nil {
			writeError(sh.lg, w, r, err)
			return
		}

		uwr := userWithRoles{User: u.User}
		for _, roleName := range u.Roles {
			var role v2auth.Role
			role, err = sh.sec.GetRole(roleName)
			if err != nil {
				writeError(sh.lg, w, r, err)
				return
			}
			uwr.Roles = append(uwr.Roles, role)
		}
		err = json.NewEncoder(w).Encode(uwr)

		if err != nil {
			if sh.lg != nil {
				sh.lg.Warn(
					"failed to encode roles",
					zap.String("url", r.URL.String()),
					zap.Error(err),
				)
			} else {
				plog.Warningf("forUser error encoding on %s", r.URL)
			}
			return
		}
		return

	case "PUT":
		var u v2auth.User
		err := json.NewDecoder(r.Body).Decode(&u)
		if err != nil {
			writeError(sh.lg, w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Invalid JSON in request body."))
			return
		}
		if u.User != user {
			writeError(sh.lg, w, r, httptypes.NewHTTPError(http.StatusBadRequest, "User JSON name does not match the name in the URL"))
			return
		}

		var (
			out     v2auth.User
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
				writeError(sh.lg, w, r, err)
				return
			}
		} else {
			// update case
			if len(u.Roles) != 0 {
				writeError(sh.lg, w, r, httptypes.NewHTTPError(http.StatusBadRequest, "User JSON contains both roles and grant/revoke"))
				return
			}
			out, err = sh.sec.UpdateUser(u)
			if err != nil {
				writeError(sh.lg, w, r, err)
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
			if sh.lg != nil {
				sh.lg.Warn(
					"failed to encode a user",
					zap.String("url", r.URL.String()),
					zap.Error(err),
				)
			} else {
				plog.Warningf("forUser error encoding on %s", r.URL)
			}
			return
		}
		return

	case "DELETE":
		err := sh.sec.DeleteUser(user)
		if err != nil {
			writeError(sh.lg, w, r, err)
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
	if !hasWriteRootAccess(sh.lg, sh.sec, r, sh.clientCertAuthEnabled) {
		writeNoAuth(sh.lg, w, r)
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
			if sh.lg != nil {
				sh.lg.Warn(
					"failed to encode a auth state",
					zap.String("url", r.URL.String()),
					zap.Error(err),
				)
			} else {
				plog.Warningf("error encoding auth state on %s", r.URL)
			}
		}

	case "PUT":
		err := sh.sec.EnableAuth()
		if err != nil {
			writeError(sh.lg, w, r, err)
			return
		}

	case "DELETE":
		err := sh.sec.DisableAuth()
		if err != nil {
			writeError(sh.lg, w, r, err)
			return
		}
	}
}
