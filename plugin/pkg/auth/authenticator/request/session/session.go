/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package session

import (
	"errors"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/session"
)

// UserConversion defined an interface to extract user information from session values
type UserConversion interface {
	GetUserInfo(map[interface{}]interface{}) (user.Info, bool, error)
}

// SessionAuthenticator implements authenticator.Request using a session store
type SessionAuthenticator struct {
	store session.Store
	name  string
	user  UserConversion
}

// New returns a new SessionAuthenticator using the given store and session name
func New(store session.Store, name string, user UserConversion) *SessionAuthenticator {
	return &SessionAuthenticator{
		store: store,
		name:  name,
		user:  user,
	}
}

// AuthenticateRequest implements authenticator.Request to get user info out of a session for the request
func (a *SessionAuthenticator) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	session, err := a.store.Get(req, a.name)
	if err != nil {
		return nil, false, err
	}
	return a.user.GetUserInfo(session.Values())
}

// AuthenticationSucceeded implements handlers.AuthenticationSuccessHandler to save the user info in the session
func (a *SessionAuthenticator) AuthenticationSucceeded(user user.Info, state string, w http.ResponseWriter, req *http.Request) (bool, error) {
	session, err := a.store.Get(req, a.name)
	if err != nil {
		return false, err
	}
	values := session.Values()
	values[UserNameKey] = user.GetName()
	values[UserUIDKey] = user.GetUID()
	return false, a.store.Save(w, req)
}

const UserNameKey = "user.name"
const UserUIDKey = "user.uid"

// defaultUserConversion implements UserConversion to extract user name and uid from the session
type defaultUserConversion struct{}

// NewDefaultUserConversion returns a new UserConversion
func NewDefaultUserConversion() UserConversion {
	return defaultUserConversion{}
}

// GetUserInfo returns a user.Info object built from the UserNameKey and UserUIDKey values in the session
func (defaultUserConversion) GetUserInfo(values map[interface{}]interface{}) (user.Info, bool, error) {
	nameObj, nameOk := values[UserNameKey]
	uidObj, uidOk := values[UserUIDKey]
	if !nameOk || !uidOk {
		return nil, false, nil
	}

	name, nameOk := nameObj.(string)
	uid, uidOk := uidObj.(string)
	if !nameOk || !uidOk {
		return nil, false, errors.New("user.name or user.uid on session is not a string")
	}
	if name == "" || uid == "" {
		return nil, false, nil
	}

	return &user.DefaultInfo{
		Name: name,
		UID:  uid,
	}, true, nil
}
