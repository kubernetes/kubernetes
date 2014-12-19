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
	"net/http"

	"github.com/gorilla/context"
	"github.com/gorilla/sessions"
)

type store struct {
	store sessions.Store
}

func NewStore(secrets ...string) Store {
	values := [][]byte{}
	for _, secret := range secrets {
		values = append(values, []byte(secret))
	}
	cookie := sessions.NewCookieStore(values...)
	// TODO: set other options (timeout, etc)
	return store{cookie}
}

func (s store) Get(req *http.Request, name string) (Session, error) {
	session, err := s.store.Get(req, name)
	return sessionWrapper{session}, err
}

func (s store) Save(w http.ResponseWriter, req *http.Request) error {
	return sessions.Save(req, w)
}

func (s store) Wrap(h http.Handler) http.Handler {
	return context.ClearHandler(h)
}

type sessionWrapper struct {
	session *sessions.Session
}

func (s sessionWrapper) Values() map[interface{}]interface{} {
	if s.session == nil {
		return map[interface{}]interface{}{}
	}
	return s.session.Values
}
