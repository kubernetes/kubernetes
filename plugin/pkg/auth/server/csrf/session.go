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

package csrf

import (
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/session"

	"code.google.com/p/go-uuid/uuid"
)

const CSRFKey = "csrf"

type sessionCsrf struct {
	store session.Store
	name  string
}

// NewSessionCSRF stores CSRF tokens in a session with the given name.
// Empty CSRF tokens or tokens that do not match the value in the session are rejected.
func NewSessionCSRF(store session.Store, name string) CSRF {
	return &sessionCsrf{
		store: store,
		name:  name,
	}
}

// Generate implements the CSRF interface
func (c *sessionCsrf) Generate(w http.ResponseWriter, req *http.Request) (string, error) {
	session, err := c.store.Get(req, c.name)
	if err != nil {
		return "", err
	}

	values := session.Values()
	csrfString, ok := values[CSRFKey].(string)
	if ok && csrfString != "" {
		return csrfString, nil
	}

	csrfString = uuid.NewUUID().String()
	values[CSRFKey] = csrfString

	// TODO: defer save until response is written?
	if err = c.store.Save(w, req); err != nil {
		return "", err
	}

	return csrfString, nil
}

// Check implements the CSRF interface
func (c *sessionCsrf) Check(req *http.Request, value string) (bool, error) {
	if len(value) == 0 {
		return false, nil
	}

	session, err := c.store.Get(req, c.name)
	if err != nil {
		return false, err
	}

	values := session.Values()
	csrfString, ok := values[CSRFKey].(string)
	if ok && csrfString == value {
		return true, nil
	}

	return false, nil
}
