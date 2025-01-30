/*
Copyright 2016 The Kubernetes Authors.

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

package anonymous

import (
	"net/http"

	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

const (
	anonymousUser        = user.Anonymous
	unauthenticatedGroup = user.AllUnauthenticated
)

type Authenticator struct {
	allowedPaths map[string]bool
}

func (a *Authenticator) AuthenticateRequest(req *http.Request) (*authenticator.Response, bool, error) {
	if len(a.allowedPaths) > 0 && !a.allowedPaths[req.URL.Path] {
		return nil, false, nil
	}

	auds, _ := authenticator.AudiencesFrom(req.Context())
	return &authenticator.Response{
		User: &user.DefaultInfo{
			Name:   anonymousUser,
			Groups: []string{unauthenticatedGroup},
		},
		Audiences: auds,
	}, true, nil
}

// NewAuthenticator returns a new anonymous authenticator.
// When conditions is empty all requests are authenticated as anonymous.
// When conditions are non-empty only those requests that match the at-least one
// condition are authenticated as anonymous.
func NewAuthenticator(conditions []apiserver.AnonymousAuthCondition) authenticator.Request {
	allowedPaths := make(map[string]bool)
	for _, c := range conditions {
		allowedPaths[c.Path] = true
	}

	return &Authenticator{allowedPaths: allowedPaths}
}
