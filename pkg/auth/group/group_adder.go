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

package group

import (
	"net/http"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
)

// GroupAdder adds groups to an authenticated user.Info
type GroupAdder struct {
	// Authenticator is delegated to make the authentication decision
	Authenticator authenticator.Request
	// Groups are additional groups to add to the user.Info from a successful authentication
	Groups []string
}

// NewGroupAdder wraps a request authenticator, and adds the specified groups to the returned user when authentication succeeds
func NewGroupAdder(auth authenticator.Request, groups []string) authenticator.Request {
	return &GroupAdder{auth, groups}
}

func (g *GroupAdder) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	u, ok, err := g.Authenticator.AuthenticateRequest(req)
	if err != nil || !ok {
		return nil, ok, err
	}
	return &user.DefaultInfo{
		Name:   u.GetName(),
		UID:    u.GetUID(),
		Groups: append(u.GetGroups(), g.Groups...),
		Extra:  u.GetExtra(),
	}, true, nil
}
