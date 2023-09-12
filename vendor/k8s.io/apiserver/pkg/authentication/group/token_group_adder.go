/*
Copyright 2017 The Kubernetes Authors.

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
	"context"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

// TokenGroupAdder adds groups to an authenticated user.Info
type TokenGroupAdder struct {
	// Authenticator is delegated to make the authentication decision
	Authenticator authenticator.Token
	// Groups are additional groups to add to the user.Info from a successful authentication
	Groups []string
}

// NewTokenGroupAdder wraps a token authenticator, and adds the specified groups to the returned user when authentication succeeds
func NewTokenGroupAdder(auth authenticator.Token, groups []string) authenticator.Token {
	return &TokenGroupAdder{auth, groups}
}

func (g *TokenGroupAdder) AuthenticateToken(ctx context.Context, token string) (*authenticator.Response, bool, error) {
	r, ok, err := g.Authenticator.AuthenticateToken(ctx, token)
	if err != nil || !ok {
		return nil, ok, err
	}

	newGroups := make([]string, 0, len(r.User.GetGroups())+len(g.Groups))
	newGroups = append(newGroups, r.User.GetGroups()...)
	newGroups = append(newGroups, g.Groups...)

	ret := *r // shallow copy
	ret.User = &user.DefaultInfo{
		Name:   r.User.GetName(),
		UID:    r.User.GetUID(),
		Groups: newGroups,
		Extra:  r.User.GetExtra(),
	}
	return &ret, true, nil
}
