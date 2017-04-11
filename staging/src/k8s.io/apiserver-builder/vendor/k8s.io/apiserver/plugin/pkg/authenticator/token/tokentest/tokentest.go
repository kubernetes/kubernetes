/*
Copyright 2014 The Kubernetes Authors.

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

package tokentest

import "k8s.io/apiserver/pkg/authentication/user"

type TokenAuthenticator struct {
	Tokens map[string]*user.DefaultInfo
}

func New() *TokenAuthenticator {
	return &TokenAuthenticator{
		Tokens: make(map[string]*user.DefaultInfo),
	}
}
func (a *TokenAuthenticator) AuthenticateToken(value string) (user.Info, bool, error) {
	user, ok := a.Tokens[value]
	if !ok {
		return nil, false, nil
	}
	return user, true, nil
}
