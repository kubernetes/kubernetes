/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package tokenconfig

import (
	"fmt"
	"os"

	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/util/yaml"
)

// TokenConfigAuthenticator read the tokens in yaml or json format for a local file
type TokenConfigAuthenticator struct {
	// a map of token to users
	tokens map[string]*user.DefaultInfo
}

// tokenEntry is a token entry in the config file
type tokenEntry struct {
	// the token for this user
	Token string `json:"token",yaml:"token"`
	// the name of the user
	Name string `json:"name",yaml:"name"`
	// the uid associated to the user
	UID string `json:"uid",yaml"uid"`
	// the groups the user is in
	Groups []string `json:"groups,omitempty", yaml:"groups,omitempty"`
}

// NewTokenConfig parses a json or yaml file and extracts the tokens from it
//  path:      the filename which contains the tokens, in either json or yaml format
func NewTokenConfig(path string) (*TokenConfigAuthenticator, error) {
	var entries []*tokenEntry

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	// step: parse and decode the file for us
	if err := yaml.NewYAMLToJSONDecoder(file).Decode(&entries); err != nil {
		return nil, err
	}

	// step: build a map and at the same time perform some validation of the input
	users := make(map[string]*user.DefaultInfo, 0)

	for i, x := range entries {
		if x.Token == "" {
			return nil, fmt.Errorf("entry on line %d should have a 'token' field", i)
		}
		if x.Name == "" {
			return nil, fmt.Errorf("entry for token: %s should have a 'name' field", x.Token)
		}
		if x.UID == "" {
			return nil, fmt.Errorf("entry for user: %s should have a 'uid' field", x.Name)
		}
		// should we error on duplicate uids / tokens perhaps?

		users[x.Token] = &user.DefaultInfo{
			Name:   x.Name,
			UID:    x.UID,
			Groups: x.Groups,
		}
	}

	return &TokenConfigAuthenticator{
		tokens: users,
	}, nil
}

func (a *TokenConfigAuthenticator) AuthenticateToken(value string) (user.Info, bool, error) {
	user, ok := a.tokens[value]
	if !ok {
		return nil, false, nil
	}
	return user, true, nil
}
