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

package accesstoken

import (
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	oauthclient "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/client"
)

// TokenAuthenticator validates tokens as unexpired OAuth access tokens
type TokenAuthenticator struct {
	client oauthclient.Interface
}

// NewTokenAuthenticator returns an OAuth TokenAuthenticator
func NewTokenAuthenticator(client oauthclient.Interface) *TokenAuthenticator {
	return &TokenAuthenticator{
		client: client,
	}
}

// AuthenticateToken implements authenticator.Token
func (a *TokenAuthenticator) AuthenticateToken(value string) (user.Info, bool, error) {
	token, err := a.client.OAuthAccessTokens().Get(value)
	if errors.IsNotFound(err) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	if token.CreationTimestamp.Time.Add(time.Duration(token.ExpiresIn) * time.Second).Before(time.Now()) {
		return nil, false, nil
	}
	return &user.DefaultInfo{
		Name: token.UserName,
		UID:  token.UserUID,
	}, true, nil
}
