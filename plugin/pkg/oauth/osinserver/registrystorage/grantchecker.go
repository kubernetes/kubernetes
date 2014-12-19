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

package registrystorage

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth"
	oauthclient "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/client"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/scope"
)

type ClientAuthorizationGrantChecker struct {
	client oauthclient.Interface
}

func NewClientAuthorizationGrantChecker(client oauthclient.Interface) *ClientAuthorizationGrantChecker {
	return &ClientAuthorizationGrantChecker{client}
}

func (c *ClientAuthorizationGrantChecker) HasAuthorizedClient(user user.Info, grant oauth.Grant) (bool, error) {
	id := c.client.OAuthClientAuthorizations().Name(user.GetName(), grant.GetClient().GetId())
	authorization, err := c.client.OAuthClientAuthorizations().Get(id)
	if errors.IsNotFound(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if len(authorization.UserUID) != 0 && authorization.UserUID != user.GetUID() {
		return false, fmt.Errorf("user %s UID %s does not match stored client authorization value for UID %s", user.GetName(), user.GetUID(), authorization.UserUID)
	}
	// TODO: improve this to allow the scope implementation to determine overlap
	if !scope.Covers(authorization.Scopes, grant.GetScopes()) {
		return false, nil
	}
	return true, nil
}
