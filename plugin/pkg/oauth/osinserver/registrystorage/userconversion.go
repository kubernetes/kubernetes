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
	"errors"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
)

type DefaultUserConversion struct{}

// NewUserConversion creates an object that can convert the user.Info object to and from
// an oauth access/authorize token object.
func NewUserConversion() UserConversion {
	return DefaultUserConversion{}
}

func (DefaultUserConversion) ConvertToAuthorizeToken(u interface{}, token *api.OAuthAuthorizeToken) error {
	info, ok := u.(user.Info)
	if !ok {
		return errors.New("did not receive user.Info")
	}
	token.UserName = info.GetName()
	if token.UserName == "" {
		return errors.New("user name is empty")
	}
	token.UserUID = info.GetUID()
	return nil
}

func (DefaultUserConversion) ConvertToAccessToken(u interface{}, token *api.OAuthAccessToken) error {
	info, ok := u.(user.Info)
	if !ok {
		return errors.New("did not receive user.Info")
	}
	token.UserName = info.GetName()
	if token.UserName == "" {
		return errors.New("user name is empty")
	}
	token.UserUID = info.GetUID()
	return nil
}

func (DefaultUserConversion) ConvertFromAuthorizeToken(token *api.OAuthAuthorizeToken) (interface{}, error) {
	if token.UserName == "" {
		return nil, errors.New("token has no user name stored")
	}
	return &user.DefaultInfo{
		Name: token.UserName,
		UID:  token.UserUID,
	}, nil
}

func (DefaultUserConversion) ConvertFromAccessToken(token *api.OAuthAccessToken) (interface{}, error) {
	if token.UserName == "" {
		return nil, errors.New("token has no user name stored")
	}
	return &user.DefaultInfo{
		Name: token.UserName,
		UID:  token.UserUID,
	}, nil
}
