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

package apiserver

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator/bearertoken"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/token/tokenfile"
)

// NewAuthenticatorFromTokenFile returns an authenticator.Request or an error
func NewAuthenticatorFromTokenFile(tokenAuthFile string) (authenticator.Request, error) {
	var authenticator authenticator.Request
	if len(tokenAuthFile) != 0 {
		tokenAuthenticator, err := tokenfile.NewCSV(tokenAuthFile)
		if err != nil {
			return nil, err
		}
		authenticator = bearertoken.New(tokenAuthenticator)
	}
	return authenticator, nil
}
