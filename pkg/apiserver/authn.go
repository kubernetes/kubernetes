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
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/gssapi/gssproxy"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/password/saslauthd"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/request/basicauth"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/request/negotiate"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/token/tokenfile"
	"strings"
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

// NewSaslauthd returns an authenticator.Request or an error
func NewAuthenticatorSaslauthd(arg string) (authenticator.Request, error) {
	var authenticator authenticator.Request
	var defaultRealm, socketPath, serviceName string
	params := strings.Split(arg, ",")
	if len(params) > 0 {
		defaultRealm = params[0]
		if len(params) > 1 {
			socketPath = params[1]
			if len(params) > 2 {
				serviceName = params[2]
			}
		}
	}
	sAuthenticator, err := saslauthd.NewSaslAuthd(defaultRealm, socketPath, serviceName)
	if err != nil {
		return nil, err
	}
	authenticator = basicauth.New(sAuthenticator)
	return authenticator, nil
}

// NewAuthenticatorGssProxy returns an authenticator.Request or an error
func NewAuthenticatorGssProxy(socketPath string) (authenticator.Request, error) {
	var authenticator authenticator.Request
	pAuthenticator, err := gssproxy.NewGssProxy(socketPath, gssproxy.DefaultUserInfo)
	if err != nil {
		return nil, err
	}
	authenticator = negotiate.New(pAuthenticator)
	return authenticator, nil
}
