/*
Copyright 2015 Google Inc. All rights reserved.

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

package gssproxy

import (
	"encoding/base64"
	"errors"
	"net"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/nalind/gss/pkg/gss/proxy"
)

type UserConversion interface {
	User(name proxy.SecCtx) (user.Info, bool, error)
}

type UserConversionFunc func(context proxy.SecCtx) (user.Info, bool, error)

func (f UserConversionFunc) User(context proxy.SecCtx) (user.Info, bool, error) {
	return f(context)
}

type GssProxyAuthenticator struct {
	proxyPath   string
	userConvert UserConversion
}

func NewGssProxy(proxyPath string, userConversion UserConversion) (authenticator.Token, error) {
	return &GssProxyAuthenticator{
		proxyPath:   proxyPath,
		userConvert: userConversion,
	}, nil
}

func (a *GssProxyAuthenticator) AuthenticateToken(b64token string) (user.Info, http.Header, bool, error) {
	challenge := http.Header{"WWW-Authenticate": {"Negotiate"}}
	var callCtx proxy.CallCtx
	var secCtx proxy.SecCtx
	if len(b64token) == 0 {
		return nil, challenge, false, errors.New("Requested Negotiate auth, but provided no auth data")
	}
	token, err := base64.StdEncoding.DecodeString(b64token)
	if err != nil {
		return nil, challenge, false, err
	}
	conn, err := net.Dial("unix", a.proxyPath)
	if err != nil {
		return nil, nil, false, errors.New("Error contacting gss-proxy: " + err.Error())
	}
	defer conn.Close()
	gccr, err := proxy.GetCallContext(&conn, &callCtx, nil)
	if err != nil {
		return nil, nil, false, err
	}
	if gccr.Status.MajorStatus != proxy.S_COMPLETE {
		return nil, nil, false, errors.New("Error returned from gss-proxy GetCallContext function: " + gccr.Status.MajorStatusString + "/" + gccr.Status.MinorStatusString)
	}
	ascr, err := proxy.AcceptSecContext(&conn, &callCtx, &secCtx, nil, token, nil, false, nil)
	if err != nil {
		return nil, challenge, false, err
	}
	if secCtx.NeedsRelease {
		defer proxy.ReleaseSecCtx(&conn, &callCtx, &secCtx)
	}
	if ascr.Status.MajorStatus == proxy.S_CONTINUE_NEEDED {
		// We can't handle multiple-round-trip requests right.
		return nil, nil, false, errors.New("Negotiate authentication canceled (incomplete): " + ascr.Status.MajorStatusString + "/" + ascr.Status.MinorStatusString)
	}
	if ascr.Status.MajorStatus != proxy.S_COMPLETE {
		// Authentication failed.
		return nil, nil, false, errors.New("Negotiate authentication failed: " + ascr.Status.MajorStatusString + "/" + ascr.Status.MinorStatusString)
	}
	headers := http.Header{}
	if ascr.OutputToken != nil {
		headers = http.Header{"WWW-Authenticate": {"Negotiate " + base64.StdEncoding.EncodeToString(*ascr.OutputToken)}}
	}
	user, ok, err := a.userConvert.User(secCtx)
	return user, headers, ok, err
}

var DefaultUserInfo UserConversionFunc = func(context proxy.SecCtx) (user.Info, bool, error) {
	return &user.DefaultInfo{Name: context.SrcName.DisplayName}, context.Open, nil
}
