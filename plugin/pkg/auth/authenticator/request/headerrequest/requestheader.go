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

package headerrequest

import (
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/util/sets"
	x509request "k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/x509"
)

var (
	// DefaultNameHeaders are the list of "normal" name headers to use for the authenticator
	DefaultNameHeaders = []string{"X-Remote-User", "Remote-User"}
)

type requestHeaderAuthRequestHandler struct {
	nameHeaders []string
}

func New(nameHeaders []string) authenticator.Request {
	return &requestHeaderAuthRequestHandler{nameHeaders: nameHeaders}
}

func NewSecure(clientCA string, proxyClientNames []string, nameHeaders []string) (authenticator.Request, error) {
	headerAuthenticator := &requestHeaderAuthRequestHandler{nameHeaders: nameHeaders}

	// Wrap with an x509 verifier
	caData, err := ioutil.ReadFile(clientCA)
	if err != nil {
		return nil, fmt.Errorf("Error reading %s: %v", clientCA, err)
	}
	opts := x509request.DefaultVerifyOptions()
	opts.Roots = x509.NewCertPool()
	if ok := opts.Roots.AppendCertsFromPEM(caData); !ok {
		return nil, fmt.Errorf("Error loading certs from %s: %v", clientCA, err)
	}

	return x509request.NewVerifier(opts, headerAuthenticator, sets.NewString(proxyClientNames...)), nil
}

func (a *requestHeaderAuthRequestHandler) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	name := headerValue(req.Header, a.nameHeaders)
	if len(name) == 0 {
		return nil, false, nil
	}

	return &user.DefaultInfo{Name: name}, true, nil
}

func headerValue(h http.Header, headerNames []string) string {
	for _, headerName := range headerNames {
		headerName = strings.TrimSpace(headerName)
		if len(headerName) == 0 {
			continue
		}
		headerValue := h.Get(headerName)
		if len(headerValue) > 0 {
			return headerValue
		}
	}
	return ""
}
