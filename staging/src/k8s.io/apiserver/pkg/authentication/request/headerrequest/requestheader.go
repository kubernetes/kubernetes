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
	"net/url"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	x509request "k8s.io/apiserver/pkg/authentication/request/x509"
	"k8s.io/apiserver/pkg/authentication/user"
	utilcert "k8s.io/client-go/util/cert"
)

type requestHeaderAuthRequestHandler struct {
	// nameHeaders are the headers to check (in order, case-insensitively) for an identity. The first header with a value wins.
	nameHeaders []string

	// groupHeaders are the headers to check (case-insensitively) for group membership.  All values of all headers will be added.
	groupHeaders []string

	// extraHeaderPrefixes are the head prefixes to check (case-insensitively) for filling in
	// the user.Info.Extra.  All values of all matching headers will be added.
	extraHeaderPrefixes []string
}

func New(nameHeaders []string, groupHeaders []string, extraHeaderPrefixes []string) (authenticator.Request, error) {
	trimmedNameHeaders, err := trimHeaders(nameHeaders...)
	if err != nil {
		return nil, err
	}
	trimmedGroupHeaders, err := trimHeaders(groupHeaders...)
	if err != nil {
		return nil, err
	}
	trimmedExtraHeaderPrefixes, err := trimHeaders(extraHeaderPrefixes...)
	if err != nil {
		return nil, err
	}

	return &requestHeaderAuthRequestHandler{
		nameHeaders:         trimmedNameHeaders,
		groupHeaders:        trimmedGroupHeaders,
		extraHeaderPrefixes: trimmedExtraHeaderPrefixes,
	}, nil
}

func trimHeaders(headerNames ...string) ([]string, error) {
	ret := []string{}
	for _, headerName := range headerNames {
		trimmedHeader := strings.TrimSpace(headerName)
		if len(trimmedHeader) == 0 {
			return nil, fmt.Errorf("empty header %q", headerName)
		}
		ret = append(ret, trimmedHeader)
	}

	return ret, nil
}

func NewSecure(clientCA string, proxyClientNames []string, nameHeaders []string, groupHeaders []string, extraHeaderPrefixes []string) (authenticator.Request, error) {
	headerAuthenticator, err := New(nameHeaders, groupHeaders, extraHeaderPrefixes)
	if err != nil {
		return nil, err
	}

	if len(clientCA) == 0 {
		return nil, fmt.Errorf("missing clientCA file")
	}

	// Wrap with an x509 verifier
	caData, err := ioutil.ReadFile(clientCA)
	if err != nil {
		return nil, fmt.Errorf("error reading %s: %v", clientCA, err)
	}
	opts := x509request.DefaultVerifyOptions()
	opts.Roots = x509.NewCertPool()
	certs, err := utilcert.ParseCertsPEM(caData)
	if err != nil {
		return nil, fmt.Errorf("error loading certs from  %s: %v", clientCA, err)
	}
	for _, cert := range certs {
		opts.Roots.AddCert(cert)
	}

	return x509request.NewVerifier(opts, headerAuthenticator, sets.NewString(proxyClientNames...)), nil
}

func (a *requestHeaderAuthRequestHandler) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	name := headerValue(req.Header, a.nameHeaders)
	if len(name) == 0 {
		return nil, false, nil
	}
	groups := allHeaderValues(req.Header, a.groupHeaders)
	extra := newExtra(req.Header, a.extraHeaderPrefixes)

	// clear headers used for authentication
	for _, headerName := range a.nameHeaders {
		req.Header.Del(headerName)
	}
	for _, headerName := range a.groupHeaders {
		req.Header.Del(headerName)
	}
	for k := range extra {
		for _, prefix := range a.extraHeaderPrefixes {
			req.Header.Del(prefix + k)
		}
	}

	return &user.DefaultInfo{
		Name:   name,
		Groups: groups,
		Extra:  extra,
	}, true, nil
}

func headerValue(h http.Header, headerNames []string) string {
	for _, headerName := range headerNames {
		headerValue := h.Get(headerName)
		if len(headerValue) > 0 {
			return headerValue
		}
	}
	return ""
}

func allHeaderValues(h http.Header, headerNames []string) []string {
	ret := []string{}
	for _, headerName := range headerNames {
		headerKey := http.CanonicalHeaderKey(headerName)
		values, ok := h[headerKey]
		if !ok {
			continue
		}

		for _, headerValue := range values {
			if len(headerValue) > 0 {
				ret = append(ret, headerValue)
			}
		}
	}
	return ret
}

func unescapeExtraKey(encodedKey string) string {
	key, err := url.PathUnescape(encodedKey) // Decode %-encoded bytes.
	if err != nil {
		return encodedKey // Always record extra strings, even if malformed/unencoded.
	}
	return key
}

func newExtra(h http.Header, headerPrefixes []string) map[string][]string {
	ret := map[string][]string{}

	// we have to iterate over prefixes first in order to have proper ordering inside the value slices
	for _, prefix := range headerPrefixes {
		for headerName, vv := range h {
			if !strings.HasPrefix(strings.ToLower(headerName), strings.ToLower(prefix)) {
				continue
			}

			extraKey := unescapeExtraKey(strings.ToLower(headerName[len(prefix):]))
			ret[extraKey] = append(ret[extraKey], vv...)
		}
	}

	return ret
}
