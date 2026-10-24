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
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	x509request "k8s.io/apiserver/pkg/authentication/request/x509"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
)

// StringSliceProvider is a way to get a string slice value.  It is heavily used for authentication headers among other places.
type StringSliceProvider interface {
	// Value returns the current string slice.  Callers should never mutate the returned value.
	Value() []string
}

// StringSliceProviderFunc is a function that matches the StringSliceProvider interface
type StringSliceProviderFunc func() []string

// Value returns the current string slice.  Callers should never mutate the returned value.
func (d StringSliceProviderFunc) Value() []string {
	return d()
}

// StaticStringSlice a StringSliceProvider that returns a fixed value
type StaticStringSlice []string

// Value returns the current string slice.  Callers should never mutate the returned value.
func (s StaticStringSlice) Value() []string {
	return s
}

type requestHeaderAuthRequestHandler struct {
	// nameHeaders are the headers to check (in order, case-insensitively) for an identity. The first header with a value wins.
	nameHeaders StringSliceProvider

	// nameHeaders are the headers to check (in order, case-insensitively) for an identity UID. The first header with a value wins.
	uidHeaders StringSliceProvider

	// groupHeaders are the headers to check (case-insensitively) for group membership.  All values of all headers will be added.
	groupHeaders StringSliceProvider

	// extraHeaderPrefixes are the head prefixes to check (case-insensitively) for filling in
	// the user.Info.Extra.  All values of all matching headers will be added.
	extraHeaderPrefixes StringSliceProvider
}

func New(nameHeaders, uidHeaders, groupHeaders, extraHeaderPrefixes []string) (authenticator.Request, error) {
	trimmedNameHeaders, err := trimHeaders(nameHeaders...)
	if err != nil {
		return nil, err
	}
	trimmedUIDHeaders, err := trimHeaders(uidHeaders...)
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

	return NewDynamic(
		StaticStringSlice(trimmedNameHeaders),
		StaticStringSlice(trimmedUIDHeaders),
		StaticStringSlice(trimmedGroupHeaders),
		StaticStringSlice(trimmedExtraHeaderPrefixes),
	), nil
}

func NewDynamic(nameHeaders, uidHeaders, groupHeaders, extraHeaderPrefixes StringSliceProvider) authenticator.Request {
	return &requestHeaderAuthRequestHandler{
		nameHeaders:         nameHeaders,
		uidHeaders:          uidHeaders,
		groupHeaders:        groupHeaders,
		extraHeaderPrefixes: extraHeaderPrefixes,
	}
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

// secureRequestHeaderAuthenticator authenticates front-proxy requests by verifying the client
// certificate against the CA bundle of the given caContentProvider and, on success, extracting
// the user info from the request headers as described by the configuration returned by the
// provider.
type secureRequestHeaderAuthenticator struct {
	caContentProvider dynamiccertificates.CAContentProvider
	provider          RequestHeaderConfigProvider
}

var _ authenticator.Request = &secureRequestHeaderAuthenticator{}

type requestHeaderConfigContextKey struct{}

type requestHeaderConfigContextValue struct {
	config *RequestHeaderConfig
}

// WithRequestHeaderConfig returns a context that supplies config to NewSecure. The context
// value allows request authentication and subsequent request header clearing to use the same
// point-in-time configuration.
func WithRequestHeaderConfig(ctx context.Context, config *RequestHeaderConfig) context.Context {
	return context.WithValue(ctx, requestHeaderConfigContextKey{}, requestHeaderConfigContextValue{config: config})
}

// RequestHeaderConfigFromContext returns the request header configuration supplied with
// WithRequestHeaderConfig. The boolean distinguishes an unavailable configuration from one
// that was not supplied in the context.
func RequestHeaderConfigFromContext(ctx context.Context) (*RequestHeaderConfig, bool) {
	value, ok := ctx.Value(requestHeaderConfigContextKey{}).(requestHeaderConfigContextValue)
	if !ok {
		return nil, false
	}
	return value.config, true
}

// NewSecure returns a request.Authenticator that verifies the request's client certificate
// against the CA bundle served by caContentProvider and then extracts user info from the
// request headers according to a configuration snapshot taken from provider.
//
// Exactly one snapshot is taken per request, so certificate verification, allowed common name
// enforcement and header extraction all observe a mutually consistent view of the configuration
// even when it changes concurrently.
func NewSecure(provider RequestHeaderConfigProvider, caContentProvider dynamiccertificates.CAContentProvider) authenticator.Request {
	return &secureRequestHeaderAuthenticator{
		caContentProvider: caContentProvider,
		provider:          provider,
	}
}

func (a *secureRequestHeaderAuthenticator) AuthenticateRequest(req *http.Request) (*authenticator.Response, bool, error) {
	config, found := RequestHeaderConfigFromContext(req.Context())
	if !found {
		config = a.provider.Snapshot()
	}
	if config == nil {
		// no configuration available (the source configmap has never been read or was deleted),
		// fail closed by expressing "no opinion" with no headers extracted and none cleared
		return nil, false, nil
	}
	return a.authenticateWithConfig(req, config)
}

// authenticateWithConfig authenticates the request using exactly the given configuration snapshot.
func (a *secureRequestHeaderAuthenticator) authenticateWithConfig(req *http.Request, config *RequestHeaderConfig) (*authenticator.Response, bool, error) {
	headerAuthenticator := NewDynamic(
		StaticStringSlice(config.UsernameHeaders),
		StaticStringSlice(config.UIDHeaders),
		StaticStringSlice(config.GroupHeaders),
		StaticStringSlice(config.ExtraHeaderPrefixes),
	)
	verifier := x509request.NewDynamicCAVerifier(
		a.caContentProvider.VerifyOptions,
		headerAuthenticator,
		StaticStringSlice(config.AllowedClientNames),
	)
	return verifier.AuthenticateRequest(req)
}

func (a *requestHeaderAuthRequestHandler) AuthenticateRequest(req *http.Request) (*authenticator.Response, bool, error) {
	name := headerValue(req.Header, a.nameHeaders.Value())
	if len(name) == 0 {
		return nil, false, nil
	}
	uid := headerValue(req.Header, a.uidHeaders.Value())
	groups := allHeaderValues(req.Header, a.groupHeaders.Value())
	extra := newExtra(req.Header, a.extraHeaderPrefixes.Value())

	// clear headers used for authentication
	ClearAuthenticationHeaders(req.Header, a.nameHeaders, a.uidHeaders, a.groupHeaders, a.extraHeaderPrefixes)

	return &authenticator.Response{
		User: &user.DefaultInfo{
			Name:   name,
			UID:    uid,
			Groups: groups,
			Extra:  extra,
		},
	}, true, nil
}

// ClearAuthenticationHeaders deletes all headers used for request header authentication
// from the given header map, according to the given providers.
func ClearAuthenticationHeaders(h http.Header, nameHeaders, uidHeaders, groupHeaders, extraHeaderPrefixes StringSliceProvider) {
	clearAuthenticationHeaders(h, nameHeaders.Value(), uidHeaders.Value(), groupHeaders.Value(), extraHeaderPrefixes.Value())
}

// ClearAuthenticationHeadersFromConfig deletes all headers used for request header authentication
// from the given header map, according to the given configuration snapshot. It is a no-op for a
// nil snapshot.
func ClearAuthenticationHeadersFromConfig(h http.Header, config *RequestHeaderConfig) {
	if config == nil {
		return
	}
	clearAuthenticationHeaders(h, config.UsernameHeaders, config.UIDHeaders, config.GroupHeaders, config.ExtraHeaderPrefixes)
}

func clearAuthenticationHeaders(h http.Header, nameHeaders, uidHeaders, groupHeaders, extraHeaderPrefixes []string) {
	for _, headerName := range nameHeaders {
		h.Del(headerName)
	}
	for _, headerName := range uidHeaders {
		h.Del(headerName)
	}
	for _, headerName := range groupHeaders {
		h.Del(headerName)
	}
	for _, prefix := range extraHeaderPrefixes {
		for k := range h {
			if hasPrefixIgnoreCase(k, prefix) {
				delete(h, k) // we have the raw key so avoid relying on canonicalization
			}
		}
	}
}

func hasPrefixIgnoreCase(s, prefix string) bool {
	return len(s) >= len(prefix) && strings.EqualFold(s[:len(prefix)], prefix)
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
			if !hasPrefixIgnoreCase(headerName, prefix) {
				continue
			}

			extraKey := unescapeExtraKey(strings.ToLower(headerName[len(prefix):]))
			ret[extraKey] = append(ret[extraKey], vv...)
		}
	}

	return ret
}
