// Copyright 2020 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package internaloption contains options used internally by Google client code.
package internaloption

import (
	"google.golang.org/api/internal"
	"google.golang.org/api/option"
)

type defaultEndpointOption string

func (o defaultEndpointOption) Apply(settings *internal.DialSettings) {
	settings.DefaultEndpoint = string(o)
}

// WithDefaultEndpoint is an option that indicates the default endpoint.
//
// It should only be used internally by generated clients.
//
// This is similar to WithEndpoint, but allows us to determine whether the user has overridden the default endpoint.
func WithDefaultEndpoint(url string) option.ClientOption {
	return defaultEndpointOption(url)
}

type defaultMTLSEndpointOption string

func (o defaultMTLSEndpointOption) Apply(settings *internal.DialSettings) {
	settings.DefaultMTLSEndpoint = string(o)
}

// WithDefaultMTLSEndpoint is an option that indicates the default mTLS endpoint.
//
// It should only be used internally by generated clients.
func WithDefaultMTLSEndpoint(url string) option.ClientOption {
	return defaultMTLSEndpointOption(url)
}

// SkipDialSettingsValidation bypasses validation on ClientOptions.
//
// It should only be used internally.
func SkipDialSettingsValidation() option.ClientOption {
	return skipDialSettingsValidation{}
}

type skipDialSettingsValidation struct{}

func (s skipDialSettingsValidation) Apply(settings *internal.DialSettings) {
	settings.SkipValidation = true
}

// EnableDirectPath returns a ClientOption that overrides the default
// attempt to use DirectPath.
//
// It should only be used internally by generated clients.
// This is an EXPERIMENTAL API and may be changed or removed in the future.
func EnableDirectPath(dp bool) option.ClientOption {
	return enableDirectPath(dp)
}

type enableDirectPath bool

func (e enableDirectPath) Apply(o *internal.DialSettings) {
	o.EnableDirectPath = bool(e)
}

// WithDefaultAudience returns a ClientOption that specifies a default audience
// to be used as the audience field ("aud") for the JWT token authentication.
//
// It should only be used internally by generated clients.
func WithDefaultAudience(audience string) option.ClientOption {
	return withDefaultAudience(audience)
}

type withDefaultAudience string

func (w withDefaultAudience) Apply(o *internal.DialSettings) {
	o.DefaultAudience = string(w)
}

// WithDefaultScopes returns a ClientOption that overrides the default OAuth2
// scopes to be used for a service.
//
// It should only be used internally by generated clients.
func WithDefaultScopes(scope ...string) option.ClientOption {
	return withDefaultScopes(scope)
}

type withDefaultScopes []string

func (w withDefaultScopes) Apply(o *internal.DialSettings) {
	o.DefaultScopes = make([]string, len(w))
	copy(o.DefaultScopes, w)
}
