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
// This is similar to WithEndpoint, but allows us to determine whether the user has overriden the default endpoint.
func WithDefaultEndpoint(url string) option.ClientOption {
	return defaultEndpointOption(url)
}
