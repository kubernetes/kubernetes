// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package swag

import (
	"github.com/go-openapi/jsonpointer/jsonname"
)

// DefaultJSONNameProvider is the default cache for types
//
// Deprecated: use [github.com/go-openapi/jsonpointer/jsonname.DefaultJSONNameProvider] instead.
var DefaultJSONNameProvider = jsonname.DefaultJSONNameProvider

// NameProvider represents an object capable of translating from go property names
// to json property names.
//
// Deprecated: use [github.com/go-openapi/jsonpointer/jsonname.NameProvider] instead.
type NameProvider = jsonname.NameProvider

// NewNameProvider creates a new name provider
//
// Deprecated: use [github.com/go-openapi/jsonpointer/jsonname.NewNameProvider] instead.
func NewNameProvider() *NameProvider { return jsonname.NewNameProvider() }
