// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package swag

import "github.com/go-openapi/swag/typeutils"

// IsZero returns true when the value passed into the function is a zero value.
// This allows for safer checking of interface values.
//
// Deprecated: use [typeutils.IsZero] instead.
func IsZero(data any) bool { return typeutils.IsZero(data) }
