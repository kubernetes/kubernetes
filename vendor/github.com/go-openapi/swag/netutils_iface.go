// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package swag

import "github.com/go-openapi/swag/netutils"

// SplitHostPort splits a network address into a host and a port.
//
// Deprecated: use [netutils.SplitHostPort] instead.
func SplitHostPort(addr string) (host string, port int, err error) {
	return netutils.SplitHostPort(addr)
}
