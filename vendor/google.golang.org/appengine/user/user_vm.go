// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build !appengine

package user

import (
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
)

// Current returns the currently logged-in user,
// or nil if the user is not signed in.
func Current(c context.Context) *User {
	h := internal.IncomingHeaders(c)
	u := &User{
		Email:             h.Get("X-AppEngine-User-Email"),
		AuthDomain:        h.Get("X-AppEngine-Auth-Domain"),
		ID:                h.Get("X-AppEngine-User-Id"),
		Admin:             h.Get("X-AppEngine-User-Is-Admin") == "1",
		FederatedIdentity: h.Get("X-AppEngine-Federated-Identity"),
		FederatedProvider: h.Get("X-AppEngine-Federated-Provider"),
	}
	if u.Email == "" && u.FederatedIdentity == "" {
		return nil
	}
	return u
}

// IsAdmin returns true if the current user is signed in and
// is currently registered as an administrator of the application.
func IsAdmin(c context.Context) bool {
	h := internal.IncomingHeaders(c)
	return h.Get("X-AppEngine-User-Is-Admin") == "1"
}
