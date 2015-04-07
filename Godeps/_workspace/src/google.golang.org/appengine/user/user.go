// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Package user provides a client for App Engine's user authentication service.
package user

import (
	"strings"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/user"
)

// User represents a user of the application.
type User struct {
	Email      string
	AuthDomain string
	Admin      bool

	// ID is the unique permanent ID of the user.
	// It is populated if the Email is associated
	// with a Google account, or empty otherwise.
	ID string

	FederatedIdentity string
	FederatedProvider string
}

// String returns a displayable name for the user.
func (u *User) String() string {
	if u.AuthDomain != "" && strings.HasSuffix(u.Email, "@"+u.AuthDomain) {
		return u.Email[:len(u.Email)-len("@"+u.AuthDomain)]
	}
	if u.FederatedIdentity != "" {
		return u.FederatedIdentity
	}
	return u.Email
}

// LoginURL returns a URL that, when visited, prompts the user to sign in,
// then redirects the user to the URL specified by dest.
func LoginURL(c appengine.Context, dest string) (string, error) {
	return LoginURLFederated(c, dest, "")
}

// LoginURLFederated is like LoginURL but accepts a user's OpenID identifier.
func LoginURLFederated(c appengine.Context, dest, identity string) (string, error) {
	req := &pb.CreateLoginURLRequest{
		DestinationUrl: proto.String(dest),
	}
	if identity != "" {
		req.FederatedIdentity = proto.String(identity)
	}
	res := &pb.CreateLoginURLResponse{}
	if err := c.Call("user", "CreateLoginURL", req, res, nil); err != nil {
		return "", err
	}
	return *res.LoginUrl, nil
}

// LogoutURL returns a URL that, when visited, signs the user out,
// then redirects the user to the URL specified by dest.
func LogoutURL(c appengine.Context, dest string) (string, error) {
	req := &pb.CreateLogoutURLRequest{
		DestinationUrl: proto.String(dest),
	}
	res := &pb.CreateLogoutURLResponse{}
	if err := c.Call("user", "CreateLogoutURL", req, res, nil); err != nil {
		return "", err
	}
	return *res.LogoutUrl, nil
}

// Current returns the currently logged-in user,
// or nil if the user is not signed in.
func Current(c appengine.Context) *User {
	u := &User{
		Email:             internal.VirtAPI(c, "user:Email"),
		AuthDomain:        internal.VirtAPI(c, "user:AuthDomain"),
		ID:                internal.VirtAPI(c, "user:ID"),
		Admin:             internal.VirtAPI(c, "user:IsAdmin") == "1",
		FederatedIdentity: internal.VirtAPI(c, "user:FederatedIdentity"),
		FederatedProvider: internal.VirtAPI(c, "user:FederatedProvider"),
	}
	if u.Email == "" && u.FederatedIdentity == "" {
		return nil
	}
	return u
}

// IsAdmin returns true if the current user is signed in and
// is currently registered as an administrator of the application.
func IsAdmin(c appengine.Context) bool {
	return internal.VirtAPI(c, "user:IsAdmin") == "1"
}

func init() {
	internal.RegisterErrorCodeMap("user", pb.UserServiceError_ErrorCode_name)
}
