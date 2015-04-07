// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package user

import (
	"google.golang.org/appengine"
	pb "google.golang.org/appengine/internal/user"
)

// CurrentOAuth returns the user associated with the OAuth consumer making this
// request. If the OAuth consumer did not make a valid OAuth request, or the
// scope is non-empty and the current user does not have this scope, this method
// will return an error.
func CurrentOAuth(c appengine.Context, scope string) (*User, error) {
	req := &pb.GetOAuthUserRequest{}
	if scope != "" {
		req.Scope = &scope
	}
	res := &pb.GetOAuthUserResponse{}

	err := c.Call("user", "GetOAuthUser", req, res, nil)
	if err != nil {
		return nil, err
	}
	return &User{
		Email:      *res.Email,
		AuthDomain: *res.AuthDomain,
		Admin:      res.GetIsAdmin(),
		ID:         *res.UserId,
	}, nil
}

// OAuthConsumerKey returns the OAuth consumer key provided with the current
// request. This method will return an error if the OAuth request was invalid.
func OAuthConsumerKey(c appengine.Context) (string, error) {
	req := &pb.CheckOAuthSignatureRequest{}
	res := &pb.CheckOAuthSignatureResponse{}

	err := c.Call("user", "CheckOAuthSignature", req, res, nil)
	if err != nil {
		return "", err
	}
	return *res.OauthConsumerKey, err
}
