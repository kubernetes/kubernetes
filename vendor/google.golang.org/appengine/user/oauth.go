// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package user

import (
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/user"
)

// CurrentOAuth returns the user associated with the OAuth consumer making this
// request. If the OAuth consumer did not make a valid OAuth request, or the
// scopes is non-empty and the current user does not have at least one of the
// scopes, this method will return an error.
func CurrentOAuth(c context.Context, scopes ...string) (*User, error) {
	req := &pb.GetOAuthUserRequest{}
	if len(scopes) != 1 || scopes[0] != "" {
		// The signature for this function used to be CurrentOAuth(Context, string).
		// Ignore the singular "" scope to preserve existing behavior.
		req.Scopes = scopes
	}

	res := &pb.GetOAuthUserResponse{}

	err := internal.Call(c, "user", "GetOAuthUser", req, res)
	if err != nil {
		return nil, err
	}
	return &User{
		Email:      *res.Email,
		AuthDomain: *res.AuthDomain,
		Admin:      res.GetIsAdmin(),
		ID:         *res.UserId,
		ClientID:   res.GetClientId(),
	}, nil
}

// OAuthConsumerKey returns the OAuth consumer key provided with the current
// request. This method will return an error if the OAuth request was invalid.
func OAuthConsumerKey(c context.Context) (string, error) {
	req := &pb.CheckOAuthSignatureRequest{}
	res := &pb.CheckOAuthSignatureResponse{}

	err := internal.Call(c, "user", "CheckOAuthSignature", req, res)
	if err != nil {
		return "", err
	}
	return *res.OauthConsumerKey, err
}
