// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build !appengine

package user

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine/internal"
	"google.golang.org/appengine/internal/aetesting"
	pb "google.golang.org/appengine/internal/user"
)

func baseReq() *http.Request {
	return &http.Request{
		Header: http.Header{},
	}
}

type basicUserTest struct {
	nickname, email, authDomain, admin string
	// expectations
	isNil, isAdmin bool
	displayName    string
}

var basicUserTests = []basicUserTest{
	{"", "", "", "0", true, false, ""},
	{"ken", "ken@example.com", "example.com", "0", false, false, "ken"},
	{"ken", "ken@example.com", "auth_domain.com", "1", false, true, "ken@example.com"},
}

func TestBasicUserAPI(t *testing.T) {
	for i, tc := range basicUserTests {
		req := baseReq()
		req.Header.Set("X-AppEngine-User-Nickname", tc.nickname)
		req.Header.Set("X-AppEngine-User-Email", tc.email)
		req.Header.Set("X-AppEngine-Auth-Domain", tc.authDomain)
		req.Header.Set("X-AppEngine-User-Is-Admin", tc.admin)

		c := internal.ContextForTesting(req)

		if ga := IsAdmin(c); ga != tc.isAdmin {
			t.Errorf("test %d: expected IsAdmin(c) = %v, got %v", i, tc.isAdmin, ga)
		}

		u := Current(c)
		if tc.isNil {
			if u != nil {
				t.Errorf("test %d: expected u == nil, got %+v", i, u)
			}
			continue
		}
		if u == nil {
			t.Errorf("test %d: expected u != nil, got nil", i)
			continue
		}
		if u.Email != tc.email {
			t.Errorf("test %d: expected u.Email = %q, got %q", i, tc.email, u.Email)
		}
		if gs := u.String(); gs != tc.displayName {
			t.Errorf("test %d: expected u.String() = %q, got %q", i, tc.displayName, gs)
		}
		if u.Admin != tc.isAdmin {
			t.Errorf("test %d: expected u.Admin = %v, got %v", i, tc.isAdmin, u.Admin)
		}
	}
}

func TestLoginURL(t *testing.T) {
	expectedQuery := &pb.CreateLoginURLRequest{
		DestinationUrl: proto.String("/destination"),
	}
	const expectedDest = "/redir/dest"
	c := aetesting.FakeSingleContext(t, "user", "CreateLoginURL", func(req *pb.CreateLoginURLRequest, res *pb.CreateLoginURLResponse) error {
		if !proto.Equal(req, expectedQuery) {
			return fmt.Errorf("got %v, want %v", req, expectedQuery)
		}
		res.LoginUrl = proto.String(expectedDest)
		return nil
	})

	url, err := LoginURL(c, "/destination")
	if err != nil {
		t.Fatalf("LoginURL failed: %v", err)
	}
	if url != expectedDest {
		t.Errorf("got %v, want %v", url, expectedDest)
	}
}

// TODO(dsymonds): Add test for LogoutURL.
