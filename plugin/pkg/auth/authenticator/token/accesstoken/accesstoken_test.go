/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package accesstoken

import (
	"errors"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrs "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
	oauthclient "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/client"
)

func TestTokenAuthenticatorRegistryError(t *testing.T) {
	client := &oauthclient.Fake{Err: errors.New("other error")}
	auth := NewTokenAuthenticator(client)
	_, _, err := auth.AuthenticateToken("foo")
	if err == nil {
		t.Fatalf("expected error, got none")
	}
}

func TestTokenAuthenticatorNotFoundError(t *testing.T) {
	client := &oauthclient.Fake{Err: apierrs.NewNotFound("accessToken", "foo")}
	auth := NewTokenAuthenticator(client)
	_, ok, err := auth.AuthenticateToken("foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Fatalf("expected auth failure, got ok")
	}
}

func TestTokenAuthenticatorExpired(t *testing.T) {
	client := &oauthclient.Fake{
		AccessToken: oapi.OAuthAccessToken{
			ObjectMeta: api.ObjectMeta{
				CreationTimestamp: util.Time{time.Now().Add(-61 * time.Second)},
			},
			ExpiresIn: 60, // seconds
		},
	}
	auth := NewTokenAuthenticator(client)
	_, ok, err := auth.AuthenticateToken("foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Fatalf("expected auth failure, got ok")
	}
}

func TestTokenAuthenticatorSuccess(t *testing.T) {
	client := &oauthclient.Fake{
		AccessToken: oapi.OAuthAccessToken{
			ObjectMeta: api.ObjectMeta{
				CreationTimestamp: util.Now(),
			},
			ExpiresIn: 1, // seconds
			UserName:  "user",
			UserUID:   "uid",
		},
	}
	auth := NewTokenAuthenticator(client)
	user, ok, err := auth.AuthenticateToken("foo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Fatalf("expected auth success, got failure")
	}
	if user.GetName() != "user" || user.GetUID() != "uid" {
		t.Fatalf("got unexpected user: %v", user)
	}
}
