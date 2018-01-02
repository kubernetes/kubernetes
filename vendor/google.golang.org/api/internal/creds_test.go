// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

type dummyTokenSource struct {
	oauth2.TokenSource
}

func TestTokenSource(t *testing.T) {
	ctx := context.Background()

	// Pass in a TokenSource, get it back.
	ts := &dummyTokenSource{}
	ds := &DialSettings{TokenSource: ts}
	got, err := Creds(ctx, ds)
	if err != nil {
		t.Fatal(err)
	}
	want := &google.DefaultCredentials{TokenSource: ts}
	if !cmp.Equal(got, want) {
		t.Error("did not get the same TokenSource back")
	}

	// Load a valid JSON file. No way to really test the contents; we just
	// verify that there is no error.
	ds = &DialSettings{CredentialsFile: "service-account.json"}
	if _, err := Creds(ctx, ds); err != nil {
		t.Errorf("got %v, wanted no error", err)
	}

	// If both a file and TokenSource are passed, the file takes precedence
	// (existing behavior).
	// TODO(jba): make this an error?
	ds = &DialSettings{
		TokenSource:     ts,
		CredentialsFile: "service-account.json",
	}
	got, err = Creds(ctx, ds)
	if err != nil {
		t.Fatal(err)
	}
	if cmp.Equal(got, want) {
		t.Error("got the same TokenSource back, wanted one from the JSON file")
	}
	// TODO(jba): find a way to test the call to google.DefaultTokenSource.
}

const validRefeshTokenJSON = `{
  "client_id": "764-aaaaa.apps.googleusercontent.com",
  "client_secret": "d-988888888",
  "refresh_token": "1/88888aaaaaaaaa",
  "type": "authorized_user"
}`

const validServiceAccountJSON = `{
  "type": "service_account",
  "project_id": "dumba-504",
  "private_key_id": "adsfsdd",
  "private_key": "-----BEGIN PRIVATE KEY-----\n\n-----END PRIVATE KEY-----\n",
  "client_email": "dumba-504@appspot.gserviceaccount.com",
  "client_id": "111",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://accounts.google.com/o/oauth2/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/dumba-504%40appspot.gserviceaccount.com"
}`

func TestRefreshTokenTokenSource(t *testing.T) {
	tests := []struct {
		name    string
		data    []byte
		wantOK  bool
		wantErr bool
	}{
		{
			name:    "empty",
			data:    []byte{},
			wantOK:  false,
			wantErr: true, // not valid JSON
		},
		{
			name:    "non refresh token JSON",
			data:    []byte("{}"),
			wantOK:  false,
			wantErr: false,
		},
		{
			name:    "service account JSON",
			data:    []byte(validServiceAccountJSON),
			wantOK:  false,
			wantErr: false,
		},
		{
			name:    "valid refresh token JSON",
			data:    []byte(validRefeshTokenJSON),
			wantOK:  true,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		_, ok, err := refreshTokenTokenSource(context.Background(), tt.data)
		if (err != nil) != tt.wantErr {
			t.Errorf("%v: refreshTokenTokenSource() err = %v, wantErr %v", tt.name, err, tt.wantErr)
		}
		if ok != tt.wantOK {
			t.Errorf("%v: refreshTokenTokenSource() ok = %v, want %v", tt.name, ok, tt.wantOK)
		}
	}
}
