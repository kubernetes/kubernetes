// Copyright 2015 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"strings"
	"testing"
)

var webJSONKey = []byte(`
{
    "web": {
        "auth_uri": "https://google.com/o/oauth2/auth",
        "client_secret": "3Oknc4jS_wA2r9i",
        "token_uri": "https://google.com/o/oauth2/token",
        "client_email": "222-nprqovg5k43uum874cs9osjt2koe97g8@developer.gserviceaccount.com",
        "redirect_uris": ["https://www.example.com/oauth2callback"],
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/222-nprqovg5k43uum874cs9osjt2koe97g8@developer.gserviceaccount.com",
        "client_id": "222-nprqovg5k43uum874cs9osjt2koe97g8.apps.googleusercontent.com",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "javascript_origins": ["https://www.example.com"]
    }
}`)

var installedJSONKey = []byte(`{
  "installed": {
      "client_id": "222-installed.apps.googleusercontent.com",
      "redirect_uris": ["https://www.example.com/oauth2callback"]
    }
}`)

func TestConfigFromJSON(t *testing.T) {
	conf, err := ConfigFromJSON(webJSONKey, "scope1", "scope2")
	if err != nil {
		t.Error(err)
	}
	if got, want := conf.ClientID, "222-nprqovg5k43uum874cs9osjt2koe97g8.apps.googleusercontent.com"; got != want {
		t.Errorf("ClientID = %q; want %q", got, want)
	}
	if got, want := conf.ClientSecret, "3Oknc4jS_wA2r9i"; got != want {
		t.Errorf("ClientSecret = %q; want %q", got, want)
	}
	if got, want := conf.RedirectURL, "https://www.example.com/oauth2callback"; got != want {
		t.Errorf("RedictURL = %q; want %q", got, want)
	}
	if got, want := strings.Join(conf.Scopes, ","), "scope1,scope2"; got != want {
		t.Errorf("Scopes = %q; want %q", got, want)
	}
	if got, want := conf.Endpoint.AuthURL, "https://google.com/o/oauth2/auth"; got != want {
		t.Errorf("AuthURL = %q; want %q", got, want)
	}
	if got, want := conf.Endpoint.TokenURL, "https://google.com/o/oauth2/token"; got != want {
		t.Errorf("TokenURL = %q; want %q", got, want)
	}
}

func TestConfigFromJSON_Installed(t *testing.T) {
	conf, err := ConfigFromJSON(installedJSONKey)
	if err != nil {
		t.Error(err)
	}
	if got, want := conf.ClientID, "222-installed.apps.googleusercontent.com"; got != want {
		t.Errorf("ClientID = %q; want %q", got, want)
	}
}
