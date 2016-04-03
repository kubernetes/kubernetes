// Copyright 2015 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import "testing"

func TestSDKConfig(t *testing.T) {
	sdkConfigPath = func() (string, error) {
		return "testdata/gcloud", nil
	}

	tests := []struct {
		account     string
		accessToken string
		err         bool
	}{
		{"", "bar_access_token", false},
		{"foo@example.com", "foo_access_token", false},
		{"bar@example.com", "bar_access_token", false},
		{"baz@serviceaccount.example.com", "", true},
	}
	for _, tt := range tests {
		c, err := NewSDKConfig(tt.account)
		if got, want := err != nil, tt.err; got != want {
			if !tt.err {
				t.Errorf("expected no error, got error: %v", tt.err, err)
			} else {
				t.Errorf("expected error, got none")
			}
			continue
		}
		if err != nil {
			continue
		}
		tok := c.initialToken
		if tok == nil {
			t.Errorf("expected token %q, got: nil", tt.accessToken)
			continue
		}
		if tok.AccessToken != tt.accessToken {
			t.Errorf("expected token %q, got: %q", tt.accessToken, tok.AccessToken)
		}
	}
}
