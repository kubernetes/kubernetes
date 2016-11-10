/*
Copyright 2016 The Kubernetes Authors.

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

package gcp

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"golang.org/x/oauth2"
)

func TestCmdTokenSource(t *testing.T) {
	fakeExpiry := time.Date(2016, 10, 31, 22, 31, 9, 123000000, time.UTC)
	customFmt := "2006-01-02 15:04:05.999999999"

	tests := []struct {
		name                              string
		output                            []byte
		cmd, tokenKey, expiryKey, timeFmt string
		tok                               *oauth2.Token
		expectErr                         error
	}{
		{
			"defaults",
			[]byte(`{
  "access_token": "faketoken",
  "token_expiry": "2016-10-31T22:31:09.123000000Z"
}`),
			"/fake/cmd/path", "", "", "",
			&oauth2.Token{
				AccessToken: "faketoken",
				TokenType:   "Bearer",
				Expiry:      fakeExpiry,
			},
			nil,
		},
		{
			"custom keys",
			[]byte(`{
  "token": "faketoken",
  "token_expiry": {
    "datetime": "2016-10-31 22:31:09.123"
  }
}`),
			"/fake/cmd/path", "{.token}", "{.token_expiry.datetime}", customFmt,
			&oauth2.Token{
				AccessToken: "faketoken",
				TokenType:   "Bearer",
				Expiry:      fakeExpiry,
			},
			nil,
		},
		{
			"missing cmd",
			nil,
			"", "", "", "",
			nil,
			fmt.Errorf("missing access token cmd"),
		},
		{
			"missing token-key",
			[]byte(`{
  "broken": "faketoken",
  "token_expiry": {
    "datetime": "2016-10-31 22:31:09.123000000Z"
  }
}`),
			"/fake/cmd/path", "{.token}", "", "",
			nil,
			fmt.Errorf("error parsing token-key %q", "{.token}"),
		},
		{
			"missing expiry-key",
			[]byte(`{
  "access_token": "faketoken",
  "expires": "2016-10-31T22:31:09.123000000Z"
}`),
			"/fake/cmd/path", "", "{.expiry}", "",
			nil,
			fmt.Errorf("error parsing expiry-key %q", "{.expiry}"),
		},
		{
			"invalid expiry timestamp",
			[]byte(`{
  "access_token": "faketoken",
  "token_expiry": "sometime soon, idk"
}`),
			"/fake/cmd/path", "", "", "",
			&oauth2.Token{
				AccessToken: "faketoken",
				TokenType:   "Bearer",
				Expiry:      time.Time{},
			},
			nil,
		},
		{
			"bad JSON",
			[]byte(`{
  "access_token": "faketoken",
  "token_expiry": "sometime soon, idk"
  ------
`),
			"/fake/cmd", "", "", "",
			nil,
			fmt.Errorf("invalid character '-' after object key:value pair"),
		},
	}

	for _, tc := range tests {
		ts, err := newCmdTokenSource(tc.cmd, tc.tokenKey, tc.expiryKey, tc.timeFmt)
		if err != nil {
			if !strings.Contains(err.Error(), tc.expectErr.Error()) {
				t.Errorf("%s newCmdTokenSource error: %v, want %v", tc.name, err, tc.expectErr)
			}
			continue
		}
		tok, err := ts.parseTokenCmdOutput(tc.output)

		if err != tc.expectErr && !strings.Contains(err.Error(), tc.expectErr.Error()) {
			t.Errorf("%s parseCmdTokenSource error: %v, want %v", tc.name, err, tc.expectErr)
		}
		if !reflect.DeepEqual(tok, tc.tok) {
			t.Errorf("%s got token %v, want %v", tc.name, tok, tc.tok)
		}
	}
}
