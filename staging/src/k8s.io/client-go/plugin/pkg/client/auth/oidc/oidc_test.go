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

package oidc

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"testing"
	"time"
)

func TestJSONTime(t *testing.T) {
	data := `{
		"t1": 1493851263,
		"t2": 1.493851263e9
	}`

	var v struct {
		T1 jsonTime `json:"t1"`
		T2 jsonTime `json:"t2"`
	}
	if err := json.Unmarshal([]byte(data), &v); err != nil {
		t.Fatal(err)
	}
	wantT1 := time.Unix(1493851263, 0)
	wantT2 := time.Unix(1493851263, 0)
	gotT1 := time.Time(v.T1)
	gotT2 := time.Time(v.T2)

	if !wantT1.Equal(gotT1) {
		t.Errorf("t1 value: wanted %s got %s", wantT1, gotT1)
	}
	if !wantT2.Equal(gotT2) {
		t.Errorf("t2 value: wanted %s got %s", wantT2, gotT2)
	}
}

func encodeJWT(header, payload, sig string) string {
	e := func(s string) string {
		return base64.RawURLEncoding.EncodeToString([]byte(s))
	}
	return e(header) + "." + e(payload) + "." + e(sig)
}

func TestExpired(t *testing.T) {
	now := time.Now()

	nowFunc := func() time.Time { return now }

	tests := []struct {
		name        string
		idToken     string
		wantErr     bool
		wantExpired bool
	}{
		{
			name: "valid",
			idToken: encodeJWT(
				"{}",
				fmt.Sprintf(`{"exp":%d}`, now.Add(time.Hour).Unix()),
				"blah", // signature isn't veified.
			),
		},
		{
			name: "expired",
			idToken: encodeJWT(
				"{}",
				fmt.Sprintf(`{"exp":%d}`, now.Add(-time.Hour).Unix()),
				"blah", // signature isn't veified.
			),
			wantExpired: true,
		},
		{
			name: "bad exp claim",
			idToken: encodeJWT(
				"{}",
				`{"exp":"foobar"}`,
				"blah", // signature isn't veified.
			),
			wantErr: true,
		},
		{
			name:    "not an id token",
			idToken: "notanidtoken",
			wantErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			valid, err := idTokenExpired(nowFunc, test.idToken)
			if err != nil {
				if !test.wantErr {
					t.Errorf("parse error: %v", err)
				}
				return
			}
			if test.wantExpired == valid {
				t.Errorf("wanted expired %t, got %", test.wantExpired, !valid)
			}
		})
	}
}

func TestClientCache(t *testing.T) {
	cache := newClientCache()

	if _, ok := cache.getClient("issuer1", "id1"); ok {
		t.Fatalf("got client before putting one in the cache")
	}

	cli1 := new(oidcAuthProvider)
	cli2 := new(oidcAuthProvider)

	gotcli := cache.setClient("issuer1", "id1", cli1)
	if cli1 != gotcli {
		t.Fatalf("set first client and got a different one")
	}

	gotcli = cache.setClient("issuer1", "id1", cli2)
	if cli1 != gotcli {
		t.Fatalf("set a second client and didn't get the first")
	}
}
