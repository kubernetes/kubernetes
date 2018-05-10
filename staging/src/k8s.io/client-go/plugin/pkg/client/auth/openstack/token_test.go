/*
Copyright 2018 The Kubernetes Authors.

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

package openstack

import (
	"testing"
	"time"
)

func TestOpenstackTokenExpiresAt(t *testing.T) {
	now := time.Now()
	cases := []struct {
		name string
		tok  *openstackToken
		want bool
	}{
		{name: "12 seconds", tok: &openstackToken{ExpiresAt: now.Add(12 * time.Second)}, want: false},
		{name: "10 seconds", tok: &openstackToken{ExpiresAt: now.Add(expiresAtDelta)}, want: true},
		{name: "-1 hour", tok: &openstackToken{ExpiresAt: now.Add(-1 * time.Hour)}, want: true},
	}
	for _, tc := range cases {
		if got, want := tc.tok.expired(), tc.want; got != want {
			t.Errorf("expired (%q) = %v; want %v", tc.name, got, want)
		}
	}
}
