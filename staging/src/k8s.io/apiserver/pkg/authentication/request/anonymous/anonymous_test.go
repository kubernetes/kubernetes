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

package anonymous

import (
	"net/http"
	"net/url"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/authentication/user"
)

func TestAnonymous(t *testing.T) {
	a := NewAuthenticator(nil)
	r, ok, err := a.AuthenticateRequest(&http.Request{})
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if !ok {
		t.Fatalf("Unexpectedly unauthenticated")
	}
	if r.User.GetName() != user.Anonymous {
		t.Fatalf("Expected username %s, got %s", user.Anonymous, r.User.GetName())
	}
	if !sets.NewString(r.User.GetGroups()...).Equal(sets.NewString(user.AllUnauthenticated)) {
		t.Fatalf("Expected group %s, got %v", user.AllUnauthenticated, r.User.GetGroups())
	}
}

func TestAnonymousRestricted(t *testing.T) {
	a := NewAuthenticator([]apiserver.AnonymousAuthCondition{
		{
			Path: "/healthz",
		},
		{
			Path: "/readyz",
		},
		{
			Path: "/livez",
		},
	})

	testCases := []struct {
		desc        string
		path        string
		want        user.DefaultInfo
		wantAllowed bool
	}{
		{
			desc: "/healthz",
			path: "https://123.123.123.123/healthz",
			want: user.DefaultInfo{
				Name:   anonymousUser,
				Groups: []string{unauthenticatedGroup},
			},
			wantAllowed: true,
		},
		{
			desc: "/readyz",
			path: "https://123.123.123.123/readyz",
			want: user.DefaultInfo{
				Name:   anonymousUser,
				Groups: []string{unauthenticatedGroup},
			},
			wantAllowed: true,
		},
		{
			desc: "/livez",
			path: "https://123.123.123.123/livez",
			want: user.DefaultInfo{
				Name:   anonymousUser,
				Groups: []string{unauthenticatedGroup},
			},
			wantAllowed: true,
		},
		{
			desc:        "/api",
			path:        "https://123.123.123.123/api",
			wantAllowed: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			u, err := url.Parse(tc.path)
			if err != nil {
				t.Fatal(err)
			}
			r, allowed, err := a.AuthenticateRequest(&http.Request{URL: u})
			if err != nil {
				t.Fatal(err)
			}

			if tc.wantAllowed != allowed {
				t.Fatalf("want allowed: %v, got allowed: %v", tc.wantAllowed, allowed)
			}

			if !tc.wantAllowed {
				return
			}

			if r.User.GetName() != tc.want.Name {
				t.Fatalf("Expected username %s, got %s", user.Anonymous, r.User.GetName())
			}
			if !sets.NewString(r.User.GetGroups()...).Equal(sets.NewString(tc.want.Groups...)) {
				t.Fatalf("Expected group %s, got %v", tc.want.Groups, r.User.GetGroups())
			}
		})
	}
}
