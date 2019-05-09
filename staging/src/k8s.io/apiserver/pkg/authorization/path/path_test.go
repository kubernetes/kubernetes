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

package path

import (
	"testing"

	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestNewAuthorizer(t *testing.T) {
	tests := []struct {
		name                       string
		excludedPaths              []string
		allowed, denied, noOpinion []string
		wantErr                    bool
	}{
		{"inner star", []string{"/foo*bar"}, nil, nil, nil, true},
		{"double star", []string{"/foo**"}, nil, nil, nil, true},
		{"empty", nil, nil, nil, []string{"/"}, false},
		{"slash", []string{"/"}, []string{"/"}, nil, []string{"/foo", "//"}, false},
		{"foo", []string{"/foo"}, []string{"/foo", "foo"}, nil, []string{"/", "", "/bar", "/foo/", "/fooooo", "//foo"}, false},
		{"foo slash", []string{"/foo/"}, []string{"/foo/"}, nil, []string{"/", "", "/bar", "/foo", "/fooooo"}, false},
		{"foo slash star", []string{"/foo/*"}, []string{"/foo/", "/foo/bar/bla"}, nil, []string{"/", "", "/foo", "/bar", "/fooooo"}, false},
		{"foo bar", []string{"/foo", "/bar"}, []string{"/foo", "/bar"}, nil, []string{"/", "", "/foo/", "/bar/", "/fooooo"}, false},
		{"foo star", []string{"/foo*"}, []string{"/foo", "/foooo"}, nil, []string{"/", "", "/fo", "/bar"}, false},
		{"star", []string{"/*"}, []string{"/", "", "/foo", "/foooo"}, nil, nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := NewAuthorizer(tt.excludedPaths)
			if err != nil && !tt.wantErr {
				t.Fatalf("unexpected error: %v", err)
			}
			if err == nil && tt.wantErr {
				t.Fatalf("expected error, didn't get any")
			}
			if err != nil {
				return
			}

			for _, cases := range []struct {
				paths []string
				want  authorizer.Decision
			}{
				{tt.allowed, authorizer.DecisionAllow},
				{tt.denied, authorizer.DecisionDeny},
				{tt.noOpinion, authorizer.DecisionNoOpinion},
			} {
				for _, pth := range cases.paths {
					info := authorizer.AttributesRecord{
						Path: pth,
					}
					if got, _, err := a.Authorize(info); err != nil {
						t.Errorf("NewAuthorizer(%v).Authorize(%q) return unexpected error: %v", tt.excludedPaths, pth, err)
					} else if got != cases.want {
						t.Errorf("NewAuthorizer(%v).Authorize(%q) = %v, want %v", tt.excludedPaths, pth, got, cases.want)
					}
				}
			}
		})
	}
}
