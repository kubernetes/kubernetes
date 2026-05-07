/*
Copyright The Kubernetes Authors.

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

package apidefinitions

import (
	"errors"
	"testing"
)

func TestGroupNameForPackage(t *testing.T) {
	cases := []struct {
		name     string
		comments []string
		want     string
		wantErr  error
	}{
		{name: "groupName tag", comments: []string{"+groupName=apps"}, want: "apps"},
		{name: "qualified groupName", comments: []string{"+groupName=foo.example.com"}, want: "foo.example.com"},
		{name: "no tag", wantErr: ErrGroupUndeclared},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := GroupNameForPackage(tc.comments)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}
