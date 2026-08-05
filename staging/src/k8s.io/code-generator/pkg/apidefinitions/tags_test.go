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
	"testing"
)

func TestGroupNameForPackage(t *testing.T) {
	cases := []struct {
		name     string
		comments []string
		want     string
		wantOK   bool
	}{
		{name: "groupName tag", comments: []string{"+groupName=apps"}, want: "apps", wantOK: true},
		{name: "qualified groupName", comments: []string{"+groupName=foo.example.com"}, want: "foo.example.com", wantOK: true},
		{name: "no tag"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok, err := GroupNameForPackage(tc.comments)
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if ok != tc.wantOK {
				t.Errorf("ok = %v, want %v", ok, tc.wantOK)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}
