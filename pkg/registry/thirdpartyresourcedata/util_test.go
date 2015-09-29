/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package thirdpartyresourcedata

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/experimental"
)

func TestExtractAPIGroupAndKind(t *testing.T) {
	tests := []struct {
		input         string
		expectedKind  string
		expectedGroup string
		expectErr     bool
	}{
		{
			input:         "foo.company.com",
			expectedKind:  "Foo",
			expectedGroup: "company.com",
		},
		{
			input:         "cron-tab.company.com",
			expectedKind:  "CronTab",
			expectedGroup: "company.com",
		},
		{
			input:     "foo",
			expectErr: true,
		},
	}

	for _, test := range tests {
		kind, group, err := ExtractApiGroupAndKind(&experimental.ThirdPartyResource{ObjectMeta: api.ObjectMeta{Name: test.input}})
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if err == nil && test.expectErr {
			t.Errorf("unexpected non-error")
			continue
		}
		if kind != test.expectedKind {
			t.Errorf("expected: %s, saw: %s", test.expectedKind, kind)
		}
		if group != test.expectedGroup {
			t.Errorf("expected: %s, saw: %s", test.expectedGroup, group)
		}
	}
}
