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

package anytoken

import (
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
)

func TestAnyTokenAuthenticator(t *testing.T) {
	tests := []struct {
		name  string
		token string

		expectedUser user.Info
	}{
		{
			name:         "user only",
			token:        "joe",
			expectedUser: &user.DefaultInfo{Name: "joe"},
		},
		{
			name:         "user with slash",
			token:        "scheme/joe/",
			expectedUser: &user.DefaultInfo{Name: "scheme/joe"},
		},
		{
			name:         "user with groups",
			token:        "joe/group1,group2",
			expectedUser: &user.DefaultInfo{Name: "joe", Groups: []string{"group1", "group2"}},
		},
		{
			name:         "user with slash and groups",
			token:        "scheme/joe/group1,group2",
			expectedUser: &user.DefaultInfo{Name: "scheme/joe", Groups: []string{"group1", "group2"}},
		},
	}

	for _, tc := range tests {
		actualUser, _, _ := AnyTokenAuthenticator{}.AuthenticateToken(tc.token)

		if len(actualUser.GetExtra()) != 0 {
			t.Errorf("%q: got extra: %v", tc.name, actualUser.GetExtra())
		}
		if len(actualUser.GetUID()) != 0 {
			t.Errorf("%q: got extra: %v", tc.name, actualUser.GetUID())
		}
		if e, a := tc.expectedUser.GetName(), actualUser.GetName(); e != a {
			t.Errorf("%q: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expectedUser.GetGroups(), actualUser.GetGroups(); !reflect.DeepEqual(e, a) {
			t.Errorf("%q: expected %v, got %v", tc.name, e, a)
		}
	}
}
