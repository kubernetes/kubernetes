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
	"testing"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestAnonymous(t *testing.T) {
	var a authenticator.Request = NewAuthenticator()
	u, ok, err := a.AuthenticateRequest(nil)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if !ok {
		t.Fatalf("Unexpectedly unauthenticated")
	}
	if u.GetName() != user.Anonymous {
		t.Fatalf("Expected username %s, got %s", user.Anonymous, u.GetName())
	}
	if !sets.NewString(u.GetGroups()...).Equal(sets.NewString(user.AllUnauthenticated)) {
		t.Fatalf("Expected group %s, got %v", user.AllUnauthenticated, u.GetGroups())
	}
}
