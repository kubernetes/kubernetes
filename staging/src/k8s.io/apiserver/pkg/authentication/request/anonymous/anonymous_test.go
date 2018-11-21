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
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

func TestAnonymous(t *testing.T) {
	var a authenticator.Request = NewAuthenticator()
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
