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

package group

import (
	"net/http"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
)

func TestGroupAdder(t *testing.T) {
	adder := authenticator.Request(
		NewGroupAdder(
			authenticator.RequestFunc(func(req *http.Request) (user.Info, bool, error) {
				return &user.DefaultInfo{Name: "user", Groups: []string{"original"}}, true, nil
			}),
			[]string{"added"},
		),
	)

	user, _, _ := adder.AuthenticateRequest(nil)
	if !reflect.DeepEqual(user.GetGroups(), []string{"original", "added"}) {
		t.Errorf("Expected original,added groups, got %#v", user.GetGroups())
	}
}
