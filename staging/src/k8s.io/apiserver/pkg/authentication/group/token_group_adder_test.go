/*
Copyright 2017 The Kubernetes Authors.

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
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

func TestTokenGroupAdder(t *testing.T) {
	capacity := make([]string, 0, 1024)
	response := &authenticator.Response{User: &user.DefaultInfo{Name: "user", Groups: append(capacity, "original")}}
	orig := toJson(response)

	adder := authenticator.Token(
		NewTokenGroupAdder(
			authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
				return response, true, nil
			}),
			[]string{"added"},
		),
	)

	r, _, _ := adder.AuthenticateToken(context.Background(), "")
	if !reflect.DeepEqual(r.User.GetGroups(), []string{"original", "added"}) {
		t.Errorf("Expected original,added groups, got %#v", r.User.GetGroups())
	}

	if got := toJson(response); got != orig {
		t.Errorf("Expected response from delegate to be unmodified: orig=%v got=%v", orig, got)
	}
}

func toJson(x interface{}) string {
	b, _ := json.Marshal(x)
	return string(b)
}
