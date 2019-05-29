/*
Copyright 2014 The Kubernetes Authors.

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

package request

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
)

// TestNamespaceContext validates that a namespace can be get/set on a context object
func TestNamespaceContext(t *testing.T) {
	ctx := NewDefaultContext()
	result, ok := NamespaceFrom(ctx)
	if !ok {
		t.Fatalf("Error getting namespace")
	}
	if metav1.NamespaceDefault != result {
		t.Fatalf("Expected: %s, Actual: %s", metav1.NamespaceDefault, result)
	}

	ctx = NewContext()
	result, ok = NamespaceFrom(ctx)
	if ok {
		t.Fatalf("Should not be ok because there is no namespace on the context")
	}
}

//TestUserContext validates that a userinfo can be get/set on a context object
func TestUserContext(t *testing.T) {
	ctx := NewContext()
	_, ok := UserFrom(ctx)
	if ok {
		t.Fatalf("Should not be ok because there is no user.Info on the context")
	}
	ctx = WithUser(
		ctx,
		&user.DefaultInfo{
			Name:   "bob",
			UID:    "123",
			Groups: []string{"group1"},
			Extra:  map[string][]string{"foo": {"bar"}},
		},
	)

	result, ok := UserFrom(ctx)
	if !ok {
		t.Fatalf("Error getting user info")
	}

	expectedName := "bob"
	if result.GetName() != expectedName {
		t.Fatalf("Get user name error, Expected: %s, Actual: %s", expectedName, result.GetName())
	}

	expectedUID := "123"
	if result.GetUID() != expectedUID {
		t.Fatalf("Get UID error, Expected: %s, Actual: %s", expectedUID, result.GetName())
	}

	expectedGroup := "group1"
	actualGroup := result.GetGroups()
	if len(actualGroup) != 1 {
		t.Fatalf("Get user group number error, Expected: 1, Actual: %d", len(actualGroup))
	} else if actualGroup[0] != expectedGroup {
		t.Fatalf("Get user group error, Expected: %s, Actual: %s", expectedGroup, actualGroup[0])
	}

	expectedExtraKey := "foo"
	expectedExtraValue := "bar"
	actualExtra := result.GetExtra()
	if len(actualExtra[expectedExtraKey]) != 1 {
		t.Fatalf("Get user extra map number error, Expected: 1, Actual: %d", len(actualExtra[expectedExtraKey]))
	} else if actualExtra[expectedExtraKey][0] != expectedExtraValue {
		t.Fatalf("Get user extra map value error, Expected: %s, Actual: %s", expectedExtraValue, actualExtra[expectedExtraKey])
	}

}
