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

package util

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
)

func TestResourceAttributesFrom(t *testing.T) {
	knownResourceAttributesNames := sets.NewString(
		// Fields we copy in ResourceAttributesFrom
		"Verb",
		"Namespace",
		"Group",
		"Version",
		"Resource",
		"Subresource",
		"Name",

		// Fields we copy in NonResourceAttributesFrom
		"Path",
		"Verb",
	)
	reflect.TypeOf(authorizationapi.ResourceAttributes{}).FieldByNameFunc(func(name string) bool {
		if !knownResourceAttributesNames.Has(name) {
			t.Errorf("authorizationapi.ResourceAttributes has a new field: %q. Add to ResourceAttributesFrom/NonResourceAttributesFrom as appropriate, then add to knownResourceAttributesNames", name)
		}
		return false
	})

	knownAttributesRecordFieldNames := sets.NewString(
		// Fields we set in ResourceAttributesFrom
		"User",
		"Verb",
		"Namespace",
		"APIGroup",
		"APIVersion",
		"Resource",
		"Subresource",
		"Name",
		"ResourceRequest",

		// Fields we set in NonResourceAttributesFrom
		"User",
		"ResourceRequest",
		"Path",
		"Verb",
	)
	reflect.TypeOf(authorizer.AttributesRecord{}).FieldByNameFunc(func(name string) bool {
		if !knownAttributesRecordFieldNames.Has(name) {
			t.Errorf("authorizer.AttributesRecord has a new field: %q. Add to ResourceAttributesFrom/NonResourceAttributesFrom as appropriate, then add to knownAttributesRecordFieldNames", name)
		}
		return false
	})
}

func TestAuthorizationAttributesFrom(t *testing.T) {
	type args struct {
		spec authorizationapi.SubjectAccessReviewSpec
	}
	tests := []struct {
		name string
		args args
		want authorizer.AttributesRecord
	}{
		{
			name: "nonresource",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					User:                  "bob",
					Groups:                []string{user.AllAuthenticated},
					NonResourceAttributes: &authorizationapi.NonResourceAttributes{Verb: "get", Path: "/mypath"},
					Extra:                 map[string]authorizationapi.ExtraValue{"scopes": {"scope-a", "scope-b"}},
				},
			},
			want: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name:   "bob",
					Groups: []string{user.AllAuthenticated},
					Extra:  map[string][]string{"scopes": {"scope-a", "scope-b"}},
				},
				Verb: "get",
				Path: "/mypath",
			},
		},
		{
			name: "resource",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					User: "bob",
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Namespace:   "myns",
						Verb:        "create",
						Group:       "extensions",
						Version:     "v1beta1",
						Resource:    "deployments",
						Subresource: "scale",
						Name:        "mydeployment",
					},
				},
			},
			want: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name: "bob",
				},
				APIGroup:        "extensions",
				APIVersion:      "v1beta1",
				Namespace:       "myns",
				Verb:            "create",
				Resource:        "deployments",
				Subresource:     "scale",
				Name:            "mydeployment",
				ResourceRequest: true,
			},
		},
		{
			name: "resource with no version",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					User: "bob",
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Namespace:   "myns",
						Verb:        "create",
						Group:       "extensions",
						Resource:    "deployments",
						Subresource: "scale",
						Name:        "mydeployment",
					},
				},
			},
			want: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name: "bob",
				},
				APIGroup:        "extensions",
				APIVersion:      "*",
				Namespace:       "myns",
				Verb:            "create",
				Resource:        "deployments",
				Subresource:     "scale",
				Name:            "mydeployment",
				ResourceRequest: true,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := AuthorizationAttributesFrom(tt.args.spec); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("AuthorizationAttributesFrom() = %v, want %v", got, tt.want)
			}
		})
	}
}
