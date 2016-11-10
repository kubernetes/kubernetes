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

package gc

import (
	"testing"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/runtime"
)

type fakeAuthorizer struct{}

func (fakeAuthorizer) Authorize(a authorizer.Attributes) (bool, string, error) {
	username := a.GetUser().GetName()

	if username == "non-deleter" {
		if a.GetVerb() == "delete" {
			return false, "", nil
		}
		return true, "", nil
	}

	if username == "non-pod-deleter" {
		if a.GetVerb() == "delete" && a.GetResource() == "pods" {
			return false, "", nil
		}
		return true, "", nil
	}

	return true, "", nil
}

func TestGCAdmission(t *testing.T) {
	tests := []struct {
		name     string
		username string
		resource unversioned.GroupVersionResource
		oldObj   runtime.Object
		newObj   runtime.Object

		expectedAllowed bool
	}{
		{
			name:            "super-user, create, no objectref change",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          &api.Pod{},
			expectedAllowed: true,
		},
		{
			name:            "super-user, create, objectref change",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			expectedAllowed: true,
		},
		{
			name:            "non-deleter, create, no objectref change",
			username:        "non-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          &api.Pod{},
			expectedAllowed: true,
		},
		{
			name:            "non-deleter, create, objectref change",
			username:        "non-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			expectedAllowed: false,
		},
		{
			name:            "non-pod-deleter, create, no objectref change",
			username:        "non-pod-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          &api.Pod{},
			expectedAllowed: true,
		},
		{
			name:            "non-pod-deleter, create, objectref change",
			username:        "non-pod-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			expectedAllowed: false,
		},
		{
			name:            "non-pod-deleter, create, objectref change, but not a pod",
			username:        "non-pod-deleter",
			resource:        api.SchemeGroupVersion.WithResource("not-pods"),
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			expectedAllowed: true,
		},

		{
			name:            "super-user, update, no objectref change",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{},
			newObj:          &api.Pod{},
			expectedAllowed: true,
		},
		{
			name:            "super-user, update, no objectref change two",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			expectedAllowed: true,
		},
		{
			name:            "super-user, update, objectref change",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{},
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			expectedAllowed: true,
		},
		{
			name:            "non-deleter, update, no objectref change",
			username:        "non-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{},
			newObj:          &api.Pod{},
			expectedAllowed: true,
		},
		{
			name:            "non-deleter, update, no objectref change two",
			username:        "non-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			expectedAllowed: true,
		},
		{
			name:            "non-deleter, update, objectref change",
			username:        "non-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{},
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			expectedAllowed: false,
		},
		{
			name:            "non-deleter, update, objectref change two",
			username:        "non-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}, {Name: "second"}}}},
			expectedAllowed: false,
		},
		{
			name:            "non-pod-deleter, update, no objectref change",
			username:        "non-pod-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{},
			newObj:          &api.Pod{},
			expectedAllowed: true,
		},
		{
			name:            "non-pod-deleter, update, objectref change",
			username:        "non-pod-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{},
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			expectedAllowed: false,
		},
		{
			name:            "non-pod-deleter, update, objectref change, but not a pod",
			username:        "non-pod-deleter",
			resource:        api.SchemeGroupVersion.WithResource("not-pods"),
			oldObj:          &api.Pod{},
			newObj:          &api.Pod{ObjectMeta: api.ObjectMeta{OwnerReferences: []api.OwnerReference{{Name: "first"}}}},
			expectedAllowed: true,
		},
	}
	gcAdmit := &gcPermissionsEnforcement{
		Handler:    admission.NewHandler(admission.Create, admission.Update),
		authorizer: fakeAuthorizer{},
	}

	for _, tc := range tests {
		operation := admission.Create
		if tc.oldObj != nil {
			operation = admission.Update
		}
		user := &user.DefaultInfo{Name: tc.username}
		attributes := admission.NewAttributesRecord(tc.newObj, tc.oldObj, unversioned.GroupVersionKind{}, api.NamespaceDefault, "foo", tc.resource, "", operation, user)

		err := gcAdmit.Admit(attributes)
		switch {
		case err != nil && !tc.expectedAllowed:
		case err != nil && tc.expectedAllowed:
			t.Errorf("%v: unexpected err: %v", tc.name, err)
		case err == nil && !tc.expectedAllowed:
			t.Errorf("%v: missing err", tc.name)
		case err == nil && tc.expectedAllowed:
		}
	}
}
