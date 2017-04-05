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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/api"
	kubeadmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
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

	if username == "non-rc-deleter" {
		if a.GetVerb() == "delete" && a.GetResource() == "replicationcontrollers" {
			return false, "", nil
		}
		return true, "", nil
	}

	return true, "", nil
}

// newGCPermissionsEnforcement returns the admission controller configured for testing.
func newGCPermissionsEnforcement() *gcPermissionsEnforcement {
	gcAdmit := &gcPermissionsEnforcement{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
	pluginInitializer := kubeadmission.NewPluginInitializer(nil, nil, fakeAuthorizer{}, nil, api.Registry.RESTMapper())
	pluginInitializer.Initialize(gcAdmit)
	return gcAdmit
}

func TestGCAdmission(t *testing.T) {
	tests := []struct {
		name     string
		username string
		resource schema.GroupVersionResource
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
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
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
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
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
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			expectedAllowed: false,
		},
		{
			name:            "non-pod-deleter, create, objectref change, but not a pod",
			username:        "non-pod-deleter",
			resource:        api.SchemeGroupVersion.WithResource("not-pods"),
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
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
			oldObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			expectedAllowed: true,
		},
		{
			name:            "super-user, update, objectref change",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{},
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
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
			oldObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			expectedAllowed: true,
		},
		{
			name:            "non-deleter, update, objectref change",
			username:        "non-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{},
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			expectedAllowed: false,
		},
		{
			name:            "non-deleter, update, objectref change two",
			username:        "non-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}, {Name: "second"}}}},
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
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			expectedAllowed: false,
		},
		{
			name:            "non-pod-deleter, update, objectref change, but not a pod",
			username:        "non-pod-deleter",
			resource:        api.SchemeGroupVersion.WithResource("not-pods"),
			oldObj:          &api.Pod{},
			newObj:          &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			expectedAllowed: true,
		},
	}
	gcAdmit := newGCPermissionsEnforcement()

	for _, tc := range tests {
		operation := admission.Create
		if tc.oldObj != nil {
			operation = admission.Update
		}
		user := &user.DefaultInfo{Name: tc.username}
		attributes := admission.NewAttributesRecord(tc.newObj, tc.oldObj, schema.GroupVersionKind{}, metav1.NamespaceDefault, "foo", tc.resource, "", operation, user)

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

func TestBlockOwnerDeletionAdmission(t *testing.T) {
	podWithOwnerRefs := func(refs ...metav1.OwnerReference) *api.Pod {
		var refSlice []metav1.OwnerReference
		for _, ref := range refs {
			refSlice = append(refSlice, ref)
		}
		return &api.Pod{
			ObjectMeta: metav1.ObjectMeta{
				OwnerReferences: refSlice,
			},
		}
	}

	getTrueVar := func() *bool {
		ret := true
		return &ret
	}

	getFalseVar := func() *bool {
		ret := false
		return &ret
	}
	blockRC1 := metav1.OwnerReference{
		APIVersion:         "v1",
		Kind:               "ReplicationController",
		Name:               "rc1",
		BlockOwnerDeletion: getTrueVar(),
	}
	blockRC2 := metav1.OwnerReference{
		APIVersion:         "v1",
		Kind:               "ReplicationController",
		Name:               "rc2",
		BlockOwnerDeletion: getTrueVar(),
	}
	notBlockRC1 := metav1.OwnerReference{
		APIVersion:         "v1",
		Kind:               "ReplicationController",
		Name:               "rc1",
		BlockOwnerDeletion: getFalseVar(),
	}
	notBlockRC2 := metav1.OwnerReference{
		APIVersion:         "v1",
		Kind:               "ReplicationController",
		Name:               "rc2",
		BlockOwnerDeletion: getFalseVar(),
	}
	nilBlockRC1 := metav1.OwnerReference{
		APIVersion: "v1",
		Kind:       "ReplicationController",
		Name:       "rc1",
	}
	nilBlockRC2 := metav1.OwnerReference{
		APIVersion: "v1",
		Kind:       "ReplicationController",
		Name:       "rc2",
	}
	blockDS1 := metav1.OwnerReference{
		APIVersion:         "v1",
		Kind:               "DaemonSet",
		Name:               "ds1",
		BlockOwnerDeletion: getTrueVar(),
	}
	notBlockDS1 := metav1.OwnerReference{
		APIVersion:         "v1",
		Kind:               "DaemonSet",
		Name:               "ds1",
		BlockOwnerDeletion: getFalseVar(),
	}
	tests := []struct {
		name     string
		username string
		resource schema.GroupVersionResource
		oldObj   runtime.Object
		newObj   runtime.Object

		expectedAllowed bool
	}{
		// cases for create
		{
			name:            "super-user, create, no ownerReferences",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          podWithOwnerRefs(),
			expectedAllowed: true,
		},
		{
			name:            "super-user, create, all ownerReferences have blockOwnerDeletion=false",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          podWithOwnerRefs(notBlockRC1, notBlockRC2),
			expectedAllowed: true,
		},
		{
			name:            "super-user, create, some ownerReferences have blockOwnerDeletion=true",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          podWithOwnerRefs(blockRC1, blockRC2),
			expectedAllowed: true,
		},
		{
			name:            "non-rc-deleter, create, no ownerReferences",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          podWithOwnerRefs(),
			expectedAllowed: true,
		},
		{
			name:            "non-rc-deleter, create, all ownerReferences have blockOwnerDeletion=false or nil",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          podWithOwnerRefs(notBlockRC1, nilBlockRC2),
			expectedAllowed: true,
		},
		{
			name:            "non-rc-deleter, create, some ownerReferences have blockOwnerDeletion=true",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          podWithOwnerRefs(blockRC1, notBlockRC2),
			expectedAllowed: false,
		},
		{
			name:            "non-rc-deleter, create, some ownerReferences have blockOwnerDeletion=true, but are pointing to daemonset",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			newObj:          podWithOwnerRefs(blockDS1),
			expectedAllowed: true,
		},
		// cases are for update
		{
			name:            "super-user, update, no ownerReferences change blockOwnerDeletion",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(nilBlockRC1),
			newObj:          podWithOwnerRefs(notBlockRC1),
			expectedAllowed: true,
		},
		{
			name:            "super-user, update, some ownerReferences change to blockOwnerDeletion=true",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(notBlockRC1),
			newObj:          podWithOwnerRefs(blockRC1),
			expectedAllowed: true,
		},
		{
			name:            "super-user, update, add new ownerReferences with blockOwnerDeletion=true",
			username:        "super",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(),
			newObj:          podWithOwnerRefs(blockRC1),
			expectedAllowed: true,
		},
		{
			name:            "non-rc-deleter, update, no ownerReferences change blockOwnerDeletion",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(nilBlockRC1),
			newObj:          podWithOwnerRefs(notBlockRC1),
			expectedAllowed: true,
		},
		{
			name:            "non-rc-deleter, update, some ownerReferences change from blockOwnerDeletion=false to true",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(notBlockRC1),
			newObj:          podWithOwnerRefs(blockRC1),
			expectedAllowed: false,
		},
		{
			name:            "non-rc-deleter, update, some ownerReferences change from blockOwnerDeletion=nil to true",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(nilBlockRC1),
			newObj:          podWithOwnerRefs(blockRC1),
			expectedAllowed: false,
		},
		{
			name:            "non-rc-deleter, update, some ownerReferences change from blockOwnerDeletion=true to false",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(blockRC1),
			newObj:          podWithOwnerRefs(notBlockRC1),
			expectedAllowed: true,
		},
		{
			name:            "non-rc-deleter, update, some ownerReferences change blockOwnerDeletion, but all such references are to daemonset",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(notBlockDS1),
			newObj:          podWithOwnerRefs(blockDS1),
			expectedAllowed: true,
		},
		{
			name:            "non-rc-deleter, update, add new ownerReferences with blockOwnerDeletion=nil or false",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(),
			newObj:          podWithOwnerRefs(notBlockRC1, nilBlockRC2),
			expectedAllowed: true,
		},
		{
			name:            "non-rc-deleter, update, add new ownerReferences with blockOwnerDeletion=true",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(),
			newObj:          podWithOwnerRefs(blockRC1),
			expectedAllowed: false,
		},
		{
			name:            "non-rc-deleter, update, add new ownerReferences with blockOwnerDeletion=true, but the references are to daemonset",
			username:        "non-rc-deleter",
			resource:        api.SchemeGroupVersion.WithResource("pods"),
			oldObj:          podWithOwnerRefs(),
			newObj:          podWithOwnerRefs(blockDS1),
			expectedAllowed: true,
		},
	}

	gcAdmit := newGCPermissionsEnforcement()

	for _, tc := range tests {
		operation := admission.Create
		if tc.oldObj != nil {
			operation = admission.Update
		}
		user := &user.DefaultInfo{Name: tc.username}
		attributes := admission.NewAttributesRecord(tc.newObj, tc.oldObj, schema.GroupVersionKind{}, metav1.NamespaceDefault, "foo", tc.resource, "", operation, user)

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
