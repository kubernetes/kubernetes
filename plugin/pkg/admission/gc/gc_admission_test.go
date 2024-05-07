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
	"context"
	"fmt"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	fakediscovery "k8s.io/client-go/discovery/fake"
	"k8s.io/client-go/restmapper"
	coretesting "k8s.io/client-go/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubeadmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

type fakeAuthorizer struct{}

func (fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	username := a.GetUser().GetName()

	if username == "non-deleter" {
		if a.GetVerb() == "delete" {
			return authorizer.DecisionNoOpinion, "", nil
		}
		if a.GetVerb() == "update" && a.GetSubresource() == "finalizers" {
			return authorizer.DecisionNoOpinion, "", nil
		}
		if a.GetAPIGroup() == "*" && a.GetResource() == "*" { // this user does not have full rights
			return authorizer.DecisionNoOpinion, "", nil
		}
		return authorizer.DecisionAllow, "", nil
	}

	if username == "non-pod-deleter" {
		if a.GetVerb() == "delete" && a.GetResource() == "pods" {
			return authorizer.DecisionNoOpinion, "", nil
		}
		if a.GetVerb() == "update" && a.GetResource() == "pods" && a.GetSubresource() == "finalizers" {
			return authorizer.DecisionNoOpinion, "", nil
		}
		if a.GetAPIGroup() == "*" && a.GetResource() == "*" { // this user does not have full rights
			return authorizer.DecisionNoOpinion, "", nil
		}
		return authorizer.DecisionAllow, "", nil
	}

	if username == "non-rc-deleter" {
		if a.GetVerb() == "delete" && a.GetResource() == "replicationcontrollers" {
			return authorizer.DecisionNoOpinion, "", nil
		}
		if a.GetVerb() == "update" && a.GetResource() == "replicationcontrollers" && a.GetSubresource() == "finalizers" {
			return authorizer.DecisionNoOpinion, "", nil
		}
		if a.GetAPIGroup() == "*" && a.GetResource() == "*" { // this user does not have full rights
			return authorizer.DecisionNoOpinion, "", nil
		}
		return authorizer.DecisionAllow, "", nil
	}

	if username == "non-node-deleter" {
		if a.GetVerb() == "delete" && a.GetResource() == "nodes" {
			return authorizer.DecisionNoOpinion, "", nil
		}
		if a.GetVerb() == "update" && a.GetResource() == "nodes" && a.GetSubresource() == "finalizers" {
			return authorizer.DecisionNoOpinion, "", nil
		}
		if a.GetAPIGroup() == "*" && a.GetResource() == "*" { // this user does not have full rights
			return authorizer.DecisionNoOpinion, "", nil
		}
		return authorizer.DecisionAllow, "", nil
	}

	return authorizer.DecisionAllow, "", nil
}

// newGCPermissionsEnforcement returns the admission controller configured for testing.
func newGCPermissionsEnforcement() (*gcPermissionsEnforcement, error) {
	// the pods/status endpoint is ignored by this plugin since old kubelets
	// corrupt them.  the pod status strategy ensures status updates cannot mutate
	// ownerRef.
	whiteList := []whiteListItem{
		{
			groupResource: schema.GroupResource{Resource: "pods"},
			subresource:   "status",
		},
	}
	gcAdmit := &gcPermissionsEnforcement{
		Handler:   admission.NewHandler(admission.Create, admission.Update),
		whiteList: whiteList,
	}

	genericPluginInitializer := initializer.New(nil, nil, nil, fakeAuthorizer{}, nil, nil)
	fakeDiscoveryClient := &fakediscovery.FakeDiscovery{Fake: &coretesting.Fake{}}
	fakeDiscoveryClient.Resources = []*metav1.APIResourceList{
		{
			GroupVersion: corev1.SchemeGroupVersion.String(),
			APIResources: []metav1.APIResource{
				{Name: "nodes", Namespaced: false, Kind: "Node"},
				{Name: "pods", Namespaced: true, Kind: "Pod"},
				{Name: "replicationcontrollers", Namespaced: true, Kind: "ReplicationController"},
			},
		},
		{
			GroupVersion: appsv1.SchemeGroupVersion.String(),
			APIResources: []metav1.APIResource{
				{Name: "daemonsets", Namespaced: true, Kind: "DaemonSet"},
			},
		},
	}

	restMapperRes, err := restmapper.GetAPIGroupResources(fakeDiscoveryClient)
	if err != nil {
		return nil, fmt.Errorf("unexpected error while constructing resource list from fake discovery client: %v", err)
	}
	restMapper := restmapper.NewDiscoveryRESTMapper(restMapperRes)
	pluginInitializer := kubeadmission.NewPluginInitializer(nil, restMapper, nil, nil)
	initializersChain := admission.PluginInitializers{}
	initializersChain = append(initializersChain, genericPluginInitializer)
	initializersChain = append(initializersChain, pluginInitializer)

	initializersChain.Initialize(gcAdmit)
	return gcAdmit, nil
}

type neverReturningRESTMapper struct{}

var _ meta.RESTMapper = &neverReturningRESTMapper{}

func (r *neverReturningRESTMapper) KindFor(resource schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	// this ok because if the test works, this method should never be called.
	panic("test failed")
}
func (r *neverReturningRESTMapper) KindsFor(resource schema.GroupVersionResource) ([]schema.GroupVersionKind, error) {
	// this ok because if the test works, this method should never be called.
	panic("test failed")
}
func (r *neverReturningRESTMapper) ResourceFor(input schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	// this ok because if the test works, this method should never be called.
	panic("test failed")
}
func (r *neverReturningRESTMapper) ResourcesFor(input schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	// this ok because if the test works, this method should never be called.
	panic("test failed")
}
func (r *neverReturningRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	// this ok because if the test works, this method should never be called.
	panic("test failed")
}
func (r *neverReturningRESTMapper) RESTMappings(gk schema.GroupKind, versions ...string) ([]*meta.RESTMapping, error) {
	// this ok because if the test works, this method should never be called.
	panic("test failed")
}
func (r *neverReturningRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	// this ok because if the test works, this method should never be called.
	panic("test failed")
}

func TestGCAdmission(t *testing.T) {
	expectNoError := func(err error) bool {
		return err == nil
	}
	expectCantSetOwnerRefError := func(err error) bool {
		if err == nil {
			return false
		}
		return strings.Contains(err.Error(), "cannot set an ownerRef on a resource you can't delete")
	}
	tests := []struct {
		name        string
		username    string
		resource    schema.GroupVersionResource
		subresource string
		oldObj      runtime.Object
		newObj      runtime.Object

		checkError func(error) bool
	}{
		{
			name:       "super-user, create, no objectref change",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     &api.Pod{},
			checkError: expectNoError,
		},
		{
			name:       "super-user, create, objectref change",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError: expectNoError,
		},
		{
			name:       "non-deleter, create, no objectref change",
			username:   "non-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     &api.Pod{},
			checkError: expectNoError,
		},
		{
			name:       "non-deleter, create, objectref change",
			username:   "non-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError: expectNoError,
		},
		{
			name:       "non-pod-deleter, create, no objectref change",
			username:   "non-pod-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     &api.Pod{},
			checkError: expectNoError,
		},
		{
			name:       "non-pod-deleter, create, objectref change",
			username:   "non-pod-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError: expectNoError,
		},
		{
			name:       "non-pod-deleter, create, objectref change, but not a pod",
			username:   "non-pod-deleter",
			resource:   api.SchemeGroupVersion.WithResource("not-pods"),
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError: expectNoError,
		},

		{
			name:       "super-user, update, no objectref change",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     &api.Pod{},
			newObj:     &api.Pod{},
			checkError: expectNoError,
		},
		{
			name:       "super-user, update, no objectref change two",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError: expectNoError,
		},
		{
			name:       "super-user, update, objectref change",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     &api.Pod{},
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError: expectNoError,
		},
		{
			name:       "non-deleter, update, no objectref change",
			username:   "non-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     &api.Pod{},
			newObj:     &api.Pod{},
			checkError: expectNoError,
		},
		{
			name:       "non-deleter, update, no objectref change two",
			username:   "non-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError: expectNoError,
		},
		{
			name:       "non-deleter, update, objectref change",
			username:   "non-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     &api.Pod{},
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError: expectCantSetOwnerRefError,
		},
		{
			name:       "non-deleter, update, objectref change two",
			username:   "non-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}, {Name: "second"}}}},
			checkError: expectCantSetOwnerRefError,
		},
		{
			name:       "non-pod-deleter, update, no objectref change",
			username:   "non-pod-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     &api.Pod{},
			newObj:     &api.Pod{},
			checkError: expectNoError,
		},
		{
			name:        "non-pod-deleter, update status, objectref change",
			username:    "non-pod-deleter",
			resource:    api.SchemeGroupVersion.WithResource("pods"),
			subresource: "status",
			oldObj:      &api.Pod{},
			newObj:      &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError:  expectNoError,
		},
		{
			name:       "non-pod-deleter, update, objectref change",
			username:   "non-pod-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     &api.Pod{},
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError: expectCantSetOwnerRefError,
		},
		{
			name:       "non-pod-deleter, update, objectref change, but not a pod",
			username:   "non-pod-deleter",
			resource:   api.SchemeGroupVersion.WithResource("not-pods"),
			oldObj:     &api.Pod{},
			newObj:     &api.Pod{ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{Name: "first"}}}},
			checkError: expectNoError,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gcAdmit, err := newGCPermissionsEnforcement()
			if err != nil {
				t.Error(err)
			}

			operation := admission.Create
			var options runtime.Object = &metav1.CreateOptions{}
			if tc.oldObj != nil {
				operation = admission.Update
				options = &metav1.UpdateOptions{}
			}
			user := &user.DefaultInfo{Name: tc.username}
			attributes := admission.NewAttributesRecord(tc.newObj, tc.oldObj, schema.GroupVersionKind{}, metav1.NamespaceDefault, "foo", tc.resource, tc.subresource, operation, options, false, user)

			err = gcAdmit.Validate(context.TODO(), attributes, nil)
			if !tc.checkError(err) {
				t.Errorf("unexpected err: %v", err)
			}
		})
	}
}

func TestBlockOwnerDeletionAdmission(t *testing.T) {
	podWithOwnerRefs := func(refs ...metav1.OwnerReference) *api.Pod {
		var refSlice []metav1.OwnerReference
		refSlice = append(refSlice, refs...)

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
		APIVersion:         "apps/v1",
		Kind:               "DaemonSet",
		Name:               "ds1",
		BlockOwnerDeletion: getTrueVar(),
	}
	notBlockDS1 := metav1.OwnerReference{
		APIVersion:         "apps/v1",
		Kind:               "DaemonSet",
		Name:               "ds1",
		BlockOwnerDeletion: getFalseVar(),
	}
	blockNode := metav1.OwnerReference{
		APIVersion:         "v1",
		Kind:               "Node",
		Name:               "node1",
		BlockOwnerDeletion: getTrueVar(),
	}
	notBlockNode := metav1.OwnerReference{
		APIVersion:         "v1",
		Kind:               "Node",
		Name:               "node",
		BlockOwnerDeletion: getFalseVar(),
	}
	nilBlockNode := metav1.OwnerReference{
		APIVersion: "v1",
		Kind:       "Node",
		Name:       "node",
	}

	expectNoError := func(err error) bool {
		return err == nil
	}
	expectCantSetBlockOwnerDeletionError := func(err error) bool {
		if err == nil {
			return false
		}
		return strings.Contains(err.Error(), "cannot set blockOwnerDeletion if an ownerReference refers to a resource you can't set finalizers on")
	}
	tests := []struct {
		name               string
		username           string
		resource           schema.GroupVersionResource
		subresource        string
		oldObj             runtime.Object
		newObj             runtime.Object
		restMapperOverride meta.RESTMapper

		checkError func(error) bool
	}{
		// cases for create
		{
			name:       "super-user, create, no ownerReferences",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     podWithOwnerRefs(),
			checkError: expectNoError,
		},
		{
			name:       "super-user, create, all ownerReferences have blockOwnerDeletion=false",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     podWithOwnerRefs(notBlockRC1, notBlockRC2),
			checkError: expectNoError,
		},
		{
			name:       "super-user, create, some ownerReferences have blockOwnerDeletion=true",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     podWithOwnerRefs(blockRC1, blockRC2, blockNode),
			checkError: expectNoError,
		},
		{
			name:               "super-user, create, some ownerReferences have blockOwnerDeletion=true, hangingRESTMapper",
			username:           "super",
			resource:           api.SchemeGroupVersion.WithResource("pods"),
			newObj:             podWithOwnerRefs(blockRC1, blockRC2, blockNode),
			restMapperOverride: &neverReturningRESTMapper{},
			checkError:         expectNoError,
		},
		{
			name:       "non-rc-deleter, create, no ownerReferences",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     podWithOwnerRefs(),
			checkError: expectNoError,
		},
		{
			name:       "non-rc-deleter, create, all ownerReferences have blockOwnerDeletion=false or nil",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     podWithOwnerRefs(notBlockRC1, nilBlockRC2),
			checkError: expectNoError,
		},
		{
			name:       "non-node-deleter, create, all ownerReferences have blockOwnerDeletion=false",
			username:   "non-node-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     podWithOwnerRefs(notBlockNode),
			checkError: expectNoError,
		},
		{
			name:       "non-rc-deleter, create, some ownerReferences have blockOwnerDeletion=true",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     podWithOwnerRefs(blockRC1, notBlockRC2),
			checkError: expectCantSetBlockOwnerDeletionError,
		},
		{
			name:       "non-rc-deleter, create, some ownerReferences have blockOwnerDeletion=true, but are pointing to daemonset",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     podWithOwnerRefs(blockDS1),
			checkError: expectNoError,
		},
		{
			name:       "non-node-deleter, create, some ownerReferences have blockOwnerDeletion=true",
			username:   "non-node-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			newObj:     podWithOwnerRefs(blockNode),
			checkError: expectCantSetBlockOwnerDeletionError,
		},
		// cases are for update
		{
			name:       "super-user, update, no ownerReferences change blockOwnerDeletion",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(nilBlockRC1, nilBlockNode),
			newObj:     podWithOwnerRefs(notBlockRC1, notBlockNode),
			checkError: expectNoError,
		},
		{
			name:       "super-user, update, some ownerReferences change to blockOwnerDeletion=true",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(notBlockRC1, notBlockNode),
			newObj:     podWithOwnerRefs(blockRC1, blockNode),
			checkError: expectNoError,
		},
		{
			name:       "super-user, update, add new ownerReferences with blockOwnerDeletion=true",
			username:   "super",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(),
			newObj:     podWithOwnerRefs(blockRC1, blockNode),
			checkError: expectNoError,
		},
		{
			name:       "non-rc-deleter, update, no ownerReferences change blockOwnerDeletion",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(nilBlockRC1),
			newObj:     podWithOwnerRefs(notBlockRC1),
			checkError: expectNoError,
		},
		{
			name:       "non-rc-deleter, update, some ownerReferences change from blockOwnerDeletion=false to true",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(notBlockRC1),
			newObj:     podWithOwnerRefs(blockRC1),
			checkError: expectCantSetBlockOwnerDeletionError,
		},
		{
			name:       "non-rc-deleter, update, some ownerReferences change from blockOwnerDeletion=nil to true",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(nilBlockRC1),
			newObj:     podWithOwnerRefs(blockRC1),
			checkError: expectCantSetBlockOwnerDeletionError,
		},
		{
			name:       "non-node-deleter, update, some ownerReferences change from blockOwnerDeletion=nil to true",
			username:   "non-node-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(nilBlockNode),
			newObj:     podWithOwnerRefs(blockNode),
			checkError: expectCantSetBlockOwnerDeletionError,
		},
		{
			name:       "non-rc-deleter, update, some ownerReferences change from blockOwnerDeletion=true to false",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(blockRC1),
			newObj:     podWithOwnerRefs(notBlockRC1),
			checkError: expectNoError,
		},
		{
			name:       "non-node-deleter, update, some ownerReferences change from blockOwnerDeletion=true to false",
			username:   "non-node-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(blockNode),
			newObj:     podWithOwnerRefs(notBlockNode),
			checkError: expectNoError,
		},
		{
			name:       "non-rc-deleter, update, some ownerReferences change blockOwnerDeletion, but all such references are to daemonset",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(notBlockDS1),
			newObj:     podWithOwnerRefs(blockDS1),
			checkError: expectNoError,
		},
		{
			name:       "non-rc-deleter, update, add new ownerReferences with blockOwnerDeletion=nil or false",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(),
			newObj:     podWithOwnerRefs(notBlockRC1, nilBlockRC2),
			checkError: expectNoError,
		},
		{
			name:       "non-rc-deleter, update, add new ownerReferences with blockOwnerDeletion=true",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(),
			newObj:     podWithOwnerRefs(blockRC1),
			checkError: expectCantSetBlockOwnerDeletionError,
		},
		{
			name:       "non-rc-deleter, update, add new ownerReferences with blockOwnerDeletion=true, but the references are to daemonset",
			username:   "non-rc-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(),
			newObj:     podWithOwnerRefs(blockDS1),
			checkError: expectNoError,
		},
		{
			name:       "non-node-deleter, update, add ownerReferences with blockOwnerDeletion=true",
			username:   "non-node-deleter",
			resource:   api.SchemeGroupVersion.WithResource("pods"),
			oldObj:     podWithOwnerRefs(),
			newObj:     podWithOwnerRefs(blockNode),
			checkError: expectCantSetBlockOwnerDeletionError,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gcAdmit, err := newGCPermissionsEnforcement()
			if err != nil {
				t.Fatal(err)
			}
			if tc.restMapperOverride != nil {
				gcAdmit.restMapper = tc.restMapperOverride
			}

			operation := admission.Create
			var options runtime.Object = &metav1.CreateOptions{}
			if tc.oldObj != nil {
				operation = admission.Update
				options = &metav1.UpdateOptions{}
			}
			user := &user.DefaultInfo{Name: tc.username}
			attributes := admission.NewAttributesRecord(tc.newObj, tc.oldObj, schema.GroupVersionKind{}, metav1.NamespaceDefault, "foo", tc.resource, tc.subresource, operation, options, false, user)

			err = gcAdmit.Validate(context.TODO(), attributes, nil)
			if !tc.checkError(err) {
				t.Fatal(err)
			}
		})
	}
}
