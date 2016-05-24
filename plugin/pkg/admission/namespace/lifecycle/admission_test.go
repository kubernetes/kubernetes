/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package lifecycle

import (
	"fmt"
	"sync"
	"testing"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

// TestAdmission
func TestAdmission(t *testing.T) {
	namespaceObj := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:      "test",
			Namespace: "",
		},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceActive,
		},
	}
	var namespaceLock sync.RWMutex

	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	store.Add(namespaceObj)
	mockClient := fake.NewSimpleClientset()
	mockClient.PrependReactor("get", "namespaces", func(action core.Action) (bool, runtime.Object, error) {
		namespaceLock.RLock()
		defer namespaceLock.RUnlock()
		if getAction, ok := action.(testclient.GetAction); ok && getAction.GetName() == namespaceObj.Name {
			return true, namespaceObj, nil
		}
		return true, nil, fmt.Errorf("No result for action %v", action)
	})
	mockClient.PrependReactor("list", "namespaces", func(action core.Action) (bool, runtime.Object, error) {
		namespaceLock.RLock()
		defer namespaceLock.RUnlock()
		return true, &api.NamespaceList{Items: []api.Namespace{*namespaceObj}}, nil
	})

	lfhandler := NewLifecycle(mockClient, sets.NewString("default")).(*lifecycle)
	lfhandler.store = store
	handler := admission.NewChainHandler(lfhandler)
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespaceObj.Name},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
	badPod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "456", Namespace: "doesnotexist"},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
	err := handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler: %v", err)
	}

	// change namespace state to terminating
	namespaceLock.Lock()
	namespaceObj.Status.Phase = api.NamespaceTerminating
	namespaceLock.Unlock()
	store.Add(namespaceObj)

	// verify create operations in the namespace cause an error
	err = handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected error rejecting creates in a namespace when it is terminating")
	}

	// verify update operations in the namespace can proceed
	err = handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Update, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler: %v", err)
	}

	// verify delete operations in the namespace can proceed
	err = handler.Admit(admission.NewAttributesRecord(nil, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Delete, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler: %v", err)
	}

	// verify delete of namespace default can never proceed
	err = handler.Admit(admission.NewAttributesRecord(nil, nil, api.Kind("Namespace").WithVersion("version"), "", api.NamespaceDefault, api.Resource("namespaces").WithVersion("version"), "", admission.Delete, nil))
	if err == nil {
		t.Errorf("Expected an error that this namespace can never be deleted")
	}

	// verify delete of namespace other than default can proceed
	err = handler.Admit(admission.NewAttributesRecord(nil, nil, api.Kind("Namespace").WithVersion("version"), "", "other", api.Resource("namespaces").WithVersion("version"), "", admission.Delete, nil))
	if err != nil {
		t.Errorf("Did not expect an error %v", err)
	}

	// verify create/update/delete of object in non-existent namespace throws error
	err = handler.Admit(admission.NewAttributesRecord(&badPod, nil, api.Kind("Pod").WithVersion("version"), badPod.Namespace, badPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected, but didn't get, an error (%v) that objects cannot be created in non-existant namespaces", err)
	}

	err = handler.Admit(admission.NewAttributesRecord(&badPod, nil, api.Kind("Pod").WithVersion("version"), badPod.Namespace, badPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Update, nil))
	if err == nil {
		t.Errorf("Expected, but didn't get, an error (%v) that objects cannot be updated in non-existant namespaces", err)
	}

	err = handler.Admit(admission.NewAttributesRecord(&badPod, nil, api.Kind("Pod").WithVersion("version"), badPod.Namespace, badPod.Name, api.Resource("pods").WithVersion("version"), "", admission.Delete, nil))
	if err == nil {
		t.Errorf("Expected, but didn't get, an error (%v) that objects cannot be deleted in non-existant namespaces", err)
	}
}
