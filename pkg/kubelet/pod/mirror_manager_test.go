/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package pod

import (
	"fmt"
	"math/rand"
	"reflect"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
)

func newRandomStaticPod() *api.Pod {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:         types.UID(strconv.Itoa(rand.Int())),
			Name:        strconv.Itoa(rand.Int()),
			Namespace:   strconv.Itoa(rand.Int()),
			Annotations: make(map[string]string),
		},
	}
	pod.Annotations[kubetypes.ConfigSourceAnnotationKey] = kubetypes.FileSource
	pod.Annotations[kubetypes.ConfigHashAnnotationKey] = string(pod.UID)
	return pod
}

func modifiedRandomStaticPod(pod *api.Pod) *api.Pod {
	modified := newRandomStaticPod()
	modified.Name = pod.Name
	modified.Namespace = pod.Namespace
	return modified
}

func expectPod(t *testing.T, pod *api.Pod, expected *api.Pod) {
	if !reflect.DeepEqual(pod, expected) {
		t.Errorf("Expected pod: %+v, Got pod: %+v", expected, pod)
	}
}

func verifyChannel(manager *mirrorPodManager, expectedCreation, expectedDeletion int) error {
	numCreation := 0
	numDeletion := 0
	for {
		hasMessage := true
		select {
		case <-manager.mirrorPodCreationChannel:
			numCreation++
		case <-manager.mirrorPodDeletionChannel:
			numDeletion++
		default:
			hasMessage = false
		}
		if !hasMessage {
			break
		}
	}
	if numCreation != expectedCreation {
		return fmt.Errorf("Expected creation number %d, got %d", expectedCreation, numCreation)
	}
	if numDeletion != expectedDeletion {
		return fmt.Errorf("Expected deletion number %d, got %d", expectedDeletion, numDeletion)
	}
	return nil
}

func verifyActions(client clientset.Interface, expectedActions []core.Action) error {
	fakeClient := client.(*fake.Clientset)
	actions := fakeClient.Actions()
	if len(actions) != len(expectedActions) {
		return fmt.Errorf("Unexpected actions, got: %s expected: %s", actions, expectedActions)
	}
	for i := 0; i < len(actions); i++ {
		e := expectedActions[i]
		a := actions[i]
		if !a.Matches(e.GetVerb(), e.GetResource().Resource) || a.GetSubresource() != e.GetSubresource() {
			return fmt.Errorf("Unexpected actions, got: %s expected: %s", actions, expectedActions)
		}
	}
	fakeClient.ClearActions()
	return nil
}

func TestMirrorPods(t *testing.T) {
	mirrorPods := newMirrorPods()
	pods := [10]*api.Pod{}
	// Create all mirror pods
	for i := range pods {
		pods[i] = newMirrorPod(newRandomStaticPod())
		mirrorPods.setPod(getKey(pods[i]), pods[i])
	}
	// Should get all mirror pods
	for _, pod := range pods {
		key := getKey(pod)
		expectPod(t, mirrorPods.getPod(key), pod)
		expectPod(t, mirrorPods.getPodByUID(pod.UID), pod)
	}
	// Should not get mirror pod after deleted
	key := getKey(pods[0])
	mirrorPods.removePod(key)
	expectPod(t, mirrorPods.getPod(key), nil)
	expectPod(t, mirrorPods.getPodByUID(pods[0].UID), nil)
}

func TestMirrorPodCreationAndDeletion(t *testing.T) {
	staticPod := newRandomStaticPod()
	mirrorPod := newMirrorPod(staticPod)
	deleteActions := []core.Action{
		core.DeleteActionImpl{ActionImpl: core.ActionImpl{Verb: "delete", Resource: unversioned.GroupVersionResource{Resource: "pods"}}},
	}
	for c, test := range []struct {
		oldMirrorPod  *api.Pod
		getError      error
		createActions []core.Action
	}{
		{
			// Should recreate mirror pod if the old one is invalid
			oldMirrorPod: newMirrorPod(newRandomStaticPod()),
			getError:     nil,
			createActions: []core.Action{
				core.GetActionImpl{ActionImpl: core.ActionImpl{Verb: "get", Resource: unversioned.GroupVersionResource{Resource: "pods"}}},
				core.DeleteActionImpl{ActionImpl: core.ActionImpl{Verb: "delete", Resource: unversioned.GroupVersionResource{Resource: "pods"}}},
				core.CreateActionImpl{ActionImpl: core.ActionImpl{Verb: "create", Resource: unversioned.GroupVersionResource{Resource: "pods"}}},
			},
		},
		{
			// Should not create mirror pod if the old one is valid
			oldMirrorPod: mirrorPod,
			getError:     nil,
			createActions: []core.Action{
				core.GetActionImpl{ActionImpl: core.ActionImpl{Verb: "get", Resource: unversioned.GroupVersionResource{Resource: "pods"}}},
			},
		},
		{
			// Should create mirror pod if the old one not exists
			oldMirrorPod: nil,
			getError:     errors.NewNotFound(unversioned.GroupResource{Resource: "pods"}, staticPod.Name),
			createActions: []core.Action{
				core.GetActionImpl{ActionImpl: core.ActionImpl{Verb: "get", Resource: unversioned.GroupVersionResource{Resource: "pods"}}},
				core.CreateActionImpl{ActionImpl: core.ActionImpl{Verb: "create", Resource: unversioned.GroupVersionResource{Resource: "pods"}}},
			},
		},
	} {
		fakeClient := &fake.Clientset{}
		pm := newMirrorPodManager(fakeClient)
		fakeClient.AddReactor("get", "pods", func(action core.Action) (bool, runtime.Object, error) {
			return true, test.oldMirrorPod, test.getError
		})
		pm.addStaticPod(staticPod)
		pm.handleMirrorPodCreation(mirrorPod)
		assert.NoError(t, verifyActions(pm.apiserverClient, test.createActions), "case %d", c)

		pm.handleMirrorPodDeletion(mirrorPod)
		assert.NoError(t, verifyActions(pm.apiserverClient, deleteActions), "case %d", c)
	}
}

func TestChangeStaticPod(t *testing.T) {
	pm := newMirrorPodManager(&fake.Clientset{})
	// Add static pod, a creation should be triggered
	staticPod := newRandomStaticPod()
	pm.addStaticPod(staticPod)
	assert.NoError(t, verifyChannel(pm, 1, 0))

	// Remove an existing static pod, a deletion should be triggered
	pm.deleteStaticPod(staticPod)
	assert.NoError(t, verifyChannel(pm, 0, 1))
}

func TestChangeMirrorPod(t *testing.T) {
	pm := newMirrorPodManager(&fake.Clientset{})
	// Add a new static pod
	staticPod := newRandomStaticPod()
	pm.addStaticPod(staticPod)
	assert.NoError(t, verifyChannel(pm, 1, 0))
	// The mirror pod hasn't been created yet, a mirror pod creation should be triggered in cleanup
	pm.handleCleanup()
	assert.NoError(t, verifyChannel(pm, 1, 0))

	// Add a changed mirror pod, a recreation should be triggered
	mirrorPod := newMirrorPod(modifiedRandomStaticPod(staticPod))
	pm.AddMirrorPod(mirrorPod)
	pm.handleCleanup()
	assert.NoError(t, verifyChannel(pm, 1, 0))

	// Add a mirror pod with DeletionTimestamp, a recreation should be triggered
	now := unversioned.Now()
	mirrorPod.DeletionTimestamp = &now
	pm.AddMirrorPod(mirrorPod)
	pm.handleCleanup()
	assert.NoError(t, verifyChannel(pm, 1, 0))

	// Delete an existing mirror pod, a recreation should be triggered
	pm.DeleteMirrorPod(mirrorPod)
	pm.handleCleanup()
	assert.NoError(t, verifyChannel(pm, 1, 0))

	// Add an unchanged mirror pod, nothing should be triggered
	mirrorPod = newMirrorPod(staticPod)
	pm.AddMirrorPod(mirrorPod)
	pm.handleCleanup()
	assert.NoError(t, verifyChannel(pm, 0, 0))

	// Add a mirror pod without corresponding static pod, a deletion should be triggered
	mirrorPod = newMirrorPod(newRandomStaticPod())
	pm.AddMirrorPod(mirrorPod)
	pm.handleCleanup()
	assert.NoError(t, verifyChannel(pm, 0, 1))

	// Delete a mirror pod without corresponding static pod, nothing should be triggered
	pm.DeleteMirrorPod(mirrorPod)
	pm.handleCleanup()
	assert.NoError(t, verifyChannel(pm, 0, 0))

	// Delete the static pod
	pm.deleteStaticPod(staticPod)
	assert.NoError(t, verifyChannel(pm, 0, 1))
	// The mirror pod hasn't been deleted yet, a mirror pod deletion should be triggered in cleanup
	pm.handleCleanup()
	assert.NoError(t, verifyChannel(pm, 0, 1))
}

func TestTranslatePodUID(t *testing.T) {
	pm := newMirrorPodManager(&fake.Clientset{})
	staticPod := newRandomStaticPod()
	pm.addStaticPod(staticPod)
	assert.NoError(t, verifyChannel(pm, 1, 0))

	mirrorPod := newMirrorPod(staticPod)
	// Change the mirror pod uid manually to test whether TranslatePodUID() works well.
	mirrorPod.UID = types.UID(strconv.Itoa(rand.Int()))
	pm.AddMirrorPod(mirrorPod)
	if uid := pm.TranslatePodUID(mirrorPod.UID); uid != staticPod.UID {
		t.Errorf("Should translate from mirror pod uid to static pod uid, expected %q, got %q", staticPod.UID, uid)
	}

	pm.DeleteMirrorPod(mirrorPod)
	if uid := pm.TranslatePodUID(mirrorPod.UID); uid != mirrorPod.UID {
		t.Errorf("Should not translate unknown uid, expected %q, got %q", mirrorPod.UID, uid)
	}
}
