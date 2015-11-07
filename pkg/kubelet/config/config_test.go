/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package config

import (
	"sort"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/conversion"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/types"
)

const (
	TestSource = "test"
)

func expectEmptyChannel(t *testing.T, ch <-chan interface{}) {
	select {
	case update := <-ch:
		t.Errorf("Expected no update in channel, Got %v", update)
	default:
	}
}

type sortedPods []*api.Pod

func (s sortedPods) Len() int {
	return len(s)
}
func (s sortedPods) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s sortedPods) Less(i, j int) bool {
	return s[i].Namespace < s[j].Namespace
}

func CreateValidPod(name, namespace string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       types.UID(name), // for the purpose of testing, this is unique enough
			Name:      name,
			Namespace: namespace,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
			Containers: []api.Container{
				{
					Name:            "ctr",
					Image:           "image",
					ImagePullPolicy: "IfNotPresent",
					SecurityContext: securitycontext.ValidSecurityContextWithContainerDefaults(),
				},
			},
		},
	}
}

func CreatePodUpdate(op kubetypes.PodOperation, source string, pods ...*api.Pod) kubetypes.PodUpdate {
	return kubetypes.PodUpdate{Pods: pods, Op: op, Source: source}
}

func createPodConfigTester(mode PodConfigNotificationMode) (chan<- interface{}, <-chan kubetypes.PodUpdate, *PodConfig) {
	eventBroadcaster := record.NewBroadcaster()
	config := NewPodConfig(mode, eventBroadcaster.NewRecorder(api.EventSource{Component: "kubelet"}))
	channel := config.Channel(TestSource)
	ch := config.Updates()
	return channel, ch, config
}

func expectPodUpdate(t *testing.T, ch <-chan kubetypes.PodUpdate, expected ...kubetypes.PodUpdate) {
	for i := range expected {
		update := <-ch
		sort.Sort(sortedPods(update.Pods))
		sort.Sort(sortedPods(expected[i].Pods))
		// Make copies of the expected/actual update to compare all fields
		// except for "Pods", which are compared separately below.
		expectedCopy, updateCopy := expected[i], update
		expectedCopy.Pods, updateCopy.Pods = nil, nil
		if !api.Semantic.DeepEqual(expectedCopy, updateCopy) {
			t.Fatalf("Expected %#v, Got %#v", expectedCopy, updateCopy)
		}

		if len(expected[i].Pods) != len(update.Pods) {
			t.Fatalf("Expected %#v, Got %#v", expected[i], update)
		}
		// Compare pods one by one. This is necessary beacuse we don't want to
		// compare local annotations.
		for j := range expected[i].Pods {
			if podsDifferSemantically(expected[i].Pods[j], update.Pods[j]) {
				t.Fatalf("Expected %#v, Got %#v", expected[i].Pods[j], update.Pods[j])
			}
		}
	}
	expectNoPodUpdate(t, ch)
}

func expectNoPodUpdate(t *testing.T, ch <-chan kubetypes.PodUpdate) {
	select {
	case update := <-ch:
		t.Errorf("Expected no update in channel, Got %#v", update)
	default:
	}
}

func TestNewPodAdded(t *testing.T) {
	channel, ch, config := createPodConfigTester(PodConfigNotificationIncremental)

	// see an update
	podUpdate := CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new")))

	config.Sync()
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.SET, kubetypes.AllSource, CreateValidPod("foo", "new")))
}

func TestNewPodAddedInvalidNamespace(t *testing.T) {
	channel, ch, config := createPodConfigTester(PodConfigNotificationIncremental)

	// see an update
	podUpdate := CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", ""))
	channel <- podUpdate
	config.Sync()
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.SET, kubetypes.AllSource))
}

func TestNewPodAddedDefaultNamespace(t *testing.T) {
	channel, ch, config := createPodConfigTester(PodConfigNotificationIncremental)

	// see an update
	podUpdate := CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "default"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "default")))

	config.Sync()
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.SET, kubetypes.AllSource, CreateValidPod("foo", "default")))
}

func TestNewPodAddedDifferentNamespaces(t *testing.T) {
	channel, ch, config := createPodConfigTester(PodConfigNotificationIncremental)

	// see an update
	podUpdate := CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "default"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "default")))

	// see an update in another namespace
	podUpdate = CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new")))

	config.Sync()
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.SET, kubetypes.AllSource, CreateValidPod("foo", "default"), CreateValidPod("foo", "new")))
}

func TestInvalidPodFiltered(t *testing.T) {
	channel, ch, _ := createPodConfigTester(PodConfigNotificationIncremental)

	// see an update
	podUpdate := CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new")))

	// add an invalid update
	podUpdate = CreatePodUpdate(kubetypes.UPDATE, TestSource, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})
	channel <- podUpdate
	expectNoPodUpdate(t, ch)
}

func TestNewPodAddedSnapshotAndUpdates(t *testing.T) {
	channel, ch, config := createPodConfigTester(PodConfigNotificationSnapshotAndUpdates)

	// see an set
	podUpdate := CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.SET, TestSource, CreateValidPod("foo", "new")))

	config.Sync()
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.SET, kubetypes.AllSource, CreateValidPod("foo", "new")))

	// container updates are separated as UPDATE
	pod := *podUpdate.Pods[0]
	pod.Spec.Containers = []api.Container{{Name: "bar", Image: "test", ImagePullPolicy: api.PullIfNotPresent}}
	channel <- CreatePodUpdate(kubetypes.ADD, TestSource, &pod)
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.UPDATE, TestSource, &pod))
}

func TestNewPodAddedSnapshot(t *testing.T) {
	channel, ch, config := createPodConfigTester(PodConfigNotificationSnapshot)

	// see an set
	podUpdate := CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.SET, TestSource, CreateValidPod("foo", "new")))

	config.Sync()
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.SET, kubetypes.AllSource, CreateValidPod("foo", "new")))

	// container updates are separated as UPDATE
	pod := *podUpdate.Pods[0]
	pod.Spec.Containers = []api.Container{{Name: "bar", Image: "test", ImagePullPolicy: api.PullIfNotPresent}}
	channel <- CreatePodUpdate(kubetypes.ADD, TestSource, &pod)
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.SET, TestSource, &pod))
}

func TestNewPodAddedUpdatedRemoved(t *testing.T) {
	channel, ch, _ := createPodConfigTester(PodConfigNotificationIncremental)

	// should register an add
	podUpdate := CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new")))

	// should ignore ADDs that are identical
	expectNoPodUpdate(t, ch)

	// an kubetypes.ADD should be converted to kubetypes.UPDATE
	pod := CreateValidPod("foo", "new")
	pod.Spec.Containers = []api.Container{{Name: "bar", Image: "test", ImagePullPolicy: api.PullIfNotPresent}}
	podUpdate = CreatePodUpdate(kubetypes.ADD, TestSource, pod)
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.UPDATE, TestSource, pod))

	podUpdate = CreatePodUpdate(kubetypes.REMOVE, TestSource, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "new"}})
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.REMOVE, TestSource, pod))
}

func TestNewPodAddedUpdatedSet(t *testing.T) {
	channel, ch, _ := createPodConfigTester(PodConfigNotificationIncremental)

	// should register an add
	podUpdate := CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new"), CreateValidPod("foo2", "new"), CreateValidPod("foo3", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new"), CreateValidPod("foo2", "new"), CreateValidPod("foo3", "new")))

	// should ignore ADDs that are identical
	expectNoPodUpdate(t, ch)

	// should be converted to an kubetypes.ADD, kubetypes.REMOVE, and kubetypes.UPDATE
	pod := CreateValidPod("foo2", "new")
	pod.Spec.Containers = []api.Container{{Name: "bar", Image: "test", ImagePullPolicy: api.PullIfNotPresent}}
	podUpdate = CreatePodUpdate(kubetypes.SET, TestSource, pod, CreateValidPod("foo3", "new"), CreateValidPod("foo4", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch,
		CreatePodUpdate(kubetypes.REMOVE, TestSource, CreateValidPod("foo", "new")),
		CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo4", "new")),
		CreatePodUpdate(kubetypes.UPDATE, TestSource, pod))
}

func TestInitialEmptySet(t *testing.T) {
	for _, test := range []struct {
		mode PodConfigNotificationMode
		op   kubetypes.PodOperation
	}{
		{PodConfigNotificationIncremental, kubetypes.ADD},
		{PodConfigNotificationSnapshot, kubetypes.SET},
		{PodConfigNotificationSnapshotAndUpdates, kubetypes.SET},
	} {
		channel, ch, _ := createPodConfigTester(test.mode)

		// should register an empty PodUpdate operation
		podUpdate := CreatePodUpdate(kubetypes.SET, TestSource)
		channel <- podUpdate
		expectPodUpdate(t, ch, CreatePodUpdate(test.op, TestSource))

		// should ignore following empty sets
		podUpdate = CreatePodUpdate(kubetypes.SET, TestSource)
		channel <- podUpdate
		podUpdate = CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo", "new"))
		channel <- podUpdate
		expectPodUpdate(t, ch, CreatePodUpdate(test.op, TestSource, CreateValidPod("foo", "new")))
	}
}

func TestPodUpdateAnnotations(t *testing.T) {
	channel, ch, _ := createPodConfigTester(PodConfigNotificationIncremental)

	pod := CreateValidPod("foo2", "new")
	pod.Annotations = make(map[string]string, 0)
	pod.Annotations["kubernetes.io/blah"] = "blah"

	clone, err := conversion.NewCloner().DeepCopy(pod)
	if err != nil {
		t.Fatalf("%v", err)
	}

	podUpdate := CreatePodUpdate(kubetypes.SET, TestSource, CreateValidPod("foo1", "new"), clone.(*api.Pod), CreateValidPod("foo3", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.ADD, TestSource, CreateValidPod("foo1", "new"), pod, CreateValidPod("foo3", "new")))

	pod.Annotations["kubenetes.io/blah"] = "superblah"
	podUpdate = CreatePodUpdate(kubetypes.SET, TestSource, CreateValidPod("foo1", "new"), pod, CreateValidPod("foo3", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.UPDATE, TestSource, pod))

	pod.Annotations["kubernetes.io/otherblah"] = "doh"
	podUpdate = CreatePodUpdate(kubetypes.SET, TestSource, CreateValidPod("foo1", "new"), pod, CreateValidPod("foo3", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.UPDATE, TestSource, pod))

	delete(pod.Annotations, "kubernetes.io/blah")
	podUpdate = CreatePodUpdate(kubetypes.SET, TestSource, CreateValidPod("foo1", "new"), pod, CreateValidPod("foo3", "new"))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.UPDATE, TestSource, pod))
}

func TestPodUpdateLables(t *testing.T) {
	channel, ch, _ := createPodConfigTester(PodConfigNotificationIncremental)

	pod := CreateValidPod("foo2", "new")
	pod.Labels = make(map[string]string, 0)
	pod.Labels["key"] = "value"

	clone, err := conversion.NewCloner().DeepCopy(pod)
	if err != nil {
		t.Fatalf("%v", err)
	}

	podUpdate := CreatePodUpdate(kubetypes.SET, TestSource, clone.(*api.Pod))
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.ADD, TestSource, pod))

	pod.Labels["key"] = "newValue"
	podUpdate = CreatePodUpdate(kubetypes.SET, TestSource, pod)
	channel <- podUpdate
	expectPodUpdate(t, ch, CreatePodUpdate(kubetypes.UPDATE, TestSource, pod))

}
