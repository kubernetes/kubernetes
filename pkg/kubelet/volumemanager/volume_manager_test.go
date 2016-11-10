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

package volumemanager

import (
	"os"
	"reflect"
	"strconv"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/config"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/pod"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	podtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/sets"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

const (
	testHostname = "test-hostname"
)

func TestGetMountedVolumesForPodAndGetVolumesInUse(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("volumeManagerTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	podManager := kubepod.NewBasicPodManager(podtest.NewFakeMirrorClient())

	node, pod, pv, claim := createObjects()
	kubeClient := fake.NewSimpleClientset(node, pod, pv, claim)

	manager, err := newTestVolumeManager(tmpDir, podManager, kubeClient)
	if err != nil {
		t.Fatalf("Failed to initialize volume manager: %v", err)
	}

	stopCh := runVolumeManager(manager)
	defer close(stopCh)

	podManager.SetPods([]*api.Pod{pod})

	// Fake node status update
	go simulateVolumeInUseUpdate(
		api.UniqueVolumeName(node.Status.VolumesAttached[0].Name),
		stopCh,
		manager)

	err = manager.WaitForAttachAndMount(pod)
	if err != nil {
		t.Errorf("Expected success: %v", err)
	}

	expectedMounted := pod.Spec.Volumes[0].Name
	actualMounted := manager.GetMountedVolumesForPod(types.UniquePodName(pod.ObjectMeta.UID))
	if _, ok := actualMounted[expectedMounted]; !ok || (len(actualMounted) != 1) {
		t.Errorf("Expected %v to be mounted to pod but got %v", expectedMounted, actualMounted)
	}

	expectedInUse := []api.UniqueVolumeName{api.UniqueVolumeName(node.Status.VolumesAttached[0].Name)}
	actualInUse := manager.GetVolumesInUse()
	if !reflect.DeepEqual(expectedInUse, actualInUse) {
		t.Errorf("Expected %v to be in use but got %v", expectedInUse, actualInUse)
	}
}

func TestGetExtraSupplementalGroupsForPod(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("volumeManagerTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	podManager := kubepod.NewBasicPodManager(podtest.NewFakeMirrorClient())

	node, pod, _, claim := createObjects()

	existingGid := pod.Spec.SecurityContext.SupplementalGroups[0]

	cases := []struct {
		gidAnnotation string
		expected      []int64
	}{
		{
			gidAnnotation: "777",
			expected:      []int64{777},
		},
		{
			gidAnnotation: strconv.FormatInt(existingGid, 10),
			expected:      []int64{},
		},
		{
			gidAnnotation: "a",
			expected:      []int64{},
		},
		{
			gidAnnotation: "",
			expected:      []int64{},
		},
	}

	for _, tc := range cases {
		pv := &api.PersistentVolume{
			ObjectMeta: api.ObjectMeta{
				Name: "pvA",
				Annotations: map[string]string{
					volumehelper.VolumeGidAnnotationKey: tc.gidAnnotation,
				},
			},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
						PDName: "fake-device",
					},
				},
				ClaimRef: &api.ObjectReference{
					Name: claim.ObjectMeta.Name,
				},
			},
		}
		kubeClient := fake.NewSimpleClientset(node, pod, pv, claim)

		manager, err := newTestVolumeManager(tmpDir, podManager, kubeClient)
		if err != nil {
			t.Errorf("Failed to initialize volume manager: %v", err)
			continue
		}

		stopCh := runVolumeManager(manager)
		defer func() {
			close(stopCh)
		}()

		podManager.SetPods([]*api.Pod{pod})

		// Fake node status update
		go simulateVolumeInUseUpdate(
			api.UniqueVolumeName(node.Status.VolumesAttached[0].Name),
			stopCh,
			manager)

		err = manager.WaitForAttachAndMount(pod)
		if err != nil {
			t.Errorf("Expected success: %v", err)
			continue
		}

		actual := manager.GetExtraSupplementalGroupsForPod(pod)
		if !reflect.DeepEqual(tc.expected, actual) {
			t.Errorf("Expected supplemental groups %v, got %v", tc.expected, actual)
		}
	}
}

func newTestVolumeManager(
	tmpDir string,
	podManager pod.Manager,
	kubeClient internalclientset.Interface) (VolumeManager, error) {
	plug := &volumetest.FakeVolumePlugin{PluginName: "fake", Host: nil}
	fakeRecorder := &record.FakeRecorder{}
	plugMgr := &volume.VolumePluginMgr{}
	plugMgr.InitPlugins([]volume.VolumePlugin{plug}, volumetest.NewFakeVolumeHost(tmpDir, kubeClient, nil))

	vm, err := NewVolumeManager(
		true,
		testHostname,
		podManager,
		kubeClient,
		plugMgr,
		&containertest.FakeRuntime{},
		&mount.FakeMounter{},
		"",
		fakeRecorder,
		false /* experimentalCheckNodeCapabilitiesBeforeMount*/)

	return vm, err
}

// createObjects returns objects for making a fake clientset. The pv is
// already attached to the node and bound to the claim used by the pod.
func createObjects() (*api.Node, *api.Pod, *api.PersistentVolume, *api.PersistentVolumeClaim) {
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: testHostname},
		Status: api.NodeStatus{
			VolumesAttached: []api.AttachedVolume{
				{
					Name:       "fake/pvA",
					DevicePath: "fake/path",
				},
			}},
		Spec: api.NodeSpec{ExternalID: testHostname},
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "nsA",
			UID:       "1234",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "vol1",
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: "claimA",
						},
					},
				},
			},
			SecurityContext: &api.PodSecurityContext{
				SupplementalGroups: []int64{555},
			},
		},
	}
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "pvA",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
					PDName: "fake-device",
				},
			},
			ClaimRef: &api.ObjectReference{
				Name: "claimA",
			},
		},
	}
	claim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claimA",
			Namespace: "nsA",
		},
		Spec: api.PersistentVolumeClaimSpec{
			VolumeName: "pvA",
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimBound,
		},
	}
	return node, pod, pv, claim
}

func simulateVolumeInUseUpdate(
	volumeName api.UniqueVolumeName,
	stopCh <-chan struct{},
	volumeManager VolumeManager) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			volumeManager.MarkVolumesAsReportedInUse(
				[]api.UniqueVolumeName{volumeName})
		case <-stopCh:
			return
		}
	}
}

func runVolumeManager(manager VolumeManager) chan struct{} {
	stopCh := make(chan struct{})
	//readyCh := make(chan bool, 1)
	//readyCh <- true
	sourcesReady := config.NewSourcesReady(func(_ sets.String) bool { return true })
	go manager.Run(sourcesReady, stopCh)
	return stopCh
}
