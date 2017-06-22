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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/configmap"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/pod"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	podtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
	"k8s.io/kubernetes/pkg/kubelet/secret"
	"k8s.io/kubernetes/pkg/kubelet/status"
	statustest "k8s.io/kubernetes/pkg/kubelet/status/testing"
	"k8s.io/kubernetes/pkg/util/mount"
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
	podManager := kubepod.NewBasicPodManager(podtest.NewFakeMirrorClient(), secret.NewFakeManager(), configmap.NewFakeManager())

	node, pod, pv, claim := createObjects()
	kubeClient := fake.NewSimpleClientset(node, pod, pv, claim)

	manager, err := newTestVolumeManager(tmpDir, podManager, kubeClient)
	if err != nil {
		t.Fatalf("Failed to initialize volume manager: %v", err)
	}

	stopCh := runVolumeManager(manager)
	defer close(stopCh)

	podManager.SetPods([]*v1.Pod{pod})

	// Fake node status update
	go simulateVolumeInUseUpdate(
		v1.UniqueVolumeName(node.Status.VolumesAttached[0].Name),
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

	expectedInUse := []v1.UniqueVolumeName{v1.UniqueVolumeName(node.Status.VolumesAttached[0].Name)}
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
	podManager := kubepod.NewBasicPodManager(podtest.NewFakeMirrorClient(), secret.NewFakeManager(), configmap.NewFakeManager())

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
			gidAnnotation: strconv.FormatInt(int64(existingGid), 10),
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
		pv := &v1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pvA",
				Annotations: map[string]string{
					volumehelper.VolumeGidAnnotationKey: tc.gidAnnotation,
				},
			},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "fake-device",
					},
				},
				ClaimRef: &v1.ObjectReference{
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

		podManager.SetPods([]*v1.Pod{pod})

		// Fake node status update
		go simulateVolumeInUseUpdate(
			v1.UniqueVolumeName(node.Status.VolumesAttached[0].Name),
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

func newTestVolumeManager(tmpDir string, podManager pod.Manager, kubeClient clientset.Interface) (VolumeManager, error) {
	plug := &volumetest.FakeVolumePlugin{PluginName: "fake", Host: nil}
	fakeRecorder := &record.FakeRecorder{}
	plugMgr := &volume.VolumePluginMgr{}
	plugMgr.InitPlugins([]volume.VolumePlugin{plug}, volumetest.NewFakeVolumeHost(tmpDir, kubeClient, nil))
	statusManager := status.NewManager(kubeClient, podManager, &statustest.FakePodDeletionSafetyProvider{})

	vm, err := NewVolumeManager(
		true,
		testHostname,
		podManager,
		statusManager,
		kubeClient,
		plugMgr,
		&containertest.FakeRuntime{},
		&mount.FakeMounter{},
		"",
		fakeRecorder,
		false, /* experimentalCheckNodeCapabilitiesBeforeMount */
		false /* keepTerminatedPodVolumes */)

	return vm, err
}

// createObjects returns objects for making a fake clientset. The pv is
// already attached to the node and bound to the claim used by the pod.
func createObjects() (*v1.Node, *v1.Pod, *v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: testHostname},
		Status: v1.NodeStatus{
			VolumesAttached: []v1.AttachedVolume{
				{
					Name:       "fake/pvA",
					DevicePath: "fake/path",
				},
			}},
		Spec: v1.NodeSpec{ExternalID: testHostname},
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "abc",
			Namespace: "nsA",
			UID:       "1234",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "vol1",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "claimA",
						},
					},
				},
			},
			SecurityContext: &v1.PodSecurityContext{
				SupplementalGroups: []int64{555},
			},
		},
	}
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pvA",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName: "fake-device",
				},
			},
			ClaimRef: &v1.ObjectReference{
				Name: "claimA",
			},
		},
	}
	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claimA",
			Namespace: "nsA",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName: "pvA",
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase: v1.ClaimBound,
		},
	}
	return node, pod, pv, claim
}

func simulateVolumeInUseUpdate(volumeName v1.UniqueVolumeName, stopCh <-chan struct{}, volumeManager VolumeManager) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			volumeManager.MarkVolumesAsReportedInUse(
				[]v1.UniqueVolumeName{volumeName})
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
