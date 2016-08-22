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

package kubelet

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func TestPodVolumesExist(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				Name: "pod1",
				UID:  "pod1uid",
			},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "vol1",
						VolumeSource: api.VolumeSource{
							GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
								PDName: "fake-device1",
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name: "pod2",
				UID:  "pod2uid",
			},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "vol2",
						VolumeSource: api.VolumeSource{
							GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
								PDName: "fake-device2",
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name: "pod3",
				UID:  "pod3uid",
			},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "vol3",
						VolumeSource: api.VolumeSource{
							GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
								PDName: "fake-device3",
							},
						},
					},
				},
			},
		},
	}

	stopCh := runVolumeManager(kubelet)
	defer func() {
		close(stopCh)
	}()

	kubelet.podManager.SetPods(pods)
	for _, pod := range pods {
		err := kubelet.volumeManager.WaitForAttachAndMount(pod)
		if err != nil {
			t.Errorf("Expected success: %v", err)
		}
	}

	for _, pod := range pods {
		podVolumesExist := kubelet.podVolumesExist(pod.UID)
		if !podVolumesExist {
			t.Errorf(
				"Expected to find volumes for pod %q, but podVolumesExist returned false",
				pod.UID)
		}
	}
}
