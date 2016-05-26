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

package volume

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller/framework/informers"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

func Test_NewAttachDetachController_Positive(t *testing.T) {
	// Arrange
	fakeKubeClient := createTestClient()
	resyncPeriod := 5 * time.Minute
	podInformer := informers.CreateSharedPodIndexInformer(fakeKubeClient, resyncPeriod)
	nodeInformer := informers.CreateSharedNodeIndexInformer(fakeKubeClient, resyncPeriod)
	pvcInformer := informers.CreateSharedPVCIndexInformer(fakeKubeClient, resyncPeriod)
	pvInformer := informers.CreateSharedPVIndexInformer(fakeKubeClient, resyncPeriod)

	// Act
	_, err := NewAttachDetachController(
		fakeKubeClient,
		podInformer,
		nodeInformer,
		pvcInformer,
		pvInformer,
		nil, /* cloud */
		nil /* plugins */)

	// Assert
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: <%v>", err)
	}
}

func createTestClient() *fake.Clientset {
	fakeClient := &fake.Clientset{}

	fakeClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := &api.PodList{}
		podNamePrefix := "mypod"
		namespace := "mynamespace"
		for i := 0; i < 5; i++ {
			podName := fmt.Sprintf("%s-%d", podNamePrefix, i)
			pod := api.Pod{
				Status: api.PodStatus{
					Phase: api.PodRunning,
				},
				ObjectMeta: api.ObjectMeta{
					Name:      podName,
					Namespace: namespace,
					Labels: map[string]string{
						"name": podName,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "containerName",
							Image: "containerImage",
							VolumeMounts: []api.VolumeMount{
								{
									Name:      "volumeMountName",
									ReadOnly:  false,
									MountPath: "/mnt",
								},
							},
						},
					},
					Volumes: []api.Volume{
						{
							Name: "volumeName",
							VolumeSource: api.VolumeSource{
								GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
									PDName:   "pdName",
									FSType:   "ext4",
									ReadOnly: false,
								},
							},
						},
					},
				},
			}
			obj.Items = append(obj.Items, pod)
		}
		return true, obj, nil
	})

	fakeWatch := watch.NewFake()
	fakeClient.AddWatchReactor("*", core.DefaultWatchReactor(fakeWatch, nil))

	return fakeClient
}
