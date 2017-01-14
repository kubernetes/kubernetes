// +build integration,!no-etcd

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

package volume

import (
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
	"k8s.io/kubernetes/test/integration/framework"
)

func fakePodWithVol(namespace string) *v1.Pod {
	fakePod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Namespace: namespace,
			Name:      "fakepod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-container",
					Image: "nginx",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "fake-mount",
							MountPath: "/var/www/html",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "fake-mount",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/www/html",
						},
					},
				},
			},
			NodeName: "node-sandbox",
		},
	}
	return fakePod
}

// Via integration test we can verify that if pod delete
// event is somehow missed by AttachDetach controller - it still
// gets cleaned up by Desired State of World populator.
func TestPodDeletionWithDswp(t *testing.T) {
	_, server := framework.RunAMaster(nil)
	defer server.Close()
	namespaceName := "test-pod-deletion"

	node := &v1.Node{
		ObjectMeta: v1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				volumehelper.ControllerManagedAttachAnnotation: "true",
			},
		},
	}

	ns := framework.CreateTestingNamespace(namespaceName, server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	testClient, ctrl, podInformer, nodeInformer := createAdClients(ns, t, server, defaultSyncPeriod)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})

	if _, err := testClient.Core().Nodes().Create(node); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go nodeInformer.Run(podStopCh)

	if _, err := testClient.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	go podInformer.Run(podStopCh)

	// start controller loop
	stopCh := make(chan struct{})
	go ctrl.Run(stopCh)

	waitToObservePods(t, podInformer, 1)
	podKey, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		t.Fatalf("MetaNamespaceKeyFunc failed with : %v", err)
	}

	podInformerObj, _, err := podInformer.GetStore().GetByKey(podKey)

	if err != nil {
		t.Fatalf("Pod not found in Pod Informer cache : %v", err)
	}

	podsToAdd := ctrl.GetDesiredStateOfWorld().GetPodToAdd()

	if len(podsToAdd) == 0 {
		t.Fatalf("Pod not added to desired state of world")
	}

	// let's stop pod events from getting triggered
	close(podStopCh)
	err = podInformer.GetStore().Delete(podInformerObj)
	if err != nil {
		t.Fatalf("Error deleting pod : %v", err)
	}

	waitToObservePods(t, podInformer, 0)
	// the populator loop turns every 1 minute
	time.Sleep(80 * time.Second)
	podsToAdd = ctrl.GetDesiredStateOfWorld().GetPodToAdd()
	if len(podsToAdd) != 0 {
		t.Fatalf("All pods should have been removed")
	}

	close(stopCh)
}

// wait for the podInformer to observe the pods. Call this function before
// running the RC manager to prevent the rc manager from creating new pods
// rather than adopting the existing ones.
func waitToObservePods(t *testing.T, podInformer cache.SharedIndexInformer, podNum int) {
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		if len(objects) == podNum {
			return true, nil
		} else {
			return false, nil
		}
	}); err != nil {
		t.Fatal(err)
	}
}

func createAdClients(ns *v1.Namespace, t *testing.T, server *httptest.Server, syncPeriod time.Duration) (*clientset.Clientset, attachdetach.AttachDetachController, cache.SharedIndexInformer, cache.SharedIndexInformer) {
	config := restclient.Config{
		Host:          server.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion},
		QPS:           1000000,
		Burst:         1000000,
	}
	resyncPeriod := 12 * time.Hour
	testClient := clientset.NewForConfigOrDie(&config)

	host := volumetest.NewFakeVolumeHost("/tmp/fake", nil, nil)
	plugin := &volumetest.FakeVolumePlugin{
		PluginName:             provisionerPluginName,
		Host:                   host,
		Config:                 volume.VolumeConfig{},
		LastProvisionerOptions: volume.VolumeOptions{},
		NewAttacherCallCount:   0,
		NewDetacherCallCount:   0,
		Mounters:               nil,
		Unmounters:             nil,
		Attachers:              nil,
		Detachers:              nil,
	}
	plugins := []volume.VolumePlugin{plugin}
	cloud := &fakecloud.FakeCloud{}
	podInformer := informers.NewPodInformer(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "pod-informer")), resyncPeriod)
	nodeInformer := informers.NewNodeInformer(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "node-informer")), resyncPeriod)
	pvcInformer := informers.NewNodeInformer(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "pvc-informer")), resyncPeriod)
	pvInformer := informers.NewNodeInformer(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "pv-informer")), resyncPeriod)
	ctrl, err := attachdetach.NewAttachDetachController(testClient, podInformer, nodeInformer, pvcInformer, pvInformer, cloud, plugins, false, time.Second*5)
	if err != nil {
		t.Fatalf("Error creating AttachDetach : %v", err)
	}
	return testClient, ctrl, podInformer, nodeInformer
}
