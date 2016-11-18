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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/restclient"
	fake_cloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

func init() {
	integration.RequireEtcd()
}

func fakePodWithVol(namespace string) *api.Pod {
	fakePod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Namespace: namespace,
			Name:      "fakepod",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "fake-container",
					Image: "nginx",
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "fake-mount",
							MountPath: "/var/www/html",
						},
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name: "fake-mount",
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{
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

func TestPodDeletionWithDswp(t *testing.T) {
	_, server := framework.RunAMaster(nil)
	defer server.Close()
	namespaceName := "test-pod-deletion"

	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
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
	go func() {
		ctrl.Run(stopCh)
	}()

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
	time.Sleep(2 * time.Minute)
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

func createAdClients(ns *api.Namespace, t *testing.T, server *httptest.Server, syncPeriod time.Duration) (*internalclientset.Clientset, attachdetach.AttachDetachController, cache.SharedIndexInformer, cache.SharedIndexInformer) {
	config := restclient.Config{
		Host:          server.URL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(api.GroupName).GroupVersion},
		QPS:           1000000,
		Burst:         1000000,
	}
	resyncPeriod := 12 * time.Hour
	testClient := internalclientset.NewForConfigOrDie(&config)

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
	cloud := &fake_cloud.FakeCloud{}
	podInformer := informers.NewPodInformer(internalclientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "pod-informer")), resyncPeriod)
	nodeInformer := informers.NewNodeInformer(internalclientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "node-informer")), resyncPeriod)
	pvcInformer := informers.NewNodeInformer(internalclientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "pvc-informer")), resyncPeriod)
	pvInformer := informers.NewNodeInformer(internalclientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "pv-informer")), resyncPeriod)
	fakeRecorder := &record.FakeRecorder{}
	ctrl, err := attachdetach.NewAttachDetachController(testClient, podInformer, nodeInformer, pvcInformer, pvInformer, cloud, plugins, fakeRecorder)
	if err != nil {
		t.Fatalf("Error creating AttachDetach : %v", err)
	}
	return testClient, ctrl, podInformer, nodeInformer
}
