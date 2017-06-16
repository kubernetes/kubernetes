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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	volumecache "k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
	"k8s.io/kubernetes/test/integration/framework"
)

func fakePodWithVol(namespace string) *v1.Pod {
	fakePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
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

type podCountFunc func(int) bool

// Via integration test we can verify that if pod delete
// event is somehow missed by AttachDetach controller - it still
// gets cleaned up by Desired State of World populator.
func TestPodDeletionWithDswp(t *testing.T) {
	_, server, closeFn := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer closeFn()
	namespaceName := "test-pod-deletion"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				volumehelper.ControllerManagedAttachAnnotation: "true",
			},
		},
	}

	ns := framework.CreateTestingNamespace(namespaceName, server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	testClient, ctrl, informers := createAdClients(ns, t, server, defaultSyncPeriod)
	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})

	if _, err := testClient.Core().Nodes().Create(node); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	stopCh := make(chan struct{})
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(stopCh)
	go informers.Core().V1().PersistentVolumes().Informer().Run(stopCh)
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

	waitForPodsInDSWP(t, ctrl.GetDesiredStateOfWorld())
	// let's stop pod events from getting triggered
	close(podStopCh)
	err = podInformer.GetStore().Delete(podInformerObj)
	if err != nil {
		t.Fatalf("Error deleting pod : %v", err)
	}

	waitToObservePods(t, podInformer, 0)
	// the populator loop turns every 1 minute
	waitForPodFuncInDSWP(t, ctrl.GetDesiredStateOfWorld(), 80*time.Second, "expected 0 pods in dsw after pod delete", 0)
	close(stopCh)
}

func TestPodUpdateWithWithADC(t *testing.T) {
	_, server, closeFn := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer closeFn()
	namespaceName := "test-pod-update"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				volumehelper.ControllerManagedAttachAnnotation: "true",
			},
		},
	}

	ns := framework.CreateTestingNamespace(namespaceName, server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	testClient, ctrl, informers := createAdClients(ns, t, server, defaultSyncPeriod)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})

	if _, err := testClient.Core().Nodes().Create(node); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	stopCh := make(chan struct{})
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(stopCh)
	go informers.Core().V1().PersistentVolumes().Informer().Run(stopCh)
	go ctrl.Run(stopCh)

	waitToObservePods(t, podInformer, 1)
	podKey, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		t.Fatalf("MetaNamespaceKeyFunc failed with : %v", err)
	}

	_, _, err = podInformer.GetStore().GetByKey(podKey)

	if err != nil {
		t.Fatalf("Pod not found in Pod Informer cache : %v", err)
	}

	waitForPodsInDSWP(t, ctrl.GetDesiredStateOfWorld())

	pod.Status.Phase = v1.PodSucceeded

	if _, err := testClient.Core().Pods(ns.Name).UpdateStatus(pod); err != nil {
		t.Errorf("Failed to update pod : %v", err)
	}

	waitForPodFuncInDSWP(t, ctrl.GetDesiredStateOfWorld(), 20*time.Second, "expected 0 pods in dsw after pod completion", 0)

	close(podStopCh)
	close(stopCh)
}

func TestPodUpdateWithKeepTerminatedPodVolumes(t *testing.T) {
	_, server, closeFn := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer closeFn()
	namespaceName := "test-pod-update"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				volumehelper.ControllerManagedAttachAnnotation:  "true",
				volumehelper.KeepTerminatedPodVolumesAnnotation: "true",
			},
		},
	}

	ns := framework.CreateTestingNamespace(namespaceName, server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	testClient, ctrl, informers := createAdClients(ns, t, server, defaultSyncPeriod)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})

	if _, err := testClient.Core().Nodes().Create(node); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	stopCh := make(chan struct{})
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(stopCh)
	go informers.Core().V1().PersistentVolumes().Informer().Run(stopCh)
	go ctrl.Run(stopCh)

	waitToObservePods(t, podInformer, 1)
	podKey, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		t.Fatalf("MetaNamespaceKeyFunc failed with : %v", err)
	}

	_, _, err = podInformer.GetStore().GetByKey(podKey)

	if err != nil {
		t.Fatalf("Pod not found in Pod Informer cache : %v", err)
	}

	waitForPodsInDSWP(t, ctrl.GetDesiredStateOfWorld())

	pod.Status.Phase = v1.PodSucceeded

	if _, err := testClient.Core().Pods(ns.Name).UpdateStatus(pod); err != nil {
		t.Errorf("Failed to update pod : %v", err)
	}

	waitForPodFuncInDSWP(t, ctrl.GetDesiredStateOfWorld(), 20*time.Second, "expected non-zero pods in dsw if KeepTerminatedPodVolumesAnnotation is set", 1)

	close(podStopCh)
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

// wait for pods to be observed in desired state of world
func waitForPodsInDSWP(t *testing.T, dswp volumecache.DesiredStateOfWorld) {
	if err := wait.Poll(time.Millisecond*500, wait.ForeverTestTimeout, func() (bool, error) {
		pods := dswp.GetPodToAdd()
		if len(pods) > 0 {
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Fatalf("Pod not added to desired state of world : %v", err)
	}
}

// wait for pods to be observed in desired state of world
func waitForPodFuncInDSWP(t *testing.T, dswp volumecache.DesiredStateOfWorld, checkTimeout time.Duration, failMessage string, podCount int) {
	if err := wait.Poll(time.Millisecond*500, checkTimeout, func() (bool, error) {
		pods := dswp.GetPodToAdd()
		if len(pods) == podCount {
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Fatalf("%s but got error %v", failMessage, err)
	}
}

func createAdClients(ns *v1.Namespace, t *testing.T, server *httptest.Server, syncPeriod time.Duration) (*clientset.Clientset, attachdetach.AttachDetachController, informers.SharedInformerFactory) {
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
	informers := informers.NewSharedInformerFactory(testClient, resyncPeriod)
	ctrl, err := attachdetach.NewAttachDetachController(
		testClient,
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		informers.Core().V1().PersistentVolumeClaims(),
		informers.Core().V1().PersistentVolumes(),
		cloud,
		plugins,
		false,
		time.Second*5)

	if err != nil {
		t.Fatalf("Error creating AttachDetach : %v", err)
	}
	return testClient, ctrl, informers
}

// Via integration test we can verify that if pod add
// event is somehow missed by AttachDetach controller - it still
// gets added by Desired State of World populator.
func TestPodAddedByDswp(t *testing.T) {
	_, server, closeFn := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer closeFn()
	namespaceName := "test-pod-deletion"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				volumehelper.ControllerManagedAttachAnnotation: "true",
			},
		},
	}

	ns := framework.CreateTestingNamespace(namespaceName, server, t)
	defer framework.DeleteTestingNamespace(ns, server, t)

	testClient, ctrl, informers := createAdClients(ns, t, server, defaultSyncPeriod)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})

	if _, err := testClient.Core().Nodes().Create(node); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.Core().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	stopCh := make(chan struct{})
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(stopCh)
	go informers.Core().V1().PersistentVolumes().Informer().Run(stopCh)
	go ctrl.Run(stopCh)

	waitToObservePods(t, podInformer, 1)
	podKey, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		t.Fatalf("MetaNamespaceKeyFunc failed with : %v", err)
	}

	_, _, err = podInformer.GetStore().GetByKey(podKey)

	if err != nil {
		t.Fatalf("Pod not found in Pod Informer cache : %v", err)
	}

	waitForPodsInDSWP(t, ctrl.GetDesiredStateOfWorld())

	// let's stop pod events from getting triggered
	close(podStopCh)
	podObj, err := api.Scheme.DeepCopy(pod)
	if err != nil {
		t.Fatalf("Error copying pod : %v", err)
	}
	podNew, ok := podObj.(*v1.Pod)
	if !ok {
		t.Fatalf("Error converting pod : %v", err)
	}
	newPodName := "newFakepod"
	podNew.SetName(newPodName)
	err = podInformer.GetStore().Add(podNew)
	if err != nil {
		t.Fatalf("Error adding pod : %v", err)
	}

	waitToObservePods(t, podInformer, 2)

	// the findAndAddActivePods loop turns every 3 minute
	waitForPodFuncInDSWP(t, ctrl.GetDesiredStateOfWorld(), 200*time.Second, "expected 2 pods in dsw after pod addition", 2)

	close(stopCh)
}
