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
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientgoinformers "k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	fakecloud "k8s.io/cloud-provider/fake"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach"
	volumecache "k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/persistentvolume"
	persistentvolumeoptions "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/options"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
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

func fakePodWithPVC(name, pvcName, namespace string) (*v1.Pod, *v1.PersistentVolumeClaim) {
	fakePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
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
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: pvcName,
						},
					},
				},
			},
			NodeName: "node-sandbox",
		},
	}
	class := "fake-sc"
	fakePVC := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      pvcName,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.VolumeResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("5Gi"),
				},
			},
			StorageClassName: &class,
		},
	}
	return fakePod, fakePVC
}

var defaultTimerConfig = attachdetach.TimerConfig{
	ReconcilerLoopPeriod:                              100 * time.Millisecond,
	ReconcilerMaxWaitForUnmountDuration:               6 * time.Second,
	DesiredStateOfWorldPopulatorLoopSleepPeriod:       1 * time.Second,
	DesiredStateOfWorldPopulatorListPodsRetryDuration: 3 * time.Second,
}

// Via integration test we can verify that if pod delete
// event is somehow missed by AttachDetach controller - it still
// gets cleaned up by Desired State of World populator.
func TestPodDeletionWithDswp(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())
	defer server.TearDownFn()

	namespaceName := "test-pod-deletion"
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				util.ControllerManagedAttachAnnotation: "true",
			},
		},
	}

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	testClient, ctrl, pvCtrl, informers := createAdClients(tCtx, t, server, defaultSyncPeriod, defaultTimerConfig)

	ns := framework.CreateNamespaceOrDie(testClient, namespaceName, t)
	defer framework.DeleteNamespaceOrDie(testClient, ns, t)

	pod := fakePodWithVol(namespaceName)

	if _, err := testClient.CoreV1().Nodes().Create(tCtx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	// start controller loop
	go informers.Core().V1().Nodes().Informer().Run(tCtx.Done())
	if _, err := testClient.CoreV1().Pods(ns.Name).Create(tCtx, pod, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(tCtx.Done())

	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(tCtx.Done())
	go informers.Core().V1().PersistentVolumes().Informer().Run(tCtx.Done())
	go informers.Storage().V1().VolumeAttachments().Informer().Run(tCtx.Done())
	initCSIObjects(tCtx.Done(), informers)
	go ctrl.Run(tCtx)
	// Run pvCtrl to avoid leaking goroutines started during its creation.
	go pvCtrl.Run(tCtx)

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
	err = podInformer.GetStore().Delete(podInformerObj)
	if err != nil {
		t.Fatalf("Error deleting pod : %v", err)
	}

	waitToObservePods(t, podInformer, 0)
	// the populator loop turns every 1 minute
	waitForPodFuncInDSWP(t, ctrl.GetDesiredStateOfWorld(), 80*time.Second, "expected 0 pods in dsw after pod delete", 0)
}

func initCSIObjects(stopCh <-chan struct{}, informers clientgoinformers.SharedInformerFactory) {
	go informers.Storage().V1().CSINodes().Informer().Run(stopCh)
	go informers.Storage().V1().CSIDrivers().Informer().Run(stopCh)
}

func TestPodUpdateWithWithADC(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())
	defer server.TearDownFn()
	namespaceName := "test-pod-update"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				util.ControllerManagedAttachAnnotation: "true",
			},
		},
	}

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	testClient, ctrl, pvCtrl, informers := createAdClients(tCtx, t, server, defaultSyncPeriod, defaultTimerConfig)

	ns := framework.CreateNamespaceOrDie(testClient, namespaceName, t)
	defer framework.DeleteNamespaceOrDie(testClient, ns, t)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})
	defer close(podStopCh)

	if _, err := testClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(tCtx.Done())
	go informers.Core().V1().PersistentVolumes().Informer().Run(tCtx.Done())
	go informers.Storage().V1().VolumeAttachments().Informer().Run(tCtx.Done())
	initCSIObjects(tCtx.Done(), informers)
	go ctrl.Run(tCtx)
	// Run pvCtrl to avoid leaking goroutines started during its creation.
	go pvCtrl.Run(tCtx)

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

	if _, err := testClient.CoreV1().Pods(ns.Name).UpdateStatus(context.TODO(), pod, metav1.UpdateOptions{}); err != nil {
		t.Errorf("Failed to update pod : %v", err)
	}

	waitForPodFuncInDSWP(t, ctrl.GetDesiredStateOfWorld(), 20*time.Second, "expected 0 pods in dsw after pod completion", 0)
}

func TestPodUpdateWithKeepTerminatedPodVolumes(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())
	defer server.TearDownFn()
	namespaceName := "test-pod-update"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				util.ControllerManagedAttachAnnotation:  "true",
				util.KeepTerminatedPodVolumesAnnotation: "true",
			},
		},
	}

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	testClient, ctrl, pvCtrl, informers := createAdClients(tCtx, t, server, defaultSyncPeriod, defaultTimerConfig)

	ns := framework.CreateNamespaceOrDie(testClient, namespaceName, t)
	defer framework.DeleteNamespaceOrDie(testClient, ns, t)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})
	defer close(podStopCh)

	if _, err := testClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(tCtx.Done())
	go informers.Core().V1().PersistentVolumes().Informer().Run(tCtx.Done())
	go informers.Storage().V1().VolumeAttachments().Informer().Run(tCtx.Done())
	initCSIObjects(tCtx.Done(), informers)
	go ctrl.Run(tCtx)
	// Run pvCtrl to avoid leaking goroutines started during its creation.
	go pvCtrl.Run(tCtx)

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

	if _, err := testClient.CoreV1().Pods(ns.Name).UpdateStatus(context.TODO(), pod, metav1.UpdateOptions{}); err != nil {
		t.Errorf("Failed to update pod : %v", err)
	}

	waitForPodFuncInDSWP(t, ctrl.GetDesiredStateOfWorld(), 20*time.Second, "expected non-zero pods in dsw if KeepTerminatedPodVolumesAnnotation is set", 1)
}

// wait for the podInformer to observe the pods. Call this function before
// running the RC manager to prevent the rc manager from creating new pods
// rather than adopting the existing ones.
func waitToObservePods(t *testing.T, podInformer cache.SharedIndexInformer, podNum int) {
	if err := wait.Poll(100*time.Millisecond, 60*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		if len(objects) == podNum {
			return true, nil
		}
		return false, nil
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

func createAdClients(ctx context.Context, t *testing.T, server *kubeapiservertesting.TestServer, syncPeriod time.Duration, timers attachdetach.TimerConfig) (*clientset.Clientset, attachdetach.AttachDetachController, *persistentvolume.PersistentVolumeController, clientgoinformers.SharedInformerFactory) {
	config := restclient.CopyConfig(server.ClientConfig)
	config.QPS = 1000000
	config.Burst = 1000000
	resyncPeriod := 12 * time.Hour
	testClient := clientset.NewForConfigOrDie(server.ClientConfig)

	host := volumetest.NewFakeVolumeHost(t, "/tmp/fake", nil, nil)
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
	cloud := &fakecloud.Cloud{}
	informers := clientgoinformers.NewSharedInformerFactory(testClient, resyncPeriod)
	ctrl, err := attachdetach.NewAttachDetachController(
		ctx,
		testClient,
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		informers.Core().V1().PersistentVolumeClaims(),
		informers.Core().V1().PersistentVolumes(),
		informers.Storage().V1().CSINodes(),
		informers.Storage().V1().CSIDrivers(),
		informers.Storage().V1().VolumeAttachments(),
		cloud,
		plugins,
		nil, /* prober */
		false,
		5*time.Second,
		false,
		timers,
	)

	if err != nil {
		t.Fatalf("Error creating AttachDetach : %v", err)
	}

	// create pv controller
	controllerOptions := persistentvolumeoptions.NewPersistentVolumeControllerOptions()
	params := persistentvolume.ControllerParameters{
		KubeClient:                testClient,
		SyncPeriod:                controllerOptions.PVClaimBinderSyncPeriod,
		VolumePlugins:             plugins,
		Cloud:                     nil,
		ClusterName:               "volume-test-cluster",
		VolumeInformer:            informers.Core().V1().PersistentVolumes(),
		ClaimInformer:             informers.Core().V1().PersistentVolumeClaims(),
		ClassInformer:             informers.Storage().V1().StorageClasses(),
		PodInformer:               informers.Core().V1().Pods(),
		NodeInformer:              informers.Core().V1().Nodes(),
		EnableDynamicProvisioning: false,
	}
	pvCtrl, err := persistentvolume.NewController(ctx, params)
	if err != nil {
		t.Fatalf("Failed to create PV controller: %v", err)
	}
	return testClient, ctrl, pvCtrl, informers
}

// Via integration test we can verify that if pod add
// event is somehow missed by AttachDetach controller - it still
// gets added by Desired State of World populator.
func TestPodAddedByDswp(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())
	defer server.TearDownFn()
	namespaceName := "test-pod-deletion"

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				util.ControllerManagedAttachAnnotation: "true",
			},
		},
	}

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")
	testClient, ctrl, pvCtrl, informers := createAdClients(tCtx, t, server, defaultSyncPeriod, defaultTimerConfig)

	ns := framework.CreateNamespaceOrDie(testClient, namespaceName, t)
	defer framework.DeleteNamespaceOrDie(testClient, ns, t)

	pod := fakePodWithVol(namespaceName)
	podStopCh := make(chan struct{})

	if _, err := testClient.CoreV1().Nodes().Create(tCtx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	go informers.Core().V1().Nodes().Informer().Run(podStopCh)

	if _, err := testClient.CoreV1().Pods(ns.Name).Create(tCtx, pod, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	podInformer := informers.Core().V1().Pods().Informer()
	go podInformer.Run(podStopCh)

	// start controller loop
	go informers.Core().V1().PersistentVolumeClaims().Informer().Run(tCtx.Done())
	go informers.Core().V1().PersistentVolumes().Informer().Run(tCtx.Done())
	go informers.Storage().V1().VolumeAttachments().Informer().Run(tCtx.Done())
	initCSIObjects(tCtx.Done(), informers)
	go ctrl.Run(tCtx)
	// Run pvCtrl to avoid leaking goroutines started during its creation.
	go pvCtrl.Run(tCtx)

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
	podNew := pod.DeepCopy()
	newPodName := "newFakepod"
	podNew.SetName(newPodName)
	err = podInformer.GetStore().Add(podNew)
	if err != nil {
		t.Fatalf("Error adding pod : %v", err)
	}

	waitToObservePods(t, podInformer, 2)

	// the findAndAddActivePods loop turns every 3 minute
	waitForPodFuncInDSWP(t, ctrl.GetDesiredStateOfWorld(), 200*time.Second, "expected 2 pods in dsw after pod addition", 2)
}

func TestPVCBoundWithADC(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())
	defer server.TearDownFn()

	tCtx := ktesting.Init(t)
	defer tCtx.Cancel("test has completed")

	namespaceName := "test-pod-deletion"

	testClient, ctrl, pvCtrl, informers := createAdClients(tCtx, t, server, defaultSyncPeriod, attachdetach.TimerConfig{
		ReconcilerLoopPeriod:                        100 * time.Millisecond,
		ReconcilerMaxWaitForUnmountDuration:         6 * time.Second,
		DesiredStateOfWorldPopulatorLoopSleepPeriod: 24 * time.Hour,
		// Use high duration to disable DesiredStateOfWorldPopulator.findAndAddActivePods loop in test.
		DesiredStateOfWorldPopulatorListPodsRetryDuration: 24 * time.Hour,
	})

	ns := framework.CreateNamespaceOrDie(testClient, namespaceName, t)
	defer framework.DeleteNamespaceOrDie(testClient, ns, t)

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-sandbox",
			Annotations: map[string]string{
				util.ControllerManagedAttachAnnotation: "true",
			},
		},
	}
	if _, err := testClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to created node : %v", err)
	}

	// pods with pvc not bound
	pvcs := []*v1.PersistentVolumeClaim{}
	for i := 0; i < 3; i++ {
		pod, pvc := fakePodWithPVC(fmt.Sprintf("fakepod-pvcnotbound-%d", i), fmt.Sprintf("fakepvc-%d", i), namespaceName)
		if _, err := testClient.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
			t.Errorf("Failed to create pod : %v", err)
		}
		if _, err := testClient.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(context.TODO(), pvc, metav1.CreateOptions{}); err != nil {
			t.Errorf("Failed to create pvc : %v", err)
		}
		pvcs = append(pvcs, pvc)
	}
	// pod with no pvc
	podNew := fakePodWithVol(namespaceName)
	podNew.SetName("fakepod")
	if _, err := testClient.CoreV1().Pods(podNew.Namespace).Create(context.TODO(), podNew, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pod : %v", err)
	}

	// start controller loop
	informers.Start(tCtx.Done())
	informers.WaitForCacheSync(tCtx.Done())
	initCSIObjects(tCtx.Done(), informers)
	go ctrl.Run(tCtx)
	go pvCtrl.Run(tCtx)

	waitToObservePods(t, informers.Core().V1().Pods().Informer(), 4)
	// Give attachdetach controller enough time to populate pods into DSWP.
	time.Sleep(10 * time.Second)
	waitForPodFuncInDSWP(t, ctrl.GetDesiredStateOfWorld(), 60*time.Second, "expected 1 pod in dsw", 1)
	for _, pvc := range pvcs {
		createPVForPVC(t, testClient, pvc)
	}
	waitForPodFuncInDSWP(t, ctrl.GetDesiredStateOfWorld(), 60*time.Second, "expected 4 pods in dsw after PVCs are bound", 4)
}

// Create PV for PVC, pv controller will bind them together.
func createPVForPVC(t *testing.T, testClient *clientset.Clientset, pvc *v1.PersistentVolumeClaim) {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("fakepv-%s", pvc.Name),
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity:    pvc.Spec.Resources.Requests,
			AccessModes: pvc.Spec.AccessModes,
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/var/www/html",
				},
			},
			ClaimRef:         &v1.ObjectReference{Name: pvc.Name, Namespace: pvc.Namespace},
			StorageClassName: *pvc.Spec.StorageClassName,
		},
	}
	if _, err := testClient.CoreV1().PersistentVolumes().Create(context.TODO(), pv, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pv : %v", err)
	}
}
