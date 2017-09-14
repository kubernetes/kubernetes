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

package attachdetach

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	"k8s.io/kubernetes/pkg/volume"
)

func Test_NewAttachDetachController_Positive(t *testing.T) {
	// Arrange
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())

	// Act
	_, err := NewAttachDetachController(
		fakeKubeClient,
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().PersistentVolumes(),
		nil, /* cloud */
		nil, /* plugins */
		nil, /* prober */
		false,
		5*time.Second,
		DefaultTimerConfig)

	// Assert
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: <%v>", err)
	}
}

func Test_AttachDetachControllerStateOfWolrdPopulators_Positive(t *testing.T) {
	// Arrange
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	podInformer := informerFactory.Core().V1().Pods()
	nodeInformer := informerFactory.Core().V1().Nodes()
	pvcInformer := informerFactory.Core().V1().PersistentVolumeClaims()
	pvInformer := informerFactory.Core().V1().PersistentVolumes()

	adc := &attachDetachController{
		kubeClient:  fakeKubeClient,
		pvcLister:   pvcInformer.Lister(),
		pvcsSynced:  pvcInformer.Informer().HasSynced,
		pvLister:    pvInformer.Lister(),
		pvsSynced:   pvInformer.Informer().HasSynced,
		podLister:   podInformer.Lister(),
		podsSynced:  podInformer.Informer().HasSynced,
		nodeLister:  nodeInformer.Lister(),
		nodesSynced: nodeInformer.Informer().HasSynced,
		cloud:       nil,
	}

	// Act
	plugins := controllervolumetesting.CreateTestPlugin()
	var prober volume.DynamicPluginProber = nil // TODO (#51147) inject mock

	if err := adc.volumePluginMgr.InitPlugins(plugins, prober, adc); err != nil {
		t.Fatalf("Could not initialize volume plugins for Attach/Detach Controller: %+v", err)
	}

	adc.actualStateOfWorld = cache.NewActualStateOfWorld(&adc.volumePluginMgr)
	adc.desiredStateOfWorld = cache.NewDesiredStateOfWorld(&adc.volumePluginMgr)

	err := adc.populateActualStateOfWorld()
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: <%v>", err)
	}

	err = adc.populateDesiredStateOfWorld()
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: %v", err)
	}

	// Test the ActualStateOfWorld contains all the node volumes
	nodes, err := adc.nodeLister.List(labels.Everything())
	if err != nil {
		t.Fatalf("Failed to list nodes in indexer. Expected: <no error> Actual: %v", err)
	}

	for _, node := range nodes {
		nodeName := types.NodeName(node.Name)
		for _, attachedVolume := range node.Status.VolumesAttached {
			found := adc.actualStateOfWorld.VolumeNodeExists(attachedVolume.Name, nodeName)
			if !found {
				t.Fatalf("Run failed with error. Node %s, volume %s not found", nodeName, attachedVolume.Name)
			}
		}
	}

	pods, err := adc.podLister.List(labels.Everything())
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: %v", err)
	}
	for _, pod := range pods {
		uniqueName := fmt.Sprintf("%s/%s", controllervolumetesting.TestPluginName, pod.Spec.Volumes[0].Name)
		nodeName := types.NodeName(pod.Spec.NodeName)
		found := adc.desiredStateOfWorld.VolumeExists(v1.UniqueVolumeName(uniqueName), nodeName)
		if !found {
			t.Fatalf("Run failed with error. Volume %s, node %s not found in DesiredStateOfWorld",
				pod.Spec.Volumes[0].Name,
				pod.Spec.NodeName)
		}
	}
}

func Test_AttachDetachControllerRecovery(t *testing.T) {
	attachDetachRecoveryTestCase(t, []*v1.Pod{}, []*v1.Pod{})
	newPod1 := controllervolumetesting.NewPodWithVolume("newpod-1", "volumeName2", "mynode-1")
	attachDetachRecoveryTestCase(t, []*v1.Pod{newPod1}, []*v1.Pod{})
	newPod1 = controllervolumetesting.NewPodWithVolume("newpod-1", "volumeName2", "mynode-1")
	attachDetachRecoveryTestCase(t, []*v1.Pod{}, []*v1.Pod{newPod1})
	newPod1 = controllervolumetesting.NewPodWithVolume("newpod-1", "volumeName2", "mynode-1")
	newPod2 := controllervolumetesting.NewPodWithVolume("newpod-2", "volumeName3", "mynode-1")
	attachDetachRecoveryTestCase(t, []*v1.Pod{newPod1}, []*v1.Pod{newPod2})
}

func attachDetachRecoveryTestCase(t *testing.T, extraPods1 []*v1.Pod, extraPods2 []*v1.Pod) {
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, time.Second*1)
	//informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, time.Second*1)
	plugins := controllervolumetesting.CreateTestPlugin()
	var prober volume.DynamicPluginProber = nil // TODO (#51147) inject mock
	nodeInformer := informerFactory.Core().V1().Nodes().Informer()
	podInformer := informerFactory.Core().V1().Pods().Informer()
	var podsNum, extraPodsNum, nodesNum, i int

	stopCh := make(chan struct{})

	pods, err := fakeKubeClient.Core().Pods(v1.NamespaceAll).List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: %v", err)
	}

	for _, pod := range pods.Items {
		podToAdd := pod
		podInformer.GetIndexer().Add(&podToAdd)
		podsNum++
	}
	nodes, err := fakeKubeClient.Core().Nodes().List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: %v", err)
	}
	for _, node := range nodes.Items {
		nodeToAdd := node
		nodeInformer.GetIndexer().Add(&nodeToAdd)
		nodesNum++
	}

	informerFactory.Start(stopCh)

	if !controller.WaitForCacheSync("attach detach", stopCh,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
		informerFactory.Core().V1().Nodes().Informer().HasSynced) {
		t.Fatalf("Error waiting for the informer caches to sync")
	}

	// Make sure the nodes and pods are in the inforer cache
	i = 0
	nodeList, err := informerFactory.Core().V1().Nodes().Lister().List(labels.Everything())
	for len(nodeList) < nodesNum {
		if err != nil {
			t.Fatalf("Error getting list of nodes %v", err)
		}
		if i > 100 {
			t.Fatalf("Time out while waiting for the node informer sync: found %d nodes, expected %d nodes", len(nodeList), nodesNum)
		}
		time.Sleep(100 * time.Millisecond)
		nodeList, err = informerFactory.Core().V1().Nodes().Lister().List(labels.Everything())
		i++
	}
	i = 0
	podList, err := informerFactory.Core().V1().Pods().Lister().List(labels.Everything())
	for len(podList) < podsNum {
		if err != nil {
			t.Fatalf("Error getting list of nodes %v", err)
		}
		if i > 100 {
			t.Fatalf("Time out while waiting for the pod informer sync: found %d pods, expected %d pods", len(podList), podsNum)
		}
		time.Sleep(100 * time.Millisecond)
		podList, err = informerFactory.Core().V1().Pods().Lister().List(labels.Everything())
		i++
	}

	// Create the controller
	adcObj, err := NewAttachDetachController(
		fakeKubeClient,
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().PersistentVolumes(),
		nil, /* cloud */
		plugins,
		prober,
		false,
		1*time.Second,
		DefaultTimerConfig)

	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: <%v>", err)
	}

	adc := adcObj.(*attachDetachController)

	// Populate ASW
	err = adc.populateActualStateOfWorld()
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: <%v>", err)
	}

	for _, newPod := range extraPods1 {
		// Add a new pod between ASW and DSW ppoulators
		_, err = adc.kubeClient.Core().Pods(newPod.ObjectMeta.Namespace).Create(newPod)
		if err != nil {
			t.Fatalf("Run failed with error. Failed to create a new pod: <%v>", err)
		}
		extraPodsNum++
		podInformer.GetIndexer().Add(newPod)

	}

	// Populate DSW
	err = adc.populateDesiredStateOfWorld()
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: %v", err)
	}

	for _, newPod := range extraPods2 {
		// Add a new pod between DSW ppoulator and reconciler run
		_, err = adc.kubeClient.Core().Pods(newPod.ObjectMeta.Namespace).Create(newPod)
		if err != nil {
			t.Fatalf("Run failed with error. Failed to create a new pod: <%v>", err)
		}
		extraPodsNum++
		podInformer.GetIndexer().Add(newPod)
	}

	go adc.reconciler.Run(stopCh)
	go adc.desiredStateOfWorldPopulator.Run(stopCh)
	defer close(stopCh)

	time.Sleep(time.Second * 1) // Wait so the reconciler calls sync at least once

	testPlugin := plugins[0].(*controllervolumetesting.TestPlugin)
	for i = 0; i <= 10; i++ {
		var attachedVolumesNum int = 0
		var detachedVolumesNum int = 0

		time.Sleep(time.Second * 1) // Wait for a second
		for _, volumeList := range testPlugin.GetAttachedVolumes() {
			attachedVolumesNum += len(volumeList)
		}
		for _, volumeList := range testPlugin.GetDetachedVolumes() {
			detachedVolumesNum += len(volumeList)
		}

		// All the "extra pods" should result in volume to be attached, the pods all share one volume
		// which should be attached (+1), the volumes found only in the nodes status should be detached
		if attachedVolumesNum == 1+extraPodsNum && detachedVolumesNum == nodesNum {
			break
		}
		if i == 10 { // 10 seconds time out
			t.Fatalf("Waiting for the volumes to attach/detach timed out: attached %d (expected %d); detached %d (%d)",
				attachedVolumesNum, 1+extraPodsNum, detachedVolumesNum, nodesNum)
		}
	}

	if testPlugin.GetErrorEncountered() {
		t.Fatalf("Fatal error encountered in the testing volume plugin")
	}

}
