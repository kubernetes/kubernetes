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

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
)

func Test_NewAttachDetachController_Positive(t *testing.T) {
	// Arrange
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	resyncPeriod := 5 * time.Minute
	podInformer := informers.NewPodInformer(fakeKubeClient, resyncPeriod)
	nodeInformer := informers.NewNodeInformer(fakeKubeClient, resyncPeriod)
	pvcInformer := informers.NewPVCInformer(fakeKubeClient, resyncPeriod)
	pvInformer := informers.NewPVInformer(fakeKubeClient, resyncPeriod)

	// Act
	_, err := NewAttachDetachController(
		fakeKubeClient,
		podInformer,
		nodeInformer,
		pvcInformer,
		pvInformer,
		nil, /* cloud */
		nil, /* plugins */
		false,
		time.Second*5)

	// Assert
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: <%v>", err)
	}
}

func Test_AttachDetachControllerStateOfWolrdPopulators_Positive(t *testing.T) {
	// Arrange
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	resyncPeriod := 5 * time.Minute
	pvcInformer := informers.NewPVCInformer(fakeKubeClient, resyncPeriod)
	pvInformer := informers.NewPVInformer(fakeKubeClient, resyncPeriod)

	adc := &attachDetachController{
		kubeClient:  fakeKubeClient,
		pvcInformer: pvcInformer,
		pvInformer:  pvInformer,
		cloud:       nil,
	}

	// Act
	plugins := controllervolumetesting.CreateTestPlugin()

	if err := adc.volumePluginMgr.InitPlugins(plugins, adc); err != nil {
		t.Fatalf("Could not initialize volume plugins for Attach/Detach Controller: %+v", err)
	}

	adc.desiredStateOfWorld = cache.NewDesiredStateOfWorld(&adc.volumePluginMgr)
	adc.actualStateOfWorld = cache.NewActualStateOfWorld(&adc.volumePluginMgr)

	err := adc.populateActualStateOfWorld()
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: <%v>", err)
	}

	adc.desiredStateOfWorld.AddNode("mynode")
	err = adc.populateDesiredStateOfWorld()
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: %v", err)
	}

	// Test the ActualStateOfWorld contains all the node volumes
	nodes, err := adc.kubeClient.Core().Nodes().List(v1.ListOptions{})
	for i := range nodes.Items {
		nodeName := types.NodeName(nodes.Items[i].Name)
		for _, attachedVolume := range nodes.Items[i].Status.VolumesAttached {
			found := adc.actualStateOfWorld.VolumeNodeExists(attachedVolume.Name, nodeName)
			if !found {
				t.Fatalf("Run failed with error. Node %s, volume %s not found", nodeName, attachedVolume.Name)
			}
		}
	}

	pods, err := adc.kubeClient.Core().Pods(v1.NamespaceAll).List(v1.ListOptions{})
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: %v", err)
	}
	for _, pod := range pods.Items {
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
