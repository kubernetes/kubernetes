/*
Copyright 2022 The Kubernetes Authors.

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

package statusupdater

import (
	"context"
	"errors"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"testing"
)

// setupNodeStatusUpdate creates all the needed objects for testing.
// the initial environment has 2 nodes with no volumes attached
// and adds one volume to attach to each node to the actual state of the world
func setupNodeStatusUpdate(logger klog.Logger, t *testing.T) (cache.ActualStateOfWorld, *fake.Clientset, NodeStatusUpdater) {
	testNode1 := corev1.Node{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Node",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "testnode-1",
		},
		Status: corev1.NodeStatus{},
	}
	testNode2 := corev1.Node{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Node",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "testnode-2",
		},
		Status: corev1.NodeStatus{},
	}
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := fake.NewSimpleClientset(&testNode1, &testNode2)
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nodeInformer := informerFactory.Core().V1().Nodes()
	nsu := NewNodeStatusUpdater(fakeKubeClient, nodeInformer.Lister(), asw)

	err := nodeInformer.Informer().GetStore().Add(&testNode1)
	if err != nil {
		t.Fatalf(".Informer().GetStore().Add failed. Expected: <no error> Actual: <%v>", err)
	}
	err = nodeInformer.Informer().GetStore().Add(&testNode2)
	if err != nil {
		t.Fatalf(".Informer().GetStore().Add failed. Expected: <no error> Actual: <%v>", err)
	}

	volumeName1 := corev1.UniqueVolumeName("volume-name-1")
	volumeName2 := corev1.UniqueVolumeName("volume-name-2")
	volumeSpec1 := controllervolumetesting.GetTestVolumeSpec(string(volumeName1), volumeName1)
	volumeSpec2 := controllervolumetesting.GetTestVolumeSpec(string(volumeName2), volumeName2)

	nodeName1 := types.NodeName("testnode-1")
	nodeName2 := types.NodeName("testnode-2")
	devicePath := "fake/device/path"

	_, err = asw.AddVolumeNode(logger, volumeName1, volumeSpec1, nodeName1, devicePath, true)
	if err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", err)
	}
	_, err = asw.AddVolumeNode(logger, volumeName2, volumeSpec2, nodeName2, devicePath, true)
	if err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", err)
	}

	return asw, fakeKubeClient, nsu
}

// TestNodeStatusUpdater_UpdateNodeStatuses_TwoNodesUpdate calls setup
// calls UpdateNodeStatuses()
// check that asw.GetVolumesToReportAttached reports nothing left to attach
// checks that each node status.volumesAttached is of length 1 and contains the correct volume
func TestNodeStatusUpdater_UpdateNodeStatuses_TwoNodesUpdate(t *testing.T) {
	ctx := context.Background()
	logger := klog.FromContext(ctx)
	asw, fakeKubeClient, nsu := setupNodeStatusUpdate(logger, t)

	err := nsu.UpdateNodeStatuses(logger)
	if err != nil {
		t.Fatalf("UpdateNodeStatuses failed. Expected: <no error> Actual: <%v>", err)
	}

	needToReport := asw.GetVolumesToReportAttached(logger)
	if len(needToReport) != 0 {
		t.Fatalf("len(asw.GetVolumesToReportAttached()) Expected: <0> Actual: <%v>", len(needToReport))
	}

	node, err := fakeKubeClient.CoreV1().Nodes().Get(ctx, "testnode-1", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Nodes().Get failed. Expected: <no error> Actual: <%v>", err)
	}
	if len(node.Status.VolumesAttached) != 1 {
		t.Fatalf("len(node.Status.VolumesAttached) Expected: <1> Actual: <%v>", len(node.Status.VolumesAttached))
	}
	if node.Status.VolumesAttached[0].Name != "volume-name-1" {
		t.Fatalf("volumeName Expected: <volume-name-1> Actual: <%s>", node.Status.VolumesAttached[0].Name)
	}

	node, err = fakeKubeClient.CoreV1().Nodes().Get(ctx, "testnode-2", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Nodes().Get failed. Expected: <no error> Actual: <%v>", err)
	}
	if len(node.Status.VolumesAttached) != 1 {
		t.Fatalf("len(node.Status.VolumesAttached) Expected: <1> Actual: <%v>", len(node.Status.VolumesAttached))
	}
	if node.Status.VolumesAttached[0].Name != "volume-name-2" {
		t.Fatalf("volumeName Expected: <volume-name-2> Actual: <%s>", node.Status.VolumesAttached[0].Name)
	}
}

func TestNodeStatusUpdater_UpdateNodeStatuses_FailureInFirstUpdate(t *testing.T) {
	ctx := context.Background()
	logger := klog.FromContext(ctx)
	asw, fakeKubeClient, nsu := setupNodeStatusUpdate(logger, t)

	var failedNode string
	failedOnce := false
	failureErr := fmt.Errorf("test generated error")
	fakeKubeClient.PrependReactor("patch", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		patchAction := action.(core.PatchAction)
		if !failedOnce {
			failedNode = patchAction.GetName()
			failedOnce = true
			return true, nil, failureErr
		}
		return false, nil, nil
	})

	err := nsu.UpdateNodeStatuses(logger)
	if errors.Is(err, failureErr) {
		t.Fatalf("UpdateNodeStatuses failed. Expected: <test generated error> Actual: <%v>", err)
	}

	needToReport := asw.GetVolumesToReportAttached(logger)
	if len(needToReport) != 1 {
		t.Fatalf("len(asw.GetVolumesToReportAttached()) Expected: <1> Actual: <%v>", len(needToReport))
	}
	if _, ok := needToReport[types.NodeName(failedNode)]; !ok {
		t.Fatalf("GetVolumesToReportAttached() did not report correct node Expected: <%s> Actual: <%v>", failedNode, needToReport)
	}

	nodes, err := fakeKubeClient.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Nodes().List failed. Expected: <no error> Actual: <%v>", err)
	}

	if len(nodes.Items) != 2 {
		t.Fatalf("len(nodes.Items) Expected: <2> Actual: <%v>", len(nodes.Items))
	}

	for _, node := range nodes.Items {
		if node.Name == failedNode {
			if len(node.Status.VolumesAttached) != 0 {
				t.Fatalf("len(node.Status.VolumesAttached) Expected: <0> Actual: <%v>", len(node.Status.VolumesAttached))
			}
		} else {
			if len(node.Status.VolumesAttached) != 1 {
				t.Fatalf("len(node.Status.VolumesAttached) Expected: <1> Actual: <%v>", len(node.Status.VolumesAttached))
			}
		}
	}
}

// TestNodeStatusUpdater_UpdateNodeStatusForNode calls setup
// calls UpdateNodeStatusesForNode on testnode-1
// check that asw.GetVolumesToReportAttached reports testnode-2 needs to be reported
// checks that testnode-1 status.volumesAttached is of length 1 and contains the correct volume
func TestNodeStatusUpdater_UpdateNodeStatusForNode(t *testing.T) {
	ctx := context.Background()
	logger := klog.FromContext(ctx)
	asw, fakeKubeClient, nsu := setupNodeStatusUpdate(logger, t)

	err := nsu.UpdateNodeStatusForNode(logger, "testnode-1")
	if err != nil {
		t.Fatalf("UpdateNodeStatuses failed. Expected: <no error> Actual: <%v>", err)
	}

	needToReport := asw.GetVolumesToReportAttached(logger)
	if len(needToReport) != 1 {
		t.Fatalf("len(asw.GetVolumesToReportAttached()) Expected: <1> Actual: <%v>", len(needToReport))
	}
	if _, ok := needToReport["testnode-2"]; !ok {
		t.Fatalf("GetVolumesToReportAttached() did not report correct node Expected: <testnode-2> Actual: <%v>", needToReport)
	}

	node, err := fakeKubeClient.CoreV1().Nodes().Get(ctx, "testnode-1", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Nodes().Get failed. Expected: <no error> Actual: <%v>", err)
	}
	if len(node.Status.VolumesAttached) != 1 {
		t.Fatalf("len(node.Status.VolumesAttached) Expected: <1> Actual: <%v>", len(node.Status.VolumesAttached))
	}
	if node.Status.VolumesAttached[0].Name != "volume-name-1" {
		t.Fatalf("volumeName Expected: <volume-name-1> Actual: <%s>", node.Status.VolumesAttached[0].Name)
	}
}
