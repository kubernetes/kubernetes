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
	"fmt"
	"testing"
	"testing/synctest"
	"time"

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
	"k8s.io/kubernetes/test/utils/ktesting"
)

// setupNodeStatusUpdate creates all the needed objects for testing.
// the initial environment has 2 nodes with no volumes attached
// and adds one volume to attach to each node to the actual state of the world
func setupNodeStatusUpdate(logger klog.Logger, t *testing.T) (cache.ActualStateOfWorld, *fake.Clientset, *nodeStatusUpdater) {
	testNode1 := corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "testnode-1",
			ResourceVersion: "1",
		},
		Status: corev1.NodeStatus{},
	}
	testNode2 := corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "testnode-2",
			ResourceVersion: "1",
		},
		Status: corev1.NodeStatus{},
	}
	volumePluginMgr, _ := volumetesting.GetTestVolumePluginMgr(t)
	asw := cache.NewActualStateOfWorld(volumePluginMgr)
	fakeKubeClient := fake.NewSimpleClientset(&testNode1, &testNode2)
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	nodeInformer := informerFactory.Core().V1().Nodes()
	nsu := NewNodeStatusUpdater(fakeKubeClient, nodeInformer.Lister(), asw).(*nodeStatusUpdater)
	t.Cleanup(func() {
		nsu.queue.ShutDown()
	})

	informerFactory.Start(t.Context().Done())
	informerFactory.WaitForCacheSync(t.Context().Done())
	t.Cleanup(func() {
		informerFactory.Shutdown()
	})

	volumeName1 := corev1.UniqueVolumeName("volume-name-1")
	volumeName2 := corev1.UniqueVolumeName("volume-name-2")
	volumeSpec1 := controllervolumetesting.GetTestVolumeSpec(string(volumeName1), volumeName1)
	volumeSpec2 := controllervolumetesting.GetTestVolumeSpec(string(volumeName2), volumeName2)

	nodeName1 := types.NodeName("testnode-1")
	nodeName2 := types.NodeName("testnode-2")
	devicePath := "fake/device/path"

	_, err := asw.AddVolumeNode(logger, volumeName1, volumeSpec1, nodeName1, devicePath, true)
	if err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", err)
	}
	_, err = asw.AddVolumeNode(logger, volumeName2, volumeSpec2, nodeName2, devicePath, true)
	if err != nil {
		t.Fatalf("AddVolumeNode failed. Expected: <no error> Actual: <%v>", err)
	}

	return asw, fakeKubeClient, nsu
}

func assertVolumeAttached(t *testing.T, ctx context.Context, client *fake.Clientset, nodeName string, volumeName corev1.UniqueVolumeName) {
	t.Helper()
	node, err := client.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Nodes().Get failed. Expected: <no error> Actual: <%v>", err)
	}
	if len(node.Status.VolumesAttached) != 1 {
		t.Fatalf("len(node.Status.VolumesAttached) Expected: <1> Actual: <%v>", len(node.Status.VolumesAttached))
	}
	if node.Status.VolumesAttached[0].Name != volumeName {
		t.Fatalf("volumeName Expected: <%s> Actual: <%s>", volumeName, node.Status.VolumesAttached[0].Name)
	}
}

// TestNodeStatusUpdater_syncNode_TwoNodesUpdate calls setup
// calls nsu.syncNode()
// check that nsu.queue.Len() reports nothing left to attach
// checks that each node status.volumesAttached is of length 1 and contains the correct volume
func TestNodeStatusUpdater_syncNode_TwoNodesUpdate(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	_, fakeKubeClient, nsu := setupNodeStatusUpdate(logger, t)

	nsu.syncNode(logger)
	nsu.syncNode(logger)

	needToReport := nsu.queue.Len()
	if needToReport != 0 {
		t.Fatalf("nsu.queue.Len() Expected: <0> Actual: <%v>", needToReport)
	}

	assertVolumeAttached(t, ctx, fakeKubeClient, "testnode-1", "volume-name-1")
	assertVolumeAttached(t, ctx, fakeKubeClient, "testnode-2", "volume-name-2")
}

func TestNodeStatusUpdater_syncNode_FailureInFirstUpdate(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	_, fakeKubeClient, nsu := setupNodeStatusUpdate(logger, t)

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

	nsu.syncNode(logger)
	nsu.syncNode(logger)

	needToReport := nsu.queue.NumRequeues(types.NodeName(failedNode))
	if needToReport != 1 {
		t.Fatalf("nsu.queue.NumRequeues() Expected: <1> Actual: <%v>", needToReport)
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

// TestNodeStatusUpdater_processNodeVolumes calls setup
// calls processNodeVolumes on testnode-1
// checks that testnode-1 status.volumesAttached is of length 1 and contains the correct volume
func TestNodeStatusUpdater_processNodeVolumes(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	_, fakeKubeClient, nsu := setupNodeStatusUpdate(logger, t)

	err := nsu.processNodeVolumes(logger, "testnode-1")
	if err != nil {
		t.Fatalf("processNodeVolumes failed. Expected: <no error> Actual: <%v>", err)
	}

	assertVolumeAttached(t, ctx, fakeKubeClient, "testnode-1", "volume-name-1")
}

func TestNodeStatusUpdater_UpdateNonExistingNode(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	_, _, nsu := setupNodeStatusUpdate(logger, t)

	err := nsu.processNodeVolumes(logger, "testnode-999")
	if err != nil {
		t.Fatalf("processNodeVolumes failed. Expected: <no error> Actual: <%v>", err)
	}
}

func TestRemove(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	asw, fakeKubeClient, nsu := setupNodeStatusUpdate(logger, t)

	// do attach
	err := nsu.processNodeVolumes(logger, "testnode-1")
	if err != nil {
		t.Fatalf("processNodeVolumes failed. Expected: <no error> Actual: <%v>", err)
	}

	// ask for removal
	removed := asw.RemoveVolumeFromReportAsAttached(logger, "volume-name-1", "testnode-1")
	if removed {
		t.Fatalf("RemoveVolumeFromReportAsAttached returned true, expected false")
	}

	// do removal
	err = nsu.processNodeVolumes(logger, "testnode-1")
	if err != nil {
		t.Fatalf("processNodeVolumes failed. Expected: <no error> Actual: <%v>", err)
	}

	// should be removed now
	removed = asw.RemoveVolumeFromReportAsAttached(logger, "volume-name-1", "testnode-1")
	if !removed {
		t.Fatalf("RemoveVolumeFromReportAsAttached returned false, expected true")
	}

	node, err := fakeKubeClient.CoreV1().Nodes().Get(ctx, "testnode-1", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Nodes().Get failed. Expected: <no error> Actual: <%v>", err)
	}
	if len(node.Status.VolumesAttached) != 0 {
		t.Fatalf("len(node.Status.VolumesAttached) Expected: <0> Actual: <%v>", len(node.Status.VolumesAttached))
	}

}

func TestNodeNotFound(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	asw, fakeKubeClient, nsu := setupNodeStatusUpdate(logger, t)

	// do attach
	err := nsu.processNodeVolumes(logger, "testnode-1")
	if err != nil {
		t.Fatalf("processNodeVolumes failed. Expected: <no error> Actual: <%v>", err)
	}

	err = fakeKubeClient.CoreV1().Nodes().Delete(ctx, "testnode-1", metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Nodes().Delete failed. Expected: <no error> Actual: <%v>", err)
	}

	// ask for removal
	removed := asw.RemoveVolumeFromReportAsAttached(logger, "volume-name-1", "testnode-1")
	if removed {
		t.Fatalf("RemoveVolumeFromReportAsAttached returned true, expected false")
	}

	// do removal
	err = nsu.processNodeVolumes(logger, "testnode-1")
	if err != nil {
		t.Fatalf("processNodeVolumes failed. Expected: <no error> Actual: <%v>", err)
	}

	// should be removed now even if the node is gone
	removed = asw.RemoveVolumeFromReportAsAttached(logger, "volume-name-1", "testnode-1")
	if !removed {
		t.Fatalf("RemoveVolumeFromReportAsAttached returned false, expected true")
	}
}

func TestNotRemoveInUseVolumes(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

	synctest.Test(t, func(t *testing.T) {
		asw, fakeKubeClient, nsu := setupNodeStatusUpdate(logger, t)
		// do attach
		err := nsu.processNodeVolumes(logger, "testnode-1")
		if err != nil {
			t.Fatalf("processNodeVolumes failed. Expected: <no error> Actual: <%v>", err)
		}

		// mark volume as in-use
		node, err := fakeKubeClient.CoreV1().Nodes().Get(ctx, "testnode-1", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Nodes().Get failed. Expected: <no error> Actual: <%v>", err)
		}
		node.Status.VolumesInUse = []corev1.UniqueVolumeName{"volume-name-1"}
		_, err = fakeKubeClient.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Nodes().UpdateStatus failed. Expected: <no error> Actual: <%v>", err)
		}
		synctest.Wait() // wait for informer sync

		// ask for removal
		removed := asw.RemoveVolumeFromReportAsAttached(logger, "volume-name-1", "testnode-1")
		if removed {
			t.Fatalf("RemoveVolumeFromReportAsAttached returned true, expected false")
		}

		// do removal, will not remove in-use volume
		err = nsu.processNodeVolumes(logger, "testnode-1")
		if err != nil {
			t.Fatalf("processNodeVolumes failed. Expected: <no error> Actual: <%v>", err)
		}

		assertVolumeAttached(t, ctx, fakeKubeClient, "testnode-1", "volume-name-1")

		// should still not removed
		removed = asw.RemoveVolumeFromReportAsAttached(logger, "volume-name-1", "testnode-1")
		if removed {
			t.Fatalf("RemoveVolumeFromReportAsAttached returned true, expected false")
		}
	})
}

func TestRun(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	_, _, nsu := setupNodeStatusUpdate(logger, t)

	go func() {
		time.Sleep(1 * time.Second)
		ctx.Cancel("test")
	}()
	nsu.Run(ctx, 2)
}
