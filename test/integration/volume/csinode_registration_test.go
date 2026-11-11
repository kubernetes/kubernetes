/*
Copyright The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume/csi/nodeinfomanager"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

// TestCSINodeDriverRegistrations tests how kubelet records CSI drivers in CSINode for
// KEP-6011 (CSI ControllerGetNodeInfo) against kube-apiserver, first with the
// CSIControllerGetNodeInfo feature gate enabled and then, on the same etcd, disabled.
//
// kube-apiserver reads the feature gate of this process when it handles a CSINode
// write, and nodeinfomanager doesn't read it. So the test calls nodeinfomanager
// directly, as kubelet does after it chooses a flow, and the feature gate only
// controls kube-apiserver. Each new nodeinfomanager simulates a kubelet start, and a
// plain CSINode update simulates external-attacher.
func TestCSINodeDriverRegistrations(t *testing.T) {
	const (
		driverName  = "csi.example.com"
		topologyKey = "topology.csi.example.com/zone"
		maxVolumes  = 10
	)
	topology := map[string]string{topologyKey: "zone-1"}
	driver := func(nodeID string) storagev1.CSINodeDriver {
		return storagev1.CSINodeDriver{
			Name:         driverName,
			NodeID:       nodeID,
			TopologyKeys: []string{topologyKey},
			Allocatable:  &storagev1.VolumeNodeResources{Count: ptr.To[int32](maxVolumes)},
		}
	}
	registration := func(nodeID string) storagev1.CSINodeDriverRegistration {
		return storagev1.CSINodeDriverRegistration{Name: driverName, NodeID: nodeID}
	}

	etcdConfig := framework.SharedEtcd()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIControllerGetNodeInfo, true)
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), etcdConfig)
	client := clientset.NewForConfigOrDie(server.ClientConfig)

	t.Run("traditional to controller-side to traditional flow", func(t *testing.T) {
		nodeName := "node-flow-transitions"
		createNode(t, client, nodeName)
		nim := newNodeInfoManager(t, client, nodeName)
		err := nim.InstallCSIDriver(driverName, "id-1", maxVolumes, topology)
		if err != nil {
			t.Fatalf("InstallCSIDriver failed: %v", err)
		}
		expectCSINodeSpec(t, getCSINode(t, client, nodeName), []storagev1.CSINodeDriver{driver("id-1")}, nil)

		// kubelet restarts with the controller-side flow and keeps the drivers entry
		// from the traditional flow.
		nim = newNodeInfoManager(t, client, nodeName)
		err = nim.InstallCSIDriverRegistration(driverName, "id-1")
		if err != nil {
			t.Fatalf("InstallCSIDriverRegistration failed: %v", err)
		}
		csiNode := getCSINode(t, client, nodeName)
		expectCSINodeSpec(t, csiNode, []storagev1.CSINodeDriver{driver("id-1")}, []storagev1.CSINodeDriverRegistration{registration("id-1")})

		// kubelet restarts again with the same node ID and has nothing to update.
		nim = newNodeInfoManager(t, client, nodeName)
		err = nim.InstallCSIDriverRegistration(driverName, "id-1")
		if err != nil {
			t.Fatalf("InstallCSIDriverRegistration after kubelet restart failed: %v", err)
		}
		expectCSINodeUnchanged(t, client, csiNode)

		// kubelet restarts with the traditional flow, for example after its feature gate
		// is disabled. It removes the registration and keeps the matching drivers entry.
		nim = newNodeInfoManager(t, client, nodeName)
		err = nim.InstallCSIDriver(driverName, "id-1", maxVolumes, topology)
		if err != nil {
			t.Fatalf("InstallCSIDriver after the controller-side flow failed: %v", err)
		}
		expectCSINodeSpec(t, getCSINode(t, client, nodeName), []storagev1.CSINodeDriver{driver("id-1")}, nil)
	})

	t.Run("changed node ID in controller-side flow", func(t *testing.T) {
		nodeName := "node-controller-side-node-id-change"
		createNode(t, client, nodeName)
		nim := newNodeInfoManager(t, client, nodeName)
		err := nim.InstallCSIDriver(driverName, "id-1", maxVolumes, topology)
		if err != nil {
			t.Fatalf("InstallCSIDriver failed: %v", err)
		}
		csiNode := getCSINode(t, client, nodeName)

		// kubelet restarts with the controller-side flow, and the driver reports another node ID.
		nim = newNodeInfoManager(t, client, nodeName)
		err = nim.InstallCSIDriverRegistration(driverName, "id-2")
		if err == nil {
			t.Fatal("InstallCSIDriverRegistration with a changed node ID succeeded, want a validation error")
		}
		if !strings.Contains(err.Error(), "must match the nodeID of the drivers entry with the same name") {
			t.Errorf("InstallCSIDriverRegistration failed with an unexpected error: %v", err)
		}
		expectCSINodeUnchanged(t, client, csiNode)

		// kubelet unregisters the driver after the failed registration, and the
		// registration is retried.
		err = nim.UninstallCSIDriver(driverName)
		if err != nil {
			t.Fatalf("UninstallCSIDriver failed: %v", err)
		}
		expectCSINodeSpec(t, getCSINode(t, client, nodeName), nil, nil)
		err = nim.InstallCSIDriverRegistration(driverName, "id-2")
		if err != nil {
			t.Fatalf("InstallCSIDriverRegistration after UninstallCSIDriver failed: %v", err)
		}
		expectCSINodeSpec(t, getCSINode(t, client, nodeName), nil, []storagev1.CSINodeDriverRegistration{registration("id-2")})
	})

	t.Run("changed node ID in traditional flow", func(t *testing.T) {
		nodeName := "node-traditional-node-id-change"
		createNode(t, client, nodeName)
		nim := newNodeInfoManager(t, client, nodeName)
		err := nim.InstallCSIDriverRegistration(driverName, "id-1")
		if err != nil {
			t.Fatalf("InstallCSIDriverRegistration failed: %v", err)
		}

		// external-attacher adds the drivers entry and keeps the registration.
		csiNode := getCSINode(t, client, nodeName)
		csiNode.Spec.Drivers = append(csiNode.Spec.Drivers, driver("id-1"))
		csiNode, err = client.StorageV1().CSINodes().Update(t.Context(), csiNode, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to add the drivers entry to CSINode %s: %v", nodeName, err)
		}
		expectCSINodeSpec(t, csiNode, []storagev1.CSINodeDriver{driver("id-1")}, []storagev1.CSINodeDriverRegistration{registration("id-1")})

		// kubelet restarts with the traditional flow, and the driver reports another node ID.
		nim = newNodeInfoManager(t, client, nodeName)
		err = nim.InstallCSIDriver(driverName, "id-2", maxVolumes, topology)
		if err == nil {
			t.Fatal("InstallCSIDriver with a changed node ID succeeded, want a validation error")
		}
		if !strings.Contains(err.Error(), "field is immutable") {
			t.Errorf("InstallCSIDriver failed with an unexpected error: %v", err)
		}
		expectCSINodeUnchanged(t, client, csiNode)

		// kubelet unregisters the driver after the failed registration, and the
		// registration is retried.
		err = nim.UninstallCSIDriver(driverName)
		if err != nil {
			t.Fatalf("UninstallCSIDriver failed: %v", err)
		}
		expectCSINodeSpec(t, getCSINode(t, client, nodeName), nil, nil)
		err = nim.InstallCSIDriver(driverName, "id-2", maxVolumes, topology)
		if err != nil {
			t.Fatalf("InstallCSIDriver after UninstallCSIDriver failed: %v", err)
		}
		expectCSINodeSpec(t, getCSINode(t, client, nodeName), []storagev1.CSINodeDriver{driver("id-2")}, nil)
	})

	// Leave a pending registration for kubelet to remove after the feature gate is disabled.
	pendingNodeName := "node-pending-registration"
	createNode(t, client, pendingNodeName)
	err := newNodeInfoManager(t, client, pendingNodeName).InstallCSIDriverRegistration(driverName, "id-1")
	if err != nil {
		t.Fatalf("InstallCSIDriverRegistration failed: %v", err)
	}
	server.TearDownFn()

	// Restart kube-apiserver on the same etcd with the feature gate disabled.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIControllerGetNodeInfo, false)
	server = kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), etcdConfig)
	defer server.TearDownFn()
	client = clientset.NewForConfigOrDie(server.ClientConfig)

	t.Run("registration dropped by kube-apiserver", func(t *testing.T) {
		nodeName := "node-dropped-registration"
		createNode(t, client, nodeName)
		nim := newNodeInfoManager(t, client, nodeName)
		err := nim.InstallCSIDriverRegistration(driverName, "id-1")
		if err == nil {
			t.Fatal("InstallCSIDriverRegistration with the feature gate disabled in kube-apiserver succeeded, want an error")
		}
		if !strings.Contains(err.Error(), "dropped the driverRegistrations field") {
			t.Errorf("InstallCSIDriverRegistration failed with an unexpected error: %v", err)
		}
		expectCSINodeSpec(t, getCSINode(t, client, nodeName), nil, nil)
	})

	t.Run("traditional flow removes registration", func(t *testing.T) {
		expectCSINodeSpec(t, getCSINode(t, client, pendingNodeName), nil, []storagev1.CSINodeDriverRegistration{registration("id-1")})

		// kubelet restarts with the traditional flow after the feature gate is disabled.
		nim := newNodeInfoManager(t, client, pendingNodeName)
		err := nim.InstallCSIDriver(driverName, "id-1", maxVolumes, topology)
		if err != nil {
			t.Fatalf("InstallCSIDriver failed: %v", err)
		}
		expectCSINodeSpec(t, getCSINode(t, client, pendingNodeName), []storagev1.CSINodeDriver{driver("id-1")}, nil)
	})
}

// newNodeInfoManager returns the nodeinfomanager of a kubelet that starts on the given node.
func newNodeInfoManager(t *testing.T, client clientset.Interface, nodeName string) nodeinfomanager.Interface {
	t.Helper()
	host := volumetest.NewFakeVolumeHostWithCSINodeName(t, t.TempDir(), client, nil, nodeName, nil, nil)
	return nodeinfomanager.NewNodeInfoManager(types.NodeName(nodeName), host, nil)
}

// createNode creates a Node object, which nodeinfomanager needs to own the CSINode object.
func createNode(t *testing.T, client clientset.Interface, nodeName string) {
	t.Helper()
	node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
	_, err := client.CoreV1().Nodes().Create(t.Context(), node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node %s: %v", nodeName, err)
	}
}

func getCSINode(t *testing.T, client clientset.Interface, nodeName string) *storagev1.CSINode {
	t.Helper()
	csiNode, err := client.StorageV1().CSINodes().Get(t.Context(), nodeName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get CSINode %s: %v", nodeName, err)
	}
	return csiNode
}

// expectCSINodeSpec checks the drivers and driverRegistrations fields of csiNode.
func expectCSINodeSpec(t *testing.T, csiNode *storagev1.CSINode, drivers []storagev1.CSINodeDriver, registrations []storagev1.CSINodeDriverRegistration) {
	t.Helper()
	diff := cmp.Diff(drivers, csiNode.Spec.Drivers, cmpopts.EquateEmpty())
	if diff != "" {
		t.Errorf("Unexpected spec.drivers in CSINode %s (-want,+got):\n%s", csiNode.Name, diff)
	}
	diff = cmp.Diff(registrations, csiNode.Spec.DriverRegistrations, cmpopts.EquateEmpty())
	if diff != "" {
		t.Errorf("Unexpected spec.driverRegistrations in CSINode %s (-want,+got):\n%s", csiNode.Name, diff)
	}
}

// expectCSINodeUnchanged checks that the stored CSINode hasn't been updated since before was read.
func expectCSINodeUnchanged(t *testing.T, client clientset.Interface, before *storagev1.CSINode) {
	t.Helper()
	after := getCSINode(t, client, before.Name)
	if after.ResourceVersion != before.ResourceVersion {
		t.Errorf("CSINode %s was updated (-before,+after):\n%s", before.Name, cmp.Diff(before, after))
	}
}
