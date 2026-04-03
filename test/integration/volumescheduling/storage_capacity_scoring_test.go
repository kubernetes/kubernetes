/*
Copyright 2021 The Kubernetes Authors.

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

package volumescheduling

// This file tests the StorageCapacityScoring feature.

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	testutil "k8s.io/kubernetes/test/integration/util"
)

var (
	waitSSDSC = makeStorageClass("ssd", &modeWait)
	waitHDDSC = makeStorageClass("hdd", &modeWait)
)

const (
	multiDriverAProvisionerName  = "mock-driver-a.kubernetes.io"
	multiDriverBProvisionerName  = "mock-driver-b.kubernetes.io"
	mixedEnabledProvisionerName  = "mock-driver-enabled.kubernetes.io"
	mixedDisabledProvisionerName = "mock-driver-disabled.kubernetes.io"
)

func mergeNodeLabels(node *v1.Node, labels map[string]string) *v1.Node {
	for k, v := range labels {
		node.Labels[k] = v
	}
	return node
}

func setupClusterForStorageCapacityScoring(t *testing.T, nsName string, resyncPeriod time.Duration, provisionDelaySeconds int) *testConfig {
	testCtx := testutil.InitTestSchedulerWithOptions(t, testutil.InitTestAPIServer(t, nsName, nil), resyncPeriod)
	testutil.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	clientset := testCtx.ClientSet
	ns := testCtx.NS.Name

	ctrl, informerFactory, err := initPVController(t, testCtx, provisionDelaySeconds)
	if err != nil {
		t.Fatalf("Failed to create PV controller: %v", err)
	}
	go ctrl.Run(testCtx.Ctx)

	// Start informer factory after all controllers are configured and running.
	informerFactory.Start(testCtx.Ctx.Done())
	informerFactory.WaitForCacheSync(testCtx.Ctx.Done())

	return &testConfig{
		client: clientset,
		ns:     ns,
		stop:   testCtx.Ctx.Done(),
		teardown: func() {
			klog.Infof("test cluster %q start to tear down", ns)
			deleteTestObjects(clientset, ns, metav1.DeleteOptions{})
		},
	}
}

func TestStorageCapacityScoring(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageCapacityScoring, true)

	config := setupClusterForStorageCapacityScoring(t, "storage-capacity-scoring", 0, 0)
	defer config.teardown()

	tests := []struct {
		name         string
		pod          *v1.Pod
		nodes        []*v1.Node
		pvs          []*v1.PersistentVolume
		pvcs         []*v1.PersistentVolumeClaim
		wantNodeName string
	}{
		{
			name: "local volumes with max capacity are preferred",
			pod:  makePod("pod", config.ns, []string{"data"}),
			nodes: []*v1.Node{
				makeNode(0),
				makeNode(1),
				makeNode(2),
			},
			pvs: []*v1.PersistentVolume{
				setPVNodeAffinity(setPVCapacity(makePV("pv-0", waitSSDSC.Name, "", config.ns, "node-0"), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-0"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-1", waitSSDSC.Name, "", config.ns, "node-0"), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-0"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-2", waitSSDSC.Name, "", config.ns, "node-1"), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-1"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-3", waitSSDSC.Name, "", config.ns, "node-1"), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-1"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-4", waitSSDSC.Name, "", config.ns, "node-2"), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-2"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-5", waitSSDSC.Name, "", config.ns, "node-2"), resource.MustParse("50Gi")), map[string][]string{v1.LabelHostname: {"node-2"}}),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				setPVCRequestStorage(makePVC("data", config.ns, &waitSSDSC.Name, ""), resource.MustParse("20Gi")),
			},
			wantNodeName: "node-0",
		},
		{
			name: "local volumes with max capacity are preferred (multiple pvcs)",
			pod:  makePod("pod", config.ns, []string{"data-0", "data-1"}),
			nodes: []*v1.Node{
				makeNode(0),
				makeNode(1),
				makeNode(2),
			},
			pvs: []*v1.PersistentVolume{
				setPVNodeAffinity(setPVCapacity(makePV("pv-0", waitSSDSC.Name, "", config.ns, "node-0"), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-0"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-1", waitSSDSC.Name, "", config.ns, "node-0"), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-0"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-2", waitSSDSC.Name, "", config.ns, "node-1"), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-1"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-3", waitSSDSC.Name, "", config.ns, "node-1"), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-1"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-4", waitSSDSC.Name, "", config.ns, "node-2"), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-2"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-5", waitSSDSC.Name, "", config.ns, "node-2"), resource.MustParse("50Gi")), map[string][]string{v1.LabelHostname: {"node-2"}}),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				setPVCRequestStorage(makePVC("data-0", config.ns, &waitSSDSC.Name, ""), resource.MustParse("80Gi")),
				setPVCRequestStorage(makePVC("data-1", config.ns, &waitSSDSC.Name, ""), resource.MustParse("80Gi")),
			},
			wantNodeName: "node-0",
		},
		{
			name: "local volumes with max capacity are preferred (multiple pvcs, multiple classes)",
			pod:  makePod("pod", config.ns, []string{"data-0", "data-1"}),
			nodes: []*v1.Node{
				makeNode(0),
				makeNode(1),
				makeNode(2),
			},
			pvs: []*v1.PersistentVolume{
				setPVNodeAffinity(setPVCapacity(makePV("pv-0", waitSSDSC.Name, "", config.ns, "node-0"), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-0"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-1", waitHDDSC.Name, "", config.ns, "node-0"), resource.MustParse("200Gi")), map[string][]string{v1.LabelHostname: {"node-0"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-2", waitSSDSC.Name, "", config.ns, "node-1"), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-1"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-3", waitHDDSC.Name, "", config.ns, "node-1"), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-1"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-4", waitSSDSC.Name, "", config.ns, "node-2"), resource.MustParse("100Gi")), map[string][]string{v1.LabelHostname: {"node-2"}}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-5", waitHDDSC.Name, "", config.ns, "node-2"), resource.MustParse("50Gi")), map[string][]string{v1.LabelHostname: {"node-2"}}),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				setPVCRequestStorage(makePVC("data-0", config.ns, &waitSSDSC.Name, ""), resource.MustParse("80Gi")),
				setPVCRequestStorage(makePVC("data-1", config.ns, &waitHDDSC.Name, ""), resource.MustParse("80Gi")),
			},
			wantNodeName: "node-0",
		},
		{
			name: "zonal volumes with max capacity are preferred (multiple pvcs, multiple classes)",
			pod:  makePod("pod", config.ns, []string{"data-0", "data-1"}),
			nodes: []*v1.Node{
				mergeNodeLabels(makeNode(0), map[string]string{
					"topology.kubernetes.io/region": "region-a",
					"topology.kubernetes.io/zone":   "zone-a",
				}),
				mergeNodeLabels(makeNode(1), map[string]string{
					"topology.kubernetes.io/region": "region-b",
					"topology.kubernetes.io/zone":   "zone-b",
				}),
				mergeNodeLabels(makeNode(2), map[string]string{
					"topology.kubernetes.io/region": "region-c",
					"topology.kubernetes.io/zone":   "zone-c",
				}),
			},
			pvs: []*v1.PersistentVolume{
				setPVNodeAffinity(setPVCapacity(makePV("pv-0", waitSSDSC.Name, "", config.ns, ""), resource.MustParse("200Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-a"},
					"topology.kubernetes.io/zone":   {"zone-a"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-1", waitHDDSC.Name, "", config.ns, ""), resource.MustParse("200Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-a"},
					"topology.kubernetes.io/zone":   {"zone-a"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-2", waitSSDSC.Name, "", config.ns, ""), resource.MustParse("100Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-b"},
					"topology.kubernetes.io/zone":   {"zone-b"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-3", waitHDDSC.Name, "", config.ns, ""), resource.MustParse("100Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-b"},
					"topology.kubernetes.io/zone":   {"zone-b"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-4", waitSSDSC.Name, "", config.ns, ""), resource.MustParse("100Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-c"},
					"topology.kubernetes.io/zone":   {"zone-c"},
				}),
				setPVNodeAffinity(setPVCapacity(makePV("pv-5", waitHDDSC.Name, "", config.ns, ""), resource.MustParse("50Gi")), map[string][]string{
					"topology.kubernetes.io/region": {"region-c"},
					"topology.kubernetes.io/zone":   {"zone-c"},
				}),
			},
			pvcs: []*v1.PersistentVolumeClaim{
				setPVCRequestStorage(makePVC("data-0", config.ns, &waitSSDSC.Name, ""), resource.MustParse("80Gi")),
				setPVCRequestStorage(makePVC("data-1", config.ns, &waitHDDSC.Name, ""), resource.MustParse("80Gi")),
			},
			wantNodeName: "node-0",
		},
	}

	c := config.client

	t.Log("Creating StorageClasses")
	classes := map[string]*storagev1.StorageClass{}
	classes[waitSSDSC.Name] = waitSSDSC
	classes[waitHDDSC.Name] = waitHDDSC
	for _, sc := range classes {
		if _, err := c.StorageV1().StorageClasses().Create(context.TODO(), sc, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create StorageClass %q: %v", sc.Name, err)
		}
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Creating Nodes")
			for _, node := range tt.nodes {
				if _, err := c.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create Node %q: %v", node.Name, err)
				}
			}

			t.Log("Creating PVs")
			for _, pv := range tt.pvs {
				if _, err := c.CoreV1().PersistentVolumes().Create(context.TODO(), pv, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create PersistentVolume %q: %v", pv.Name, err)
				}
			}

			// https://github.com/kubernetes/kubernetes/issues/85320
			t.Log("Waiting for PVs to become available to avoid race condition in PV controller")
			for _, pv := range tt.pvs {
				if err := waitForPVPhase(c, pv.Name, v1.VolumeAvailable); err != nil {
					t.Fatalf("failed to wait for PersistentVolume %q to become available: %v", pv.Name, err)
				}
			}

			t.Log("Creating PVCs")
			for _, pvc := range tt.pvcs {
				if _, err := c.CoreV1().PersistentVolumeClaims(config.ns).Create(context.TODO(), pvc, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create PersistentVolumeClaim %q: %v", pvc.Name, err)
				}
			}

			t.Log("Create Pod")
			if _, err := c.CoreV1().Pods(config.ns).Create(context.TODO(), tt.pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create Pod %q: %v", tt.pod.Name, err)
			}
			if err := waitForPodToSchedule(c, tt.pod); err != nil {
				t.Errorf("failed to schedule Pod %q: %v", tt.pod.Name, err)
			}

			t.Log("Verify the assigned node")
			pod, err := c.CoreV1().Pods(config.ns).Get(context.TODO(), tt.pod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("failed to get Pod %q: %v", tt.pod.Name, err)
			}
			if pod.Spec.NodeName != tt.wantNodeName {
				t.Errorf("pod %s assigned node expects %q, got %q", pod.Name, tt.wantNodeName, pod.Spec.NodeName)
			}

			t.Log("Cleanup test objects")
			c.CoreV1().Nodes().DeleteCollection(context.TODO(), deleteOption, metav1.ListOptions{})
			c.CoreV1().Pods(config.ns).DeleteCollection(context.TODO(), deleteOption, metav1.ListOptions{})
			c.CoreV1().PersistentVolumeClaims(config.ns).DeleteCollection(context.TODO(), deleteOption, metav1.ListOptions{})
			c.CoreV1().PersistentVolumes().DeleteCollection(context.TODO(), deleteOption, metav1.ListOptions{})
		})
	}
}

func TestStorageCapacityScoringMultiDriver(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageCapacityScoring, true)

	t.Run("multiple CSI drivers with storage capacity reporting enabled", func(t *testing.T) {
		config := setupClusterForStorageCapacityScoring(t, "scoring-multi-all", 0, 0)
		defer config.teardown()

		c := config.client
		driverA := multiDriverAProvisionerName
		driverB := multiDriverBProvisionerName

		nodes := []*v1.Node{
			mergeNodeLabels(makeNode(1), map[string]string{"topology.kubernetes.io/zone": "zone-a"}),
			mergeNodeLabels(makeNode(2), map[string]string{"topology.kubernetes.io/zone": "zone-b"}),
		}
		for _, node := range nodes {
			if _, err := c.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create Node %q: %v", node.Name, err)
			}
		}

		scA := makeDynamicProvisionerStorageClass("wait-driver-a", &modeWait, nil)
		scA.Provisioner = driverA
		scB := makeDynamicProvisionerStorageClass("wait-driver-b", &modeWait, nil)
		scB.Provisioner = driverB

		for _, sc := range []*storagev1.StorageClass{scA, scB} {
			if _, err := c.StorageV1().StorageClasses().Create(context.TODO(), sc, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create StorageClass %q: %v", sc.Name, err)
			}
		}

		capacityEnabled := true
		for _, driverName := range []string{driverA, driverB} {
			if _, err := c.StorageV1().CSIDrivers().Create(context.TODO(), &storagev1.CSIDriver{
				ObjectMeta: metav1.ObjectMeta{Name: driverName},
				Spec: storagev1.CSIDriverSpec{
					StorageCapacity: &capacityEnabled,
				},
			}, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create CSIDriver %q: %v", driverName, err)
			}
		}

		// Wait until scheduler observes CSIDriver updates.
		time.Sleep(5 * time.Second)

		if _, err := c.StorageV1().CSIStorageCapacities("default").Create(context.TODO(), &storagev1.CSIStorageCapacity{
			ObjectMeta:       metav1.ObjectMeta{GenerateName: "driver-a-capacity-zone-a-"},
			StorageClassName: scA.Name,
			NodeTopology:     &metav1.LabelSelector{MatchLabels: map[string]string{"topology.kubernetes.io/zone": "zone-a"}},
			Capacity:         resource.NewQuantity(12*1024*1024*1024, resource.BinarySI),
		}, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create CSIStorageCapacity for %q in zone-a: %v", driverA, err)
		}
		if _, err := c.StorageV1().CSIStorageCapacities("default").Create(context.TODO(), &storagev1.CSIStorageCapacity{
			ObjectMeta:       metav1.ObjectMeta{GenerateName: "driver-a-capacity-zone-b-"},
			StorageClassName: scA.Name,
			NodeTopology:     &metav1.LabelSelector{MatchLabels: map[string]string{"topology.kubernetes.io/zone": "zone-b"}},
			Capacity:         resource.NewQuantity(6*1024*1024*1024, resource.BinarySI),
		}, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create CSIStorageCapacity for %q in zone-b: %v", driverA, err)
		}

		if _, err := c.StorageV1().CSIStorageCapacities("default").Create(context.TODO(), &storagev1.CSIStorageCapacity{
			ObjectMeta:       metav1.ObjectMeta{GenerateName: "driver-b-capacity-zone-a-"},
			StorageClassName: scB.Name,
			NodeTopology:     &metav1.LabelSelector{MatchLabels: map[string]string{"topology.kubernetes.io/zone": "zone-a"}},
			Capacity:         resource.NewQuantity(6*1024*1024*1024, resource.BinarySI),
		}, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create CSIStorageCapacity for %q in zone-a: %v", driverB, err)
		}
		if _, err := c.StorageV1().CSIStorageCapacities("default").Create(context.TODO(), &storagev1.CSIStorageCapacity{
			ObjectMeta:       metav1.ObjectMeta{GenerateName: "driver-b-capacity-zone-b-"},
			StorageClassName: scB.Name,
			NodeTopology:     &metav1.LabelSelector{MatchLabels: map[string]string{"topology.kubernetes.io/zone": "zone-b"}},
			Capacity:         resource.NewQuantity(12*1024*1024*1024, resource.BinarySI),
		}, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create CSIStorageCapacity for %q in zone-b: %v", driverB, err)
		}

		pvcA := makePVC("pvc-driver-a", config.ns, &scA.Name, "")
		pvcB := makePVC("pvc-driver-b", config.ns, &scB.Name, "")
		for _, pvc := range []*v1.PersistentVolumeClaim{pvcA, pvcB} {
			if _, err := c.CoreV1().PersistentVolumeClaims(config.ns).Create(context.TODO(), pvc, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create PVC %q: %v", pvc.Name, err)
			}
		}

		podA := makePod("pod-driver-a", config.ns, []string{pvcA.Name})
		podB := makePod("pod-driver-b", config.ns, []string{pvcB.Name})
		for _, pod := range []*v1.Pod{podA, podB} {
			if _, err := c.CoreV1().Pods(config.ns).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create Pod %q: %v", pod.Name, err)
			}
		}

		if err := waitForPodToSchedule(c, podA); err != nil {
			t.Fatalf("failed to schedule Pod %q: %v", podA.Name, err)
		}
		if err := waitForPodToSchedule(c, podB); err != nil {
			t.Fatalf("failed to schedule Pod %q: %v", podB.Name, err)
		}

		scheduledPodA, err := c.CoreV1().Pods(config.ns).Get(context.TODO(), podA.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get Pod %q: %v", podA.Name, err)
		}
		scheduledPodB, err := c.CoreV1().Pods(config.ns).Get(context.TODO(), podB.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get Pod %q: %v", podB.Name, err)
		}

		if scheduledPodA.Spec.NodeName != node1 {
			t.Errorf("pod %q expected node %q, got %q", scheduledPodA.Name, node1, scheduledPodA.Spec.NodeName)
		}
		if scheduledPodB.Spec.NodeName != node2 {
			t.Errorf("pod %q expected node %q, got %q", scheduledPodB.Name, node2, scheduledPodB.Spec.NodeName)
		}
	})

	t.Run("mixed CSI driver storage capacity reporting enabled and disabled", func(t *testing.T) {
		config := setupClusterForStorageCapacityScoring(t, "scoring-multi-mixed", 0, 0)
		defer config.teardown()

		c := config.client
		enabledDriver := mixedEnabledProvisionerName
		disabledDriver := mixedDisabledProvisionerName

		nodes := []*v1.Node{
			mergeNodeLabels(makeNode(1), map[string]string{"topology.kubernetes.io/zone": "zone-a"}),
			mergeNodeLabels(makeNode(2), map[string]string{"topology.kubernetes.io/zone": "zone-b"}),
		}
		for _, node := range nodes {
			if _, err := c.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create Node %q: %v", node.Name, err)
			}
		}

		scEnabled := makeDynamicProvisionerStorageClass("wait-driver-enabled", &modeWait, nil)
		scEnabled.Provisioner = enabledDriver
		scDisabled := makeDynamicProvisionerStorageClass("wait-driver-disabled", &modeWait, nil)
		scDisabled.Provisioner = disabledDriver

		for _, sc := range []*storagev1.StorageClass{scEnabled, scDisabled} {
			if _, err := c.StorageV1().StorageClasses().Create(context.TODO(), sc, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create StorageClass %q: %v", sc.Name, err)
			}
		}

		enabled := true
		disabled := false
		for _, driver := range []*storagev1.CSIDriver{
			{ObjectMeta: metav1.ObjectMeta{Name: enabledDriver}, Spec: storagev1.CSIDriverSpec{StorageCapacity: &enabled}},
			{ObjectMeta: metav1.ObjectMeta{Name: disabledDriver}, Spec: storagev1.CSIDriverSpec{StorageCapacity: &disabled}},
		} {
			if _, err := c.StorageV1().CSIDrivers().Create(context.TODO(), driver, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create CSIDriver %q: %v", driver.Name, err)
			}
		}

		// Wait until scheduler observes CSIDriver updates.
		time.Sleep(5 * time.Second)

		if _, err := c.StorageV1().CSIStorageCapacities("default").Create(context.TODO(), &storagev1.CSIStorageCapacity{
			ObjectMeta:       metav1.ObjectMeta{GenerateName: "enabled-driver-capacity-zone-a-"},
			StorageClassName: scEnabled.Name,
			NodeTopology:     &metav1.LabelSelector{MatchLabels: map[string]string{"topology.kubernetes.io/zone": "zone-a"}},
			Capacity:         resource.NewQuantity(12*1024*1024*1024, resource.BinarySI),
		}, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create CSIStorageCapacity for %q: %v", enabledDriver, err)
		}

		pvcEnabled := makePVC("pvc-driver-enabled", config.ns, &scEnabled.Name, "")
		pvcDisabled := makePVC("pvc-driver-disabled", config.ns, &scDisabled.Name, "")
		for _, pvc := range []*v1.PersistentVolumeClaim{pvcEnabled, pvcDisabled} {
			if _, err := c.CoreV1().PersistentVolumeClaims(config.ns).Create(context.TODO(), pvc, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create PVC %q: %v", pvc.Name, err)
			}
		}

		podEnabled := makePod("pod-driver-enabled", config.ns, []string{pvcEnabled.Name})
		podDisabled := makePod("pod-driver-disabled", config.ns, []string{pvcDisabled.Name})
		podDisabled.Spec.NodeSelector = map[string]string{"topology.kubernetes.io/zone": "zone-b"}

		for _, pod := range []*v1.Pod{podEnabled, podDisabled} {
			if _, err := c.CoreV1().Pods(config.ns).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create Pod %q: %v", pod.Name, err)
			}
		}

		if err := waitForPodToSchedule(c, podEnabled); err != nil {
			t.Fatalf("failed to schedule Pod %q: %v", podEnabled.Name, err)
		}
		if err := waitForPodToSchedule(c, podDisabled); err != nil {
			t.Fatalf("failed to schedule Pod %q: %v", podDisabled.Name, err)
		}

		scheduledEnabledPod, err := c.CoreV1().Pods(config.ns).Get(context.TODO(), podEnabled.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get Pod %q: %v", podEnabled.Name, err)
		}
		scheduledDisabledPod, err := c.CoreV1().Pods(config.ns).Get(context.TODO(), podDisabled.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get Pod %q: %v", podDisabled.Name, err)
		}

		if scheduledEnabledPod.Spec.NodeName != node1 {
			t.Errorf("pod %q expected node %q, got %q", scheduledEnabledPod.Name, node1, scheduledEnabledPod.Spec.NodeName)
		}
		if scheduledDisabledPod.Spec.NodeName != node2 {
			t.Errorf("pod %q expected node %q, got %q", scheduledDisabledPod.Name, node2, scheduledDisabledPod.Spec.NodeName)
		}
	})
}

func setPVNodeAffinity(pv *v1.PersistentVolume, keyValues map[string][]string) *v1.PersistentVolume {
	matchExpressions := make([]v1.NodeSelectorRequirement, 0)
	for key, values := range keyValues {
		matchExpressions = append(matchExpressions, v1.NodeSelectorRequirement{
			Key:      key,
			Operator: v1.NodeSelectorOpIn,
			Values:   values,
		})
	}
	pv.Spec.NodeAffinity = &v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: matchExpressions,
				},
			},
		},
	}
	return pv
}

func setPVCapacity(pv *v1.PersistentVolume, capacity resource.Quantity) *v1.PersistentVolume {
	if pv.Spec.Capacity == nil {
		pv.Spec.Capacity = make(v1.ResourceList)
	}
	pv.Spec.Capacity[v1.ResourceName(v1.ResourceStorage)] = capacity
	return pv
}

func setPVCRequestStorage(pvc *v1.PersistentVolumeClaim, request resource.Quantity) *v1.PersistentVolumeClaim {
	pvc.Spec.Resources = v1.VolumeResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceName(v1.ResourceStorage): request,
		},
	}
	return pvc
}
