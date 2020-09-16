/*
Copyright 2018 The Kubernetes Authors.

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

/*
 * This file defines various csi volume test drivers for TestSuites.
 *
 * There are two ways, how to prepare test drivers:
 * 1) With containerized server (NFS, Ceph, Gluster, iSCSI, ...)
 * It creates a server pod which defines one volume for the tests.
 * These tests work only when privileged containers are allowed, exporting
 * various filesystems (NFS, GlusterFS, ...) usually needs some mounting or
 * other privileged magic in the server pod.
 *
 * Note that the server containers are for testing purposes only and should not
 * be used in production.
 *
 * 2) With server or cloud provider outside of Kubernetes (Cinder, GCE, AWS, Azure, ...)
 * Appropriate server or cloud provider must exist somewhere outside
 * the tested Kubernetes cluster. CreateVolume will create a new volume to be
 * used in the TestSuites for inlineVolume or DynamicPV tests.
 */

package drivers

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"gopkg.in/yaml.v2"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	// GCEPDCSIDriverName is the name of GCE Persistent Disk CSI driver
	GCEPDCSIDriverName = "pd.csi.storage.gke.io"
	// GCEPDCSIZoneTopologyKey is the key of GCE Persistent Disk CSI zone topology
	GCEPDCSIZoneTopologyKey = "topology.gke.io/zone"
)

// hostpathCSI
type hostpathCSIDriver struct {
	driverInfo       testsuites.DriverInfo
	manifests        []string
	cleanupHandle    framework.CleanupActionHandle
	volumeAttributes []map[string]string
}

func initHostPathCSIDriver(name string, capabilities map[testsuites.Capability]bool, volumeAttributes []map[string]string, manifests ...string) testsuites.TestDriver {
	return &hostpathCSIDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        name,
			FeatureTag:  "",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
			Capabilities: capabilities,
			StressTestOptions: &testsuites.StressTestOptions{
				NumPods:     10,
				NumRestarts: 10,
			},
			VolumeSnapshotStressTestOptions: &testsuites.VolumeSnapshotStressTestOptions{
				NumPods:      10,
				NumSnapshots: 10,
			},
		},
		manifests:        manifests,
		volumeAttributes: volumeAttributes,
	}
}

var _ testsuites.TestDriver = &hostpathCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &hostpathCSIDriver{}
var _ testsuites.SnapshottableTestDriver = &hostpathCSIDriver{}
var _ testsuites.EphemeralTestDriver = &hostpathCSIDriver{}

// InitHostPathCSIDriver returns hostpathCSIDriver that implements TestDriver interface
func InitHostPathCSIDriver() testsuites.TestDriver {
	capabilities := map[testsuites.Capability]bool{
		testsuites.CapPersistence:         true,
		testsuites.CapSnapshotDataSource:  true,
		testsuites.CapMultiPODs:           true,
		testsuites.CapBlock:               true,
		testsuites.CapPVCDataSource:       true,
		testsuites.CapControllerExpansion: true,
		testsuites.CapSingleNodeVolume:    true,
		testsuites.CapVolumeLimits:        true,
	}
	return initHostPathCSIDriver("csi-hostpath",
		capabilities,
		// Volume attributes don't matter, but we have to provide at least one map.
		[]map[string]string{
			{"foo": "bar"},
		},
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-snapshotter/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-resizer/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-attacher.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-driverinfo.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-plugin.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-provisioner.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-resizer.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-snapshotter.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/e2e-test-rbac.yaml",
	)
}

func (h *hostpathCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &h.driverInfo
}

func (h *hostpathCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	if pattern.VolType == testpatterns.CSIInlineVolume && len(h.volumeAttributes) == 0 {
		e2eskipper.Skipf("%s has no volume attributes defined, doesn't support ephemeral inline volumes", h.driverInfo.Name)
	}
}

func (h *hostpathCSIDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := config.GetUniqueDriverName()
	parameters := map[string]string{}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", provisioner)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (h *hostpathCSIDriver) GetVolume(config *testsuites.PerTestConfig, volumeNumber int) (map[string]string, bool, bool) {
	return h.volumeAttributes[volumeNumber%len(h.volumeAttributes)], false /* not shared */, false /* read-write */
}

func (h *hostpathCSIDriver) GetCSIDriverName(config *testsuites.PerTestConfig) string {
	return config.GetUniqueDriverName()
}

func (h *hostpathCSIDriver) GetSnapshotClass(config *testsuites.PerTestConfig) *unstructured.Unstructured {
	snapshotter := config.GetUniqueDriverName()
	parameters := map[string]string{}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-vsc", snapshotter)

	return testsuites.GetSnapshotClass(snapshotter, parameters, ns, suffix)
}

func (h *hostpathCSIDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	// Create secondary namespace which will be used for creating driver
	driverNamespace := utils.CreateDriverNamespace(f)
	ns2 := driverNamespace.Name
	ns1 := f.Namespace.Name

	ginkgo.By(fmt.Sprintf("deploying %s driver", h.driverInfo.Name))
	cancelLogging := testsuites.StartPodLogs(f, driverNamespace)
	cs := f.ClientSet

	// The hostpath CSI driver only works when everything runs on the same node.
	node, err := e2enode.GetRandomReadySchedulableNode(cs)
	framework.ExpectNoError(err)
	config := &testsuites.PerTestConfig{
		Driver:              h,
		Prefix:              "hostpath",
		Framework:           f,
		ClientNodeSelection: e2epod.NodeSelection{Name: node.Name},
		DriverNamespace:     driverNamespace,
	}

	o := utils.PatchCSIOptions{
		OldDriverName:            h.driverInfo.Name,
		NewDriverName:            config.GetUniqueDriverName(),
		DriverContainerName:      "hostpath",
		DriverContainerArguments: []string{"--drivername=" + config.GetUniqueDriverName()},
		ProvisionerContainerName: "csi-provisioner",
		SnapshotterContainerName: "csi-snapshotter",
		NodeName:                 node.Name,
	}
	cleanup, err := utils.CreateFromManifests(config.Framework, driverNamespace, func(item interface{}) error {
		return utils.PatchCSIDeployment(config.Framework, o, item)
	}, h.manifests...)

	if err != nil {
		framework.Failf("deploying %s driver: %v", h.driverInfo.Name, err)
	}

	// Cleanup CSI driver and namespaces. This function needs to be idempotent and can be
	// concurrently called from defer (or AfterEach) and AfterSuite action hooks.
	cleanupFunc := func() {
		ginkgo.By(fmt.Sprintf("deleting the test namespace: %s", ns1))
		// Delete the primary namespace but its okay to fail here because this namespace will
		// also be deleted by framework.Aftereach hook
		tryFunc(func() { f.DeleteNamespace(ns1) })

		ginkgo.By("uninstalling csi mock driver")
		tryFunc(cleanup)
		tryFunc(cancelLogging)

		ginkgo.By(fmt.Sprintf("deleting the driver namespace: %s", ns2))
		tryFunc(func() { f.DeleteNamespace(ns2) })
		// cleanup function has already ran and hence we don't need to run it again.
		// We do this as very last action because in-case defer(or AfterEach) races
		// with AfterSuite and test routine gets killed then this block still
		// runs in AfterSuite
		framework.RemoveCleanupAction(h.cleanupHandle)

	}
	h.cleanupHandle = framework.AddCleanupAction(cleanupFunc)

	return config, cleanupFunc
}

// mockCSI
type mockCSIDriver struct {
	driverInfo          testsuites.DriverInfo
	manifests           []string
	podInfo             *bool
	storageCapacity     *bool
	attachable          bool
	attachLimit         int
	enableTopology      bool
	enableNodeExpansion bool
	cleanupHandle       framework.CleanupActionHandle
	javascriptHooks     map[string]string
}

// CSIMockDriverOpts defines options used for csi driver
type CSIMockDriverOpts struct {
	RegisterDriver      bool
	DisableAttach       bool
	PodInfo             *bool
	StorageCapacity     *bool
	AttachLimit         int
	EnableTopology      bool
	EnableResizing      bool
	EnableNodeExpansion bool
	EnableSnapshot      bool
	JavascriptHooks     map[string]string
}

var _ testsuites.TestDriver = &mockCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &mockCSIDriver{}
var _ testsuites.SnapshottableTestDriver = &mockCSIDriver{}

// InitMockCSIDriver returns a mockCSIDriver that implements TestDriver interface
func InitMockCSIDriver(driverOpts CSIMockDriverOpts) testsuites.TestDriver {
	driverManifests := []string{
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-resizer/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-snapshotter/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/mock/csi-mock-rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/mock/csi-storageclass.yaml",
		"test/e2e/testing-manifests/storage-csi/mock/csi-mock-driver.yaml",
	}

	if driverOpts.RegisterDriver {
		driverManifests = append(driverManifests, "test/e2e/testing-manifests/storage-csi/mock/csi-mock-driverinfo.yaml")
	}

	if !driverOpts.DisableAttach {
		driverManifests = append(driverManifests, "test/e2e/testing-manifests/storage-csi/mock/csi-mock-driver-attacher.yaml")
	}

	if driverOpts.EnableResizing {
		driverManifests = append(driverManifests, "test/e2e/testing-manifests/storage-csi/mock/csi-mock-driver-resizer.yaml")
	}

	if driverOpts.EnableSnapshot {
		driverManifests = append(driverManifests, "test/e2e/testing-manifests/storage-csi/mock/csi-mock-driver-snapshotter.yaml")
	}

	return &mockCSIDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        "csi-mock",
			FeatureTag:  "",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence:  false,
				testsuites.CapFsGroup:      false,
				testsuites.CapExec:         false,
				testsuites.CapVolumeLimits: true,
			},
		},
		manifests:           driverManifests,
		podInfo:             driverOpts.PodInfo,
		storageCapacity:     driverOpts.StorageCapacity,
		enableTopology:      driverOpts.EnableTopology,
		attachable:          !driverOpts.DisableAttach,
		attachLimit:         driverOpts.AttachLimit,
		enableNodeExpansion: driverOpts.EnableNodeExpansion,
		javascriptHooks:     driverOpts.JavascriptHooks,
	}
}

func (m *mockCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &m.driverInfo
}

func (m *mockCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (m *mockCSIDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := config.GetUniqueDriverName()
	parameters := map[string]string{}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", provisioner)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (m *mockCSIDriver) GetSnapshotClass(config *testsuites.PerTestConfig) *unstructured.Unstructured {
	parameters := map[string]string{}
	snapshotter := m.driverInfo.Name + "-" + config.Framework.UniqueName
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-vsc", snapshotter)

	return testsuites.GetSnapshotClass(snapshotter, parameters, ns, suffix)
}

func (m *mockCSIDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	// Create secondary namespace which will be used for creating driver
	driverNamespace := utils.CreateDriverNamespace(f)
	ns2 := driverNamespace.Name
	ns1 := f.Namespace.Name

	ginkgo.By("deploying csi mock driver")
	cancelLogging := testsuites.StartPodLogs(f, driverNamespace)
	cs := f.ClientSet

	// pods should be scheduled on the node
	node, err := e2enode.GetRandomReadySchedulableNode(cs)
	framework.ExpectNoError(err)
	config := &testsuites.PerTestConfig{
		Driver:              m,
		Prefix:              "mock",
		Framework:           f,
		ClientNodeSelection: e2epod.NodeSelection{Name: node.Name},
		DriverNamespace:     driverNamespace,
	}

	containerArgs := []string{"--name=csi-mock-" + f.UniqueName}
	if !m.attachable {
		containerArgs = append(containerArgs, "--disable-attach")
	}

	if m.enableTopology {
		containerArgs = append(containerArgs, "--enable-topology")
	}

	if m.attachLimit > 0 {
		containerArgs = append(containerArgs, "--attach-limit", strconv.Itoa(m.attachLimit))
	}

	if m.enableNodeExpansion {
		containerArgs = append(containerArgs, "--node-expand-required=true")
	}

	// Create a config map with javascript hooks. Create it even when javascriptHooks
	// are empty, so we can unconditionally add it to the mock pod.
	const hooksConfigMapName = "mock-driver-hooks"
	hooksYaml, err := yaml.Marshal(m.javascriptHooks)
	framework.ExpectNoError(err)
	hooks := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: hooksConfigMapName,
		},
		Data: map[string]string{
			"hooks.yaml": string(hooksYaml),
		},
	}

	_, err = f.ClientSet.CoreV1().ConfigMaps(ns2).Create(context.TODO(), hooks, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	if len(m.javascriptHooks) > 0 {
		containerArgs = append(containerArgs, "--hooks-file=/etc/hooks/hooks.yaml")
	}

	o := utils.PatchCSIOptions{
		OldDriverName:            "csi-mock",
		NewDriverName:            "csi-mock-" + f.UniqueName,
		DriverContainerName:      "mock",
		DriverContainerArguments: containerArgs,
		ProvisionerContainerName: "csi-provisioner",
		NodeName:                 node.Name,
		PodInfo:                  m.podInfo,
		StorageCapacity:          m.storageCapacity,
		CanAttach:                &m.attachable,
		VolumeLifecycleModes: &[]storagev1.VolumeLifecycleMode{
			storagev1.VolumeLifecyclePersistent,
			storagev1.VolumeLifecycleEphemeral,
		},
	}
	cleanup, err := utils.CreateFromManifests(f, driverNamespace, func(item interface{}) error {
		return utils.PatchCSIDeployment(f, o, item)
	}, m.manifests...)

	if err != nil {
		framework.Failf("deploying csi mock driver: %v", err)
	}

	// Cleanup CSI driver and namespaces. This function needs to be idempotent and can be
	// concurrently called from defer (or AfterEach) and AfterSuite action hooks.
	cleanupFunc := func() {
		ginkgo.By(fmt.Sprintf("deleting the test namespace: %s", ns1))
		// Delete the primary namespace but its okay to fail here because this namespace will
		// also be deleted by framework.Aftereach hook
		tryFunc(func() { f.DeleteNamespace(ns1) })

		ginkgo.By("uninstalling csi mock driver")
		tryFunc(func() {
			err := f.ClientSet.CoreV1().ConfigMaps(ns2).Delete(context.TODO(), hooksConfigMapName, metav1.DeleteOptions{})
			if err != nil {
				framework.Logf("deleting failed: %s", err)
			}
		})

		tryFunc(cleanup)
		tryFunc(cancelLogging)
		ginkgo.By(fmt.Sprintf("deleting the driver namespace: %s", ns2))
		tryFunc(func() { f.DeleteNamespace(ns2) })
		// cleanup function has already ran and hence we don't need to run it again.
		// We do this as very last action because in-case defer(or AfterEach) races
		// with AfterSuite and test routine gets killed then this block still
		// runs in AfterSuite
		framework.RemoveCleanupAction(m.cleanupHandle)

	}

	m.cleanupHandle = framework.AddCleanupAction(cleanupFunc)

	return config, cleanupFunc
}

// gce-pd
type gcePDCSIDriver struct {
	driverInfo    testsuites.DriverInfo
	cleanupHandle framework.CleanupActionHandle
}

var _ testsuites.TestDriver = &gcePDCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &gcePDCSIDriver{}
var _ testsuites.SnapshottableTestDriver = &gcePDCSIDriver{}

// InitGcePDCSIDriver returns gcePDCSIDriver that implements TestDriver interface
func InitGcePDCSIDriver() testsuites.TestDriver {
	return &gcePDCSIDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        GCEPDCSIDriverName,
			FeatureTag:  "[Serial]",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext2",
				"ext3",
				"ext4",
				"xfs",
			),
			SupportedMountOption: sets.NewString("debug", "nouid32"),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapBlock:       true,
				testsuites.CapFsGroup:     true,
				testsuites.CapExec:        true,
				testsuites.CapMultiPODs:   true,
				// GCE supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				testsuites.CapVolumeLimits:        false,
				testsuites.CapTopology:            true,
				testsuites.CapControllerExpansion: true,
				testsuites.CapNodeExpansion:       true,
				testsuites.CapSnapshotDataSource:  true,
			},
			RequiredAccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			TopologyKeys:        []string{GCEPDCSIZoneTopologyKey},
			StressTestOptions: &testsuites.StressTestOptions{
				NumPods:     10,
				NumRestarts: 10,
			},
			VolumeSnapshotStressTestOptions: &testsuites.VolumeSnapshotStressTestOptions{
				// GCE only allows for one snapshot per volume to be created at a time,
				// which can cause test timeouts. We reduce the likelihood of test timeouts
				// by increasing the number of pods (and volumes) and reducing the number
				// of snapshots per volume.
				NumPods:      20,
				NumSnapshots: 2,
			},
		},
	}
}

func (g *gcePDCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &g.driverInfo
}

func (g *gcePDCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("gce", "gke")
	if pattern.FsType == "xfs" {
		e2eskipper.SkipUnlessNodeOSDistroIs("ubuntu", "custom")
	}
	if pattern.FeatureTag == "[sig-windows]" {
		e2eskipper.Skipf("Skipping tests for windows since CSI does not support it yet")
	}
}

func (g *gcePDCSIDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	ns := config.Framework.Namespace.Name
	provisioner := g.driverInfo.Name
	suffix := fmt.Sprintf("%s-sc", g.driverInfo.Name)

	parameters := map[string]string{"type": "pd-standard"}
	if fsType != "" {
		parameters["csi.storage.k8s.io/fstype"] = fsType
	}
	delayedBinding := storagev1.VolumeBindingWaitForFirstConsumer

	return testsuites.GetStorageClass(provisioner, parameters, &delayedBinding, ns, suffix)
}

func (g *gcePDCSIDriver) GetSnapshotClass(config *testsuites.PerTestConfig) *unstructured.Unstructured {
	parameters := map[string]string{}
	snapshotter := g.driverInfo.Name
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-vsc", snapshotter)

	return testsuites.GetSnapshotClass(snapshotter, parameters, ns, suffix)
}

func (g *gcePDCSIDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	ginkgo.By("deploying csi gce-pd driver")
	// Create secondary namespace which will be used for creating driver
	driverNamespace := utils.CreateDriverNamespace(f)
	ns2 := driverNamespace.Name
	ns1 := f.Namespace.Name

	cancelLogging := testsuites.StartPodLogs(f, driverNamespace)
	// It would be safer to rename the gcePD driver, but that
	// hasn't been done before either and attempts to do so now led to
	// errors during driver registration, therefore it is disabled
	// by passing a nil function below.
	//
	// These are the options which would have to be used:
	// o := utils.PatchCSIOptions{
	// 	OldDriverName:            g.driverInfo.Name,
	// 	NewDriverName:            testsuites.GetUniqueDriverName(g),
	// 	DriverContainerName:      "gce-driver",
	// 	ProvisionerContainerName: "csi-external-provisioner",
	// }
	createGCESecrets(f.ClientSet, ns2)

	manifests := []string{
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/csi-controller-rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/node_ds.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/controller_ss.yaml",
	}

	cleanup, err := utils.CreateFromManifests(f, driverNamespace, nil, manifests...)
	if err != nil {
		framework.Failf("deploying csi gce-pd driver: %v", err)
	}

	if err = WaitForCSIDriverRegistrationOnAllNodes(GCEPDCSIDriverName, f.ClientSet); err != nil {
		framework.Failf("waiting for csi driver node registration on: %v", err)
	}

	// Cleanup CSI driver and namespaces. This function needs to be idempotent and can be
	// concurrently called from defer (or AfterEach) and AfterSuite action hooks.
	cleanupFunc := func() {
		ginkgo.By(fmt.Sprintf("deleting the test namespace: %s", ns1))
		// Delete the primary namespace but its okay to fail here because this namespace will
		// also be deleted by framework.Aftereach hook
		tryFunc(func() { f.DeleteNamespace(ns1) })

		ginkgo.By("uninstalling csi mock driver")
		tryFunc(cleanup)
		tryFunc(cancelLogging)

		ginkgo.By(fmt.Sprintf("deleting the driver namespace: %s", ns2))
		tryFunc(func() { f.DeleteNamespace(ns2) })
		// cleanup function has already ran and hence we don't need to run it again.
		// We do this as very last action because in-case defer(or AfterEach) races
		// with AfterSuite and test routine gets killed then this block still
		// runs in AfterSuite
		framework.RemoveCleanupAction(g.cleanupHandle)

	}
	g.cleanupHandle = framework.AddCleanupAction(cleanupFunc)

	return &testsuites.PerTestConfig{
		Driver:          g,
		Prefix:          "gcepd",
		Framework:       f,
		DriverNamespace: driverNamespace,
	}, cleanupFunc
}

// WaitForCSIDriverRegistrationOnAllNodes waits for the CSINode object to be updated
// with the given driver on all schedulable nodes.
func WaitForCSIDriverRegistrationOnAllNodes(driverName string, cs clientset.Interface) error {
	nodes, err := e2enode.GetReadySchedulableNodes(cs)
	if err != nil {
		return err
	}
	for _, node := range nodes.Items {
		if err := WaitForCSIDriverRegistrationOnNode(node.Name, driverName, cs); err != nil {
			return err
		}
	}
	return nil
}

// WaitForCSIDriverRegistrationOnNode waits for the CSINode object generated by the node-registrar on a certain node
func WaitForCSIDriverRegistrationOnNode(nodeName string, driverName string, cs clientset.Interface) error {
	framework.Logf("waiting for CSIDriver %v to register on node %v", driverName, nodeName)

	// About 8.6 minutes timeout
	backoff := wait.Backoff{
		Duration: 2 * time.Second,
		Factor:   1.5,
		Steps:    12,
	}

	waitErr := wait.ExponentialBackoff(backoff, func() (bool, error) {
		csiNode, err := cs.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		for _, driver := range csiNode.Spec.Drivers {
			if driver.Name == driverName {
				return true, nil
			}
		}
		return false, nil
	})
	if waitErr != nil {
		return fmt.Errorf("error waiting for CSI driver %s registration on node %s: %v", driverName, nodeName, waitErr)
	}
	return nil
}

func tryFunc(f func()) error {
	var err error
	if f == nil {
		return nil
	}
	defer func() {
		if recoverError := recover(); recoverError != nil {
			err = fmt.Errorf("%v", recoverError)
		}
	}()
	f()
	return err
}
