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
	"fmt"
	"math/rand"
	"strconv"

	. "github.com/onsi/ginkgo"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	GCEPDCSIProvisionerName = "pd.csi.storage.gke.io"
	GCEPDCSIZoneTopologyKey = "topology.gke.io/zone"
)

// hostpathCSI
type hostpathCSIDriver struct {
	driverInfo testsuites.DriverInfo
	manifests  []string
}

func initHostPathCSIDriver(name string, capabilities map[testsuites.Capability]bool, manifests ...string) testsuites.TestDriver {
	return &hostpathCSIDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        name,
			FeatureTag:  "",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: capabilities,
		},
		manifests: manifests,
	}
}

var _ testsuites.TestDriver = &hostpathCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &hostpathCSIDriver{}
var _ testsuites.SnapshottableTestDriver = &hostpathCSIDriver{}

// InitHostPathCSIDriver returns hostpathCSIDriver that implements TestDriver interface
func InitHostPathCSIDriver() testsuites.TestDriver {
	return initHostPathCSIDriver("csi-hostpath",
		map[testsuites.Capability]bool{testsuites.CapPersistence: true, testsuites.CapDataSource: true,
			testsuites.CapMultiPODs: true, testsuites.CapBlock: true},
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-snapshotter/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-attacher.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-provisioner.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-snapshotter.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpathplugin.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/e2e-test-rbac.yaml",
	)
}

func (h *hostpathCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &h.driverInfo
}

func (h *hostpathCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (h *hostpathCSIDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := config.GetUniqueDriverName()
	parameters := map[string]string{}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", provisioner)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (h *hostpathCSIDriver) GetSnapshotClass(config *testsuites.PerTestConfig) *unstructured.Unstructured {
	snapshotter := config.GetUniqueDriverName()
	parameters := map[string]string{}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-vsc", snapshotter)

	return testsuites.GetSnapshotClass(snapshotter, parameters, ns, suffix)
}

func (h *hostpathCSIDriver) GetClaimSize() string {
	return "5Gi"
}

func (h *hostpathCSIDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	By(fmt.Sprintf("deploying %s driver", h.driverInfo.Name))
	cancelLogging := testsuites.StartPodLogs(f)
	cs := f.ClientSet

	// The hostpath CSI driver only works when everything runs on the same node.
	nodes := framework.GetReadySchedulableNodesOrDie(cs)
	nodeName := nodes.Items[rand.Intn(len(nodes.Items))].Name
	config := &testsuites.PerTestConfig{
		Driver:         h,
		Prefix:         "hostpath",
		Framework:      f,
		ClientNodeName: nodeName,
	}

	// TODO (?): the storage.csi.image.version and storage.csi.image.registry
	// settings are ignored for this test. We could patch the image definitions.
	o := utils.PatchCSIOptions{
		OldDriverName:            h.driverInfo.Name,
		NewDriverName:            config.GetUniqueDriverName(),
		DriverContainerName:      "hostpath",
		DriverContainerArguments: []string{"--drivername=" + config.GetUniqueDriverName()},
		ProvisionerContainerName: "csi-provisioner",
		SnapshotterContainerName: "csi-snapshotter",
		NodeName:                 nodeName,
	}
	cleanup, err := config.Framework.CreateFromManifests(func(item interface{}) error {
		return utils.PatchCSIDeployment(config.Framework, o, item)
	},
		h.manifests...)
	if err != nil {
		framework.Failf("deploying %s driver: %v", h.driverInfo.Name, err)
	}

	return config, func() {
		By(fmt.Sprintf("uninstalling %s driver", h.driverInfo.Name))
		cleanup()
		cancelLogging()
	}
}

// mockCSI
type mockCSIDriver struct {
	driverInfo          testsuites.DriverInfo
	manifests           []string
	podInfo             *bool
	attachable          bool
	attachLimit         int
	enableNodeExpansion bool
}

// CSIMockDriverOpts defines options used for csi driver
type CSIMockDriverOpts struct {
	RegisterDriver      bool
	DisableAttach       bool
	PodInfo             *bool
	AttachLimit         int
	EnableResizing      bool
	EnableNodeExpansion bool
}

var _ testsuites.TestDriver = &mockCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &mockCSIDriver{}

// InitMockCSIDriver returns a mockCSIDriver that implements TestDriver interface
func InitMockCSIDriver(driverOpts CSIMockDriverOpts) testsuites.TestDriver {
	driverManifests := []string{
		"test/e2e/testing-manifests/storage-csi/cluster-driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-resizer/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/mock/csi-mock-rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/mock/csi-storageclass.yaml",
		"test/e2e/testing-manifests/storage-csi/mock/csi-mock-driver.yaml",
	}

	if driverOpts.RegisterDriver {
		driverManifests = append(driverManifests, "test/e2e/testing-manifests/storage-csi/mock/csi-mock-cluster-driver-registrar.yaml")
	}

	if !driverOpts.DisableAttach {
		driverManifests = append(driverManifests, "test/e2e/testing-manifests/storage-csi/mock/csi-mock-driver-attacher.yaml")
	}

	if driverOpts.EnableResizing {
		driverManifests = append(driverManifests, "test/e2e/testing-manifests/storage-csi/mock/csi-mock-driver-resizer.yaml")
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
				testsuites.CapPersistence: false,
				testsuites.CapFsGroup:     false,
				testsuites.CapExec:        false,
			},
		},
		manifests:           driverManifests,
		podInfo:             driverOpts.PodInfo,
		attachable:          !driverOpts.DisableAttach,
		attachLimit:         driverOpts.AttachLimit,
		enableNodeExpansion: driverOpts.EnableNodeExpansion,
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

func (m *mockCSIDriver) GetClaimSize() string {
	return "5Gi"
}

func (m *mockCSIDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	By("deploying csi mock driver")
	cancelLogging := testsuites.StartPodLogs(f)
	cs := f.ClientSet

	// pods should be scheduled on the node
	nodes := framework.GetReadySchedulableNodesOrDie(cs)
	node := nodes.Items[rand.Intn(len(nodes.Items))]
	config := &testsuites.PerTestConfig{
		Driver:         m,
		Prefix:         "mock",
		Framework:      f,
		ClientNodeName: node.Name,
	}

	containerArgs := []string{"--name=csi-mock-" + f.UniqueName}
	if !m.attachable {
		containerArgs = append(containerArgs, "--disable-attach")
	}

	if m.attachLimit > 0 {
		containerArgs = append(containerArgs, "--attach-limit", strconv.Itoa(m.attachLimit))
	}

	if m.enableNodeExpansion {
		containerArgs = append(containerArgs, "--node-expand-required=true")
	}

	// TODO (?): the storage.csi.image.version and storage.csi.image.registry
	// settings are ignored for this test. We could patch the image definitions.
	o := utils.PatchCSIOptions{
		OldDriverName:                 "csi-mock",
		NewDriverName:                 "csi-mock-" + f.UniqueName,
		DriverContainerName:           "mock",
		DriverContainerArguments:      containerArgs,
		ProvisionerContainerName:      "csi-provisioner",
		ClusterRegistrarContainerName: "csi-cluster-driver-registrar",
		NodeName:                      config.ClientNodeName,
		PodInfo:                       m.podInfo,
	}
	cleanup, err := f.CreateFromManifests(func(item interface{}) error {
		return utils.PatchCSIDeployment(f, o, item)
	},
		m.manifests...)
	if err != nil {
		framework.Failf("deploying csi mock driver: %v", err)
	}

	return config, func() {
		By("uninstalling csi mock driver")
		cleanup()
		cancelLogging()
	}
}

// InitHostPathV0CSIDriver returns a variant of hostpathCSIDriver with different manifests.
func InitHostPathV0CSIDriver() testsuites.TestDriver {
	return initHostPathCSIDriver("csi-hostpath-v0",
		map[testsuites.Capability]bool{testsuites.CapPersistence: true, testsuites.CapMultiPODs: true},
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath-v0/csi-hostpath-attacher.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath-v0/csi-hostpath-provisioner.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath-v0/csi-hostpathplugin.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath-v0/e2e-test-rbac.yaml",
	)
}

// gce-pd
type gcePDCSIDriver struct {
	driverInfo testsuites.DriverInfo
}

var _ testsuites.TestDriver = &gcePDCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &gcePDCSIDriver{}

// InitGcePDCSIDriver returns gcePDCSIDriver that implements TestDriver interface
func InitGcePDCSIDriver() testsuites.TestDriver {
	return &gcePDCSIDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        GCEPDCSIProvisionerName,
			FeatureTag:  "[Serial]",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext2",
				"ext3",
				"ext4",
				"xfs",
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapExec:        true,
				testsuites.CapMultiPODs:   true,
			},
		},
	}
}

func (g *gcePDCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &g.driverInfo
}

func (g *gcePDCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("gce", "gke")
	if pattern.FsType == "xfs" {
		framework.SkipUnlessNodeOSDistroIs("ubuntu", "custom")
	}
	if pattern.FeatureTag == "[sig-windows]" {
		framework.Skipf("Skipping tests for windows since CSI does not support it yet")
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

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (g *gcePDCSIDriver) GetClaimSize() string {
	return "5Gi"
}

func (g *gcePDCSIDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	By("deploying csi gce-pd driver")
	cancelLogging := testsuites.StartPodLogs(f)
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
	createGCESecrets(f.ClientSet, f.Namespace.Name)

	manifests := []string{
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/csi-controller-rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/node_ds.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/controller_ss.yaml",
	}

	cleanup, err := f.CreateFromManifests(nil, manifests...)
	if err != nil {
		framework.Failf("deploying csi gce-pd driver: %v", err)
	}

	return &testsuites.PerTestConfig{
			Driver:    g,
			Prefix:    "gcepd",
			Framework: f,
		}, func() {
			By("uninstalling gce-pd driver")
			cleanup()
			cancelLogging()
		}
}

// gcePd-external
type gcePDExternalCSIDriver struct {
	driverInfo testsuites.DriverInfo
}

var _ testsuites.TestDriver = &gcePDExternalCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &gcePDExternalCSIDriver{}

// InitGcePDExternalCSIDriver returns gcePDExternalCSIDriver that implements TestDriver interface
func InitGcePDExternalCSIDriver() testsuites.TestDriver {
	return &gcePDExternalCSIDriver{
		driverInfo: testsuites.DriverInfo{
			Name: GCEPDCSIProvisionerName,
			// TODO(#70258): this is temporary until we can figure out how to make e2e tests a library
			FeatureTag:  "[Feature: gcePD-external]",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext2",
				"ext3",
				"ext4",
				"xfs",
			),
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapExec:        true,
				testsuites.CapMultiPODs:   true,
			},
		},
	}
}

func (g *gcePDExternalCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &g.driverInfo
}

func (g *gcePDExternalCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("gce", "gke")
	if pattern.FsType == "xfs" {
		framework.SkipUnlessNodeOSDistroIs("ubuntu", "custom")
	}
	if pattern.FeatureTag == "[sig-windows]" {
		framework.Skipf("Skipping tests for windows since CSI does not support it yet")
	}
}

func (g *gcePDExternalCSIDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	ns := config.Framework.Namespace.Name
	provisioner := g.driverInfo.Name
	suffix := fmt.Sprintf("%s-sc", g.driverInfo.Name)

	parameters := map[string]string{"type": "pd-standard"}
	if fsType != "" {
		parameters["csi.storage.k8s.io/fstype"] = fsType
	}

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (g *gcePDExternalCSIDriver) GetClaimSize() string {
	return "5Gi"
}

func (g *gcePDExternalCSIDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	framework.SkipIfMultizone(f.ClientSet)

	return &testsuites.PerTestConfig{
		Driver:    g,
		Prefix:    "gcepdext",
		Framework: f,
	}, func() {}
}
