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
	cleanup    func()
	driverInfo testsuites.DriverInfo
	manifests  []string
}

func initHostPathCSIDriver(name string, config testsuites.TestConfig, capabilities map[testsuites.Capability]bool, manifests ...string) testsuites.TestDriver {
	return &hostpathCSIDriver{
		driverInfo: testsuites.DriverInfo{
			Name:        name,
			FeatureTag:  "",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: capabilities,
			Config:       config,
		},
		manifests: manifests,
	}
}

var _ testsuites.TestDriver = &hostpathCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &hostpathCSIDriver{}
var _ testsuites.SnapshottableTestDriver = &hostpathCSIDriver{}

// InitHostPathCSIDriver returns hostpathCSIDriver that implements TestDriver interface
func InitHostPathCSIDriver(config testsuites.TestConfig) testsuites.TestDriver {
	return initHostPathCSIDriver("csi-hostpath", config,
		map[testsuites.Capability]bool{testsuites.CapPersistence: true, testsuites.CapDataSource: true, testsuites.CapMultiPODs: true},
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

func (h *hostpathCSIDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	provisioner := testsuites.GetUniqueDriverName(h)
	parameters := map[string]string{}
	ns := h.driverInfo.Config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", provisioner)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (h *hostpathCSIDriver) GetSnapshotClass() *unstructured.Unstructured {
	snapshotter := testsuites.GetUniqueDriverName(h)
	parameters := map[string]string{}
	ns := h.driverInfo.Config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-vsc", snapshotter)

	return testsuites.GetSnapshotClass(snapshotter, parameters, ns, suffix)
}

func (h *hostpathCSIDriver) GetClaimSize() string {
	return "5Gi"
}

func (h *hostpathCSIDriver) CreateDriver() {
	By(fmt.Sprintf("deploying %s driver", h.driverInfo.Name))
	f := h.driverInfo.Config.Framework
	cs := f.ClientSet

	// The hostpath CSI driver only works when everything runs on the same node.
	nodes := framework.GetReadySchedulableNodesOrDie(cs)
	nodeName := nodes.Items[rand.Intn(len(nodes.Items))].Name
	h.driverInfo.Config.ClientNodeName = nodeName

	// TODO (?): the storage.csi.image.version and storage.csi.image.registry
	// settings are ignored for this test. We could patch the image definitions.
	o := utils.PatchCSIOptions{
		OldDriverName:            h.driverInfo.Name,
		NewDriverName:            testsuites.GetUniqueDriverName(h),
		DriverContainerName:      "hostpath",
		DriverContainerArguments: []string{"--drivername=csi-hostpath-" + f.UniqueName},
		ProvisionerContainerName: "csi-provisioner",
		SnapshotterContainerName: "csi-snapshotter",
		NodeName:                 nodeName,
	}
	cleanup, err := h.driverInfo.Config.Framework.CreateFromManifests(func(item interface{}) error {
		return utils.PatchCSIDeployment(h.driverInfo.Config.Framework, o, item)
	},
		h.manifests...)
	h.cleanup = cleanup
	if err != nil {
		framework.Failf("deploying %s driver: %v", h.driverInfo.Name, err)
	}
}

func (h *hostpathCSIDriver) CleanupDriver() {
	if h.cleanup != nil {
		By(fmt.Sprintf("uninstalling %s driver", h.driverInfo.Name))
		h.cleanup()
	}
}

// mockCSI
type mockCSIDriver struct {
	cleanup    func()
	driverInfo testsuites.DriverInfo
}

var _ testsuites.TestDriver = &mockCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &mockCSIDriver{}

// InitMockCSIDriver returns a mockCSIDriver that implements TestDriver interface
func InitMockCSIDriver(config testsuites.TestConfig) testsuites.TestDriver {
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
			Config: config,
		},
	}
}

func (m *mockCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &m.driverInfo
}

func (m *mockCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (m *mockCSIDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	provisioner := testsuites.GetUniqueDriverName(m)
	parameters := map[string]string{}
	ns := m.driverInfo.Config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", provisioner)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (m *mockCSIDriver) GetClaimSize() string {
	return "5Gi"
}

func (m *mockCSIDriver) CreateDriver() {
	By("deploying csi mock driver")
	f := m.driverInfo.Config.Framework
	cs := f.ClientSet

	// pods should be scheduled on the node
	nodes := framework.GetReadySchedulableNodesOrDie(cs)
	node := nodes.Items[rand.Intn(len(nodes.Items))]
	m.driverInfo.Config.ClientNodeName = node.Name

	// TODO (?): the storage.csi.image.version and storage.csi.image.registry
	// settings are ignored for this test. We could patch the image definitions.
	o := utils.PatchCSIOptions{
		OldDriverName:            "csi-mock",
		NewDriverName:            "csi-mock-" + f.UniqueName,
		DriverContainerName:      "mock",
		DriverContainerArguments: []string{"--name=csi-mock-" + f.UniqueName},
		ProvisionerContainerName: "csi-provisioner",
		NodeName:                 m.driverInfo.Config.ClientNodeName,
	}
	cleanup, err := f.CreateFromManifests(func(item interface{}) error {
		return utils.PatchCSIDeployment(f, o, item)
	},
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/mock/csi-mock-rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/mock/csi-storageclass.yaml",
		"test/e2e/testing-manifests/storage-csi/mock/csi-mock-driver.yaml",
	)
	m.cleanup = cleanup
	if err != nil {
		framework.Failf("deploying csi mock driver: %v", err)
	}
}

func (m *mockCSIDriver) CleanupDriver() {
	if m.cleanup != nil {
		By("uninstalling csi mock driver")
		m.cleanup()
	}
}

// InitHostPathV0CSIDriver returns a variant of hostpathCSIDriver with different manifests.
func InitHostPathV0CSIDriver(config testsuites.TestConfig) testsuites.TestDriver {
	return initHostPathCSIDriver("csi-hostpath-v0", config,
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
	cleanup    func()
	driverInfo testsuites.DriverInfo
}

var _ testsuites.TestDriver = &gcePDCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &gcePDCSIDriver{}

// InitGcePDCSIDriver returns gcePDCSIDriver that implements TestDriver interface
func InitGcePDCSIDriver(config testsuites.TestConfig) testsuites.TestDriver {
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

			Config: config,
		},
	}
}

func (g *gcePDCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &g.driverInfo
}

func (g *gcePDCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	f := g.driverInfo.Config.Framework
	framework.SkipUnlessProviderIs("gce", "gke")
	if !g.driverInfo.Config.TopologyEnabled {
		// Topology is disabled in external-provisioner, so in a multizone cluster, a pod could be
		// scheduled in a different zone from the provisioned volume, causing basic provisioning
		// tests to fail.
		framework.SkipIfMultizone(f.ClientSet)
	}
	if pattern.FsType == "xfs" {
		framework.SkipUnlessNodeOSDistroIs("ubuntu", "custom")
	}
}

func (g *gcePDCSIDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	ns := g.driverInfo.Config.Framework.Namespace.Name
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

func (g *gcePDCSIDriver) CreateDriver() {
	By("deploying csi gce-pd driver")
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
	createGCESecrets(g.driverInfo.Config.Framework.ClientSet, g.driverInfo.Config.Framework.Namespace.Name)

	manifests := []string{
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/csi-controller-rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/node_ds.yaml",
	}

	if g.driverInfo.Config.TopologyEnabled {
		manifests = append(manifests, "test/e2e/testing-manifests/storage-csi/gce-pd/controller_ss_alpha.yaml")
	} else {
		manifests = append(manifests, "test/e2e/testing-manifests/storage-csi/gce-pd/controller_ss.yaml")
	}
	cleanup, err := g.driverInfo.Config.Framework.CreateFromManifests(nil, manifests...)
	g.cleanup = cleanup
	if err != nil {
		framework.Failf("deploying csi gce-pd driver: %v", err)
	}
}

func (g *gcePDCSIDriver) CleanupDriver() {
	By("uninstalling gce-pd driver")
	if g.cleanup != nil {
		g.cleanup()
	}
}

// gcePd-external
type gcePDExternalCSIDriver struct {
	driverInfo testsuites.DriverInfo
}

var _ testsuites.TestDriver = &gcePDExternalCSIDriver{}
var _ testsuites.DynamicPVTestDriver = &gcePDExternalCSIDriver{}

// InitGcePDExternalCSIDriver returns gcePDExternalCSIDriver that implements TestDriver interface
func InitGcePDExternalCSIDriver(config testsuites.TestConfig) testsuites.TestDriver {
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

			Config: config,
		},
	}
}

func (g *gcePDExternalCSIDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &g.driverInfo
}

func (g *gcePDExternalCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("gce", "gke")
	framework.SkipIfMultizone(g.driverInfo.Config.Framework.ClientSet)
	if pattern.FsType == "xfs" {
		framework.SkipUnlessNodeOSDistroIs("ubuntu", "custom")
	}
}

func (g *gcePDExternalCSIDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	ns := g.driverInfo.Config.Framework.Namespace.Name
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

func (g *gcePDExternalCSIDriver) CreateDriver() {
}

func (g *gcePDExternalCSIDriver) CleanupDriver() {
}
