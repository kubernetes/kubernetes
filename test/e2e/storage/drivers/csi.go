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
	"time"

	. "github.com/onsi/ginkgo"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// hostpathCSI
type hostpathCSIDriver struct {
	cleanup    func()
	driverInfo DriverInfo
}

var _ TestDriver = &hostpathCSIDriver{}
var _ DynamicPVTestDriver = &hostpathCSIDriver{}

// InitHostPathCSIDriver returns hostpathCSIDriver that implements TestDriver interface
func InitHostPathCSIDriver() TestDriver {
	return &hostpathCSIDriver{
		driverInfo: DriverInfo{
			Name:        "csi-hostpath",
			FeatureTag:  "",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[Capability]bool{
				CapPersistence: true,
			},
		},
	}
}

func (h *hostpathCSIDriver) GetDriverInfo() *DriverInfo {
	return &h.driverInfo
}

func (h *hostpathCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (h *hostpathCSIDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	provisioner := GetUniqueDriverName(h)
	parameters := map[string]string{}
	ns := h.driverInfo.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", provisioner)

	return getStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (h *hostpathCSIDriver) CreateDriver() {
	By("deploying csi hostpath driver")
	f := h.driverInfo.Framework
	cs := f.ClientSet

	// pods should be scheduled on the node
	nodes := framework.GetReadySchedulableNodesOrDie(cs)
	node := nodes.Items[rand.Intn(len(nodes.Items))]
	h.driverInfo.Config.ClientNodeName = node.Name
	h.driverInfo.Config.ServerNodeName = node.Name

	// TODO (?): the storage.csi.image.version and storage.csi.image.registry
	// settings are ignored for this test. We could patch the image definitions.
	o := utils.PatchCSIOptions{
		OldDriverName:            h.driverInfo.Name,
		NewDriverName:            GetUniqueDriverName(h),
		DriverContainerName:      "hostpath",
		ProvisionerContainerName: "csi-provisioner",
		NodeName:                 h.driverInfo.Config.ServerNodeName,
	}
	cleanup, err := h.driverInfo.Framework.CreateFromManifests(func(item interface{}) error {
		return utils.PatchCSIDeployment(h.driverInfo.Framework, o, item)
	},
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-attacher.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-provisioner.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpathplugin.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/e2e-test-rbac.yaml",
	)
	h.cleanup = cleanup
	if err != nil {
		framework.Failf("deploying csi hostpath driver: %v", err)
	}
}

func (h *hostpathCSIDriver) CleanupDriver() {
	if h.cleanup != nil {
		By("uninstalling csi hostpath driver")
		h.cleanup()
	}
}

// hostpathV0CSIDriver
type hostpathV0CSIDriver struct {
	cleanup    func()
	driverInfo DriverInfo
}

var _ TestDriver = &hostpathV0CSIDriver{}
var _ DynamicPVTestDriver = &hostpathV0CSIDriver{}

// InitHostPathV0CSIDriver returns hostpathV0CSIDriver that implements TestDriver interface
func InitHostV0PathCSIDriver() TestDriver {
	return &hostpathV0CSIDriver{
		driverInfo: DriverInfo{
			Name:        "csi-hostpath-v0",
			FeatureTag:  "",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			Capabilities: map[Capability]bool{
				CapPersistence: true,
			},
		},
	}
}

func (h *hostpathV0CSIDriver) GetDriverInfo() *DriverInfo {
	return &h.driverInfo
}

func (h *hostpathV0CSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
}

func (h *hostpathV0CSIDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	provisioner := GetUniqueDriverName(h)
	parameters := map[string]string{}
	ns := h.driverInfo.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", provisioner)

	return getStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (h *hostpathV0CSIDriver) CreateDriver() {
	By("deploying csi hostpath v0 driver")
	f := h.driverInfo.Framework
	cs := f.ClientSet

	// pods should be scheduled on the node
	nodes := framework.GetReadySchedulableNodesOrDie(cs)
	node := nodes.Items[rand.Intn(len(nodes.Items))]
	h.driverInfo.Config.ClientNodeName = node.Name
	h.driverInfo.Config.ServerNodeName = node.Name

	// TODO (?): the storage.csi.image.version and storage.csi.image.registry
	// settings are ignored for this test. We could patch the image definitions.
	o := utils.PatchCSIOptions{
		OldDriverName:            h.driverInfo.Name,
		NewDriverName:            GetUniqueDriverName(h),
		DriverContainerName:      "hostpath",
		ProvisionerContainerName: "csi-provisioner-v0",
		NodeName:                 h.driverInfo.Config.ServerNodeName,
	}
	cleanup, err := h.driverInfo.Framework.CreateFromManifests(func(item interface{}) error {
		return utils.PatchCSIDeployment(h.driverInfo.Framework, o, item)
	},
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath-v0/csi-hostpath-attacher.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath-v0/csi-hostpath-provisioner.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath-v0/csi-hostpathplugin.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath-v0/e2e-test-rbac.yaml",
	)
	h.cleanup = cleanup
	if err != nil {
		framework.Failf("deploying csi hostpath v0 driver: %v", err)
	}
}

func (h *hostpathV0CSIDriver) CleanupDriver() {
	if h.cleanup != nil {
		By("uninstalling csi hostpath v0 driver")
		h.cleanup()
	}
}

// gce-pd
type gcePDCSIDriver struct {
	cleanup    func()
	driverInfo DriverInfo
}

var _ TestDriver = &gcePDCSIDriver{}
var _ DynamicPVTestDriver = &gcePDCSIDriver{}

// InitGcePDCSIDriver returns gcePDCSIDriver that implements TestDriver interface
func InitGcePDCSIDriver() TestDriver {
	return &gcePDCSIDriver{
		driverInfo: DriverInfo{
			Name:        "pd.csi.storage.gke.io",
			FeatureTag:  "[Serial]",
			MaxFileSize: testpatterns.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext2",
				"ext3",
				"ext4",
				"xfs",
			),
			Capabilities: map[Capability]bool{
				CapPersistence: true,
				CapFsGroup:     true,
				CapExec:        true,
			},
		},
	}
}

func (g *gcePDCSIDriver) GetDriverInfo() *DriverInfo {
	return &g.driverInfo
}

func (g *gcePDCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	f := g.driverInfo.Framework
	cs := f.ClientSet
	config := g.driverInfo.Config
	framework.SkipUnlessProviderIs("gce", "gke")
	framework.SkipIfMultizone(cs)

	// TODO(#62561): Use credentials through external pod identity when that goes GA instead of downloading keys.
	createGCESecrets(cs, config)
	framework.SkipUnlessSecretExistsAfterWait(cs, "cloud-sa", config.Namespace, 3*time.Minute)
}

func (g *gcePDCSIDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	ns := g.driverInfo.Framework.Namespace.Name
	provisioner := g.driverInfo.Name
	suffix := fmt.Sprintf("%s-sc", g.driverInfo.Name)

	parameters := map[string]string{"type": "pd-standard"}

	return getStorageClass(provisioner, parameters, nil, ns, suffix)
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
	// 	NewDriverName:            GetUniqueDriverName(g),
	// 	DriverContainerName:      "gce-driver",
	// 	ProvisionerContainerName: "csi-external-provisioner",
	// }
	cleanup, err := g.driverInfo.Framework.CreateFromManifests(nil,
		"test/e2e/testing-manifests/storage-csi/driver-registrar/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/csi-controller-rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/node_ds.yaml",
		"test/e2e/testing-manifests/storage-csi/gce-pd/controller_ss.yaml",
	)
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
	driverInfo DriverInfo
}

var _ TestDriver = &gcePDExternalCSIDriver{}
var _ DynamicPVTestDriver = &gcePDExternalCSIDriver{}

// InitGcePDExternalCSIDriver returns gcePDExternalCSIDriver that implements TestDriver interface
func InitGcePDExternalCSIDriver() TestDriver {
	return &gcePDExternalCSIDriver{
		driverInfo: DriverInfo{
			Name: "pd.csi.storage.gke.io",
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
			Capabilities: map[Capability]bool{
				CapPersistence: true,
				CapFsGroup:     true,
				CapExec:        true,
			},
		},
	}
}

func (g *gcePDExternalCSIDriver) GetDriverInfo() *DriverInfo {
	return &g.driverInfo
}

func (g *gcePDExternalCSIDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	framework.SkipUnlessProviderIs("gce", "gke")
	framework.SkipIfMultizone(g.driverInfo.Framework.ClientSet)
}

func (g *gcePDExternalCSIDriver) GetDynamicProvisionStorageClass(fsType string) *storagev1.StorageClass {
	ns := g.driverInfo.Framework.Namespace.Name
	provisioner := g.driverInfo.Name
	suffix := fmt.Sprintf("%s-sc", g.driverInfo.Name)

	parameters := map[string]string{"type": "pd-standard"}

	return getStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (g *gcePDExternalCSIDriver) CreateDriver() {
}

func (g *gcePDExternalCSIDriver) CleanupDriver() {
}
