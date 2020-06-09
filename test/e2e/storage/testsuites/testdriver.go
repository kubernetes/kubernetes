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

package testsuites

import (
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

// TestDriver represents an interface for a driver to be tested in TestSuite.
// Except for GetDriverInfo, all methods will be called at test runtime and thus
// can use e2eskipper.Skipf, framework.Fatal, Gomega assertions, etc.
type TestDriver interface {
	// GetDriverInfo returns DriverInfo for the TestDriver. This must be static
	// information.
	GetDriverInfo() *DriverInfo

	// SkipUnsupportedTest skips test if Testpattern is not
	// suitable to test with the TestDriver. It gets called after
	// parsing parameters of the test suite and before the
	// framework is initialized. Cheap tests that just check
	// parameters like the cloud provider can and should be
	// done in SkipUnsupportedTest to avoid setting up more
	// expensive resources like framework.Framework. Tests that
	// depend on a connection to the cluster can be done in
	// PrepareTest once the framework is ready.
	SkipUnsupportedTest(testpatterns.TestPattern)

	// PrepareTest is called at test execution time each time a new test case is about to start.
	// It sets up all necessary resources and returns the per-test configuration
	// plus a cleanup function that frees all allocated resources.
	PrepareTest(f *framework.Framework) (*PerTestConfig, func())
}

// TestVolume is the result of PreprovisionedVolumeTestDriver.CreateVolume.
// The only common functionality is to delete it. Individual driver interfaces
// have additional methods that work with volumes created by them.
type TestVolume interface {
	DeleteVolume()
}

// PreprovisionedVolumeTestDriver represents an interface for a TestDriver that has pre-provisioned volume
type PreprovisionedVolumeTestDriver interface {
	TestDriver
	// CreateVolume creates a pre-provisioned volume of the desired volume type.
	CreateVolume(config *PerTestConfig, volumeType testpatterns.TestVolType) TestVolume
}

// InlineVolumeTestDriver represents an interface for a TestDriver that supports InlineVolume
type InlineVolumeTestDriver interface {
	PreprovisionedVolumeTestDriver

	// GetVolumeSource returns a volumeSource for inline volume.
	// It will set readOnly and fsType to the volumeSource, if TestDriver supports both of them.
	// It will return nil, if the TestDriver doesn't support either of the parameters.
	GetVolumeSource(readOnly bool, fsType string, testVolume TestVolume) *v1.VolumeSource
}

// PreprovisionedPVTestDriver represents an interface for a TestDriver that supports PreprovisionedPV
type PreprovisionedPVTestDriver interface {
	PreprovisionedVolumeTestDriver
	// GetPersistentVolumeSource returns a PersistentVolumeSource with volume node affinity for pre-provisioned Persistent Volume.
	// It will set readOnly and fsType to the PersistentVolumeSource, if TestDriver supports both of them.
	// It will return nil, if the TestDriver doesn't support either of the parameters.
	GetPersistentVolumeSource(readOnly bool, fsType string, testVolume TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity)
}

// DynamicPVTestDriver represents an interface for a TestDriver that supports DynamicPV
type DynamicPVTestDriver interface {
	TestDriver
	// GetDynamicProvisionStorageClass returns a StorageClass dynamic provision Persistent Volume.
	// The StorageClass must be created in the current test's namespace and have
	// a unique name inside that namespace because GetDynamicProvisionStorageClass might
	// be called more than once per test.
	// It will set fsType to the StorageClass, if TestDriver supports it.
	// It will return nil, if the TestDriver doesn't support it.
	GetDynamicProvisionStorageClass(config *PerTestConfig, fsType string) *storagev1.StorageClass
}

// EphemeralTestDriver represents an interface for a TestDriver that supports ephemeral inline volumes.
type EphemeralTestDriver interface {
	TestDriver

	// GetVolume returns the volume attributes for a certain
	// inline ephemeral volume, enumerated starting with #0. Some
	// tests might require more than one volume. They can all be
	// the same or different, depending what the driver supports
	// and/or wants to test.
	//
	// For each volume, the test driver can return volume attributes,
	// whether the resulting volume is shared between different pods (i.e.
	// changes made in one pod are visible in another), and whether the
	// volume can be mounted read/write or only read-only.
	GetVolume(config *PerTestConfig, volumeNumber int) (attributes map[string]string, shared bool, readOnly bool)

	// GetCSIDriverName returns the name that was used when registering with
	// kubelet. Depending on how the driver was deployed, this can be different
	// from DriverInfo.Name. Starting with Kubernetes 1.16, there must also
	// be a CSIDriver object under the same name with a "mode" field that enables
	// usage of the driver for ephemeral inline volumes.
	GetCSIDriverName(config *PerTestConfig) string
}

// SnapshottableTestDriver represents an interface for a TestDriver that supports DynamicSnapshot
type SnapshottableTestDriver interface {
	TestDriver
	// GetSnapshotClass returns a SnapshotClass to create snapshot.
	// It will return nil, if the TestDriver doesn't support it.
	GetSnapshotClass(config *PerTestConfig) *unstructured.Unstructured
}

// Capability represents a feature that a volume plugin supports
type Capability string

// Constants related to capability
const (
	CapPersistence        Capability = "persistence"        // data is persisted across pod restarts
	CapBlock              Capability = "block"              // raw block mode
	CapFsGroup            Capability = "fsGroup"            // volume ownership via fsGroup
	CapExec               Capability = "exec"               // exec a file in the volume
	CapSnapshotDataSource Capability = "snapshotDataSource" // support populate data from snapshot
	CapPVCDataSource      Capability = "pvcDataSource"      // support populate data from pvc

	// multiple pods on a node can use the same volume concurrently;
	// for CSI, see:
	// - https://github.com/container-storage-interface/spec/pull/150
	// - https://github.com/container-storage-interface/spec/issues/178
	// - NodeStageVolume in the spec
	CapMultiPODs Capability = "multipods"

	CapRWX                 Capability = "RWX"                 // support ReadWriteMany access modes
	CapControllerExpansion Capability = "controllerExpansion" // support volume expansion for controller
	CapNodeExpansion       Capability = "nodeExpansion"       // support volume expansion for node
	CapVolumeLimits        Capability = "volumeLimits"        // support volume limits (can be *very* slow)
	CapSingleNodeVolume    Capability = "singleNodeVolume"    // support volume that can run on single node (like hostpath)
	CapTopology            Capability = "topology"            // support topology
)

// DriverInfo represents static information about a TestDriver.
type DriverInfo struct {
	// Internal name of the driver, this is used as a display name in the test
	// case and test objects
	Name string
	// Fully qualified plugin name as registered in Kubernetes of the in-tree
	// plugin if it exists and is empty if this DriverInfo represents a CSI
	// Driver
	InTreePluginName string
	FeatureTag       string // FeatureTag for the driver

	// Maximum single file size supported by this driver
	MaxFileSize int64
	// The range of disk size supported by this driver
	SupportedSizeRange e2evolume.SizeRange
	// Map of string for supported fs type
	SupportedFsType sets.String
	// Map of string for supported mount option
	SupportedMountOption sets.String
	// [Optional] Map of string for required mount option
	RequiredMountOption sets.String
	// Map that represents plugin capabilities
	Capabilities map[Capability]bool
	// [Optional] List of access modes required for provisioning, defaults to
	// RWO if unset
	RequiredAccessModes []v1.PersistentVolumeAccessMode
	// [Optional] List of topology keys driver supports
	TopologyKeys []string
	// [Optional] Number of allowed topologies the driver requires.
	// Only relevant if TopologyKeys is set. Defaults to 1.
	// Example: multi-zonal disk requires at least 2 allowed topologies.
	NumAllowedTopologies int
	// [Optional] Scale parameters for stress tests.
	StressTestOptions *StressTestOptions
}

// StressTestOptions contains parameters used for stress tests.
type StressTestOptions struct {
	// Number of pods to create in the test. This may also create
	// up to 1 volume per pod.
	NumPods int
	// Number of times to restart each Pod.
	NumRestarts int
}

// PerTestConfig represents parameters that control test execution.
// One instance gets allocated for each test and is then passed
// via pointer to functions involved in the test.
type PerTestConfig struct {
	// The test driver for the test.
	Driver TestDriver

	// Some short word that gets inserted into dynamically
	// generated entities (pods, paths) as first part of the name
	// to make debugging easier. Can be the same for different
	// tests inside the test suite.
	Prefix string

	// The framework instance allocated for the current test.
	Framework *framework.Framework

	// If non-empty, Pods using a volume will be scheduled
	// according to the NodeSelection. Otherwise Kubernetes will
	// pick a node.
	ClientNodeSelection e2epod.NodeSelection

	// Some test drivers initialize a storage server. This is
	// the configuration that then has to be used to run tests.
	// The values above are ignored for such tests.
	ServerConfig *e2evolume.TestConfig

	// Some drivers run in their own namespace
	DriverNamespace *v1.Namespace
}

// GetUniqueDriverName returns unique driver name that can be used parallelly in tests
func (config *PerTestConfig) GetUniqueDriverName() string {
	return config.Driver.GetDriverInfo().Name + "-" + config.Framework.UniqueName
}
