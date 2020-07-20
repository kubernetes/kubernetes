// +build !providerless

package drivers

import (
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	vspheretest "k8s.io/kubernetes/test/e2e/storage/vsphere"
)

// Cinder
// This driver assumes that OpenStack client tools are installed
// (/usr/bin/nova, /usr/bin/cinder and /usr/bin/keystone)
// and that the usual OpenStack authentication env. variables are set
// (OS_USERNAME, OS_PASSWORD, OS_TENANT_NAME at least).
type cinderDriver struct {
	driverInfo testsuites.DriverInfo
}

type cinderVolume struct {
	volumeName string
	volumeID   string
}

var _ testsuites.TestDriver = &cinderDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &cinderDriver{}
var _ testsuites.InlineVolumeTestDriver = &cinderDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &cinderDriver{}
var _ testsuites.DynamicPVTestDriver = &cinderDriver{}

// InitCinderDriver returns cinderDriver that implements TestDriver interface
func InitCinderDriver() testsuites.TestDriver {
	return &cinderDriver{
		driverInfo: testsuites.DriverInfo{
			Name:             "cinder",
			InTreePluginName: "kubernetes.io/cinder",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext3",
			),
			TopologyKeys: []string{v1.LabelZoneFailureDomain},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapExec:        true,
				testsuites.CapBlock:       true,
				// Cinder supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				testsuites.CapVolumeLimits: false,
				testsuites.CapTopology:     true,
			},
		},
	}
}

func (c *cinderDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &c.driverInfo
}

func (c *cinderDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("openstack")
}

func (c *cinderDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	cv, ok := e2evolume.(*cinderVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Cinder test volume")

	volSource := v1.VolumeSource{
		Cinder: &v1.CinderVolumeSource{
			VolumeID: cv.volumeID,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.Cinder.FSType = fsType
	}
	return &volSource
}

func (c *cinderDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	cv, ok := e2evolume.(*cinderVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Cinder test volume")

	pvSource := v1.PersistentVolumeSource{
		Cinder: &v1.CinderPersistentVolumeSource{
			VolumeID: cv.volumeID,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.Cinder.FSType = fsType
	}
	return &pvSource, nil
}

func (c *cinderDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/cinder"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", c.driverInfo.Name)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (c *cinderDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    c,
		Prefix:    "cinder",
		Framework: f,
	}, func() {}
}

func (c *cinderDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	ns := f.Namespace

	// We assume that namespace.Name is a random string
	volumeName := ns.Name
	ginkgo.By("creating a test Cinder volume")
	output, err := exec.Command("cinder", "create", "--display-name="+volumeName, "1").CombinedOutput()
	outputString := string(output[:])
	framework.Logf("cinder output:\n%s", outputString)
	framework.ExpectNoError(err)

	// Parse 'id'' from stdout. Expected format:
	// |     attachments     |                  []                  |
	// |  availability_zone  |                 nova                 |
	// ...
	// |          id         | 1d6ff08f-5d1c-41a4-ad72-4ef872cae685 |
	volumeID := ""
	for _, line := range strings.Split(outputString, "\n") {
		fields := strings.Fields(line)
		if len(fields) != 5 {
			continue
		}
		if fields[1] != "id" {
			continue
		}
		volumeID = fields[3]
		break
	}
	framework.Logf("Volume ID: %s", volumeID)
	framework.ExpectNotEqual(volumeID, "")
	return &cinderVolume{
		volumeName: volumeName,
		volumeID:   volumeID,
	}
}

func (v *cinderVolume) DeleteVolume() {
	name := v.volumeName

	// Try to delete the volume for several seconds - it takes
	// a while for the plugin to detach it.
	var output []byte
	var err error
	timeout := time.Second * 120

	framework.Logf("Waiting up to %v for removal of cinder volume %s", timeout, name)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		output, err = exec.Command("cinder", "delete", name).CombinedOutput()
		if err == nil {
			framework.Logf("Cinder volume %s deleted", name)
			return
		}
		framework.Logf("Failed to delete volume %s: %v", name, err)
	}
	framework.Logf("Giving up deleting volume %s: %v\n%s", name, err, string(output[:]))
}

// GCE
type gcePdDriver struct {
	driverInfo testsuites.DriverInfo
}

type gcePdVolume struct {
	volumeName string
}

var _ testsuites.TestDriver = &gcePdDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &gcePdDriver{}
var _ testsuites.InlineVolumeTestDriver = &gcePdDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &gcePdDriver{}
var _ testsuites.DynamicPVTestDriver = &gcePdDriver{}

// InitGcePdDriver returns gcePdDriver that implements TestDriver interface
func InitGcePdDriver() testsuites.TestDriver {
	// In current test structure, it first initialize the driver and then set up
	// the new framework, so we cannot get the correct OS here. So here set to
	// support all fs types including both linux and windows. We have code to check Node OS later
	// during test.
	supportedTypes := sets.NewString(
		"", // Default fsType
		"ext2",
		"ext3",
		"ext4",
		"xfs",
		"ntfs",
	)
	return &gcePdDriver{
		driverInfo: testsuites.DriverInfo{
			Name:             "gcepd",
			InTreePluginName: "kubernetes.io/gce-pd",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
			SupportedFsType:      supportedTypes,
			SupportedMountOption: sets.NewString("debug", "nouid32"),
			TopologyKeys:         []string{v1.LabelZoneFailureDomain},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence:         true,
				testsuites.CapFsGroup:             true,
				testsuites.CapBlock:               true,
				testsuites.CapExec:                true,
				testsuites.CapMultiPODs:           true,
				testsuites.CapControllerExpansion: true,
				testsuites.CapNodeExpansion:       true,
				// GCE supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				testsuites.CapVolumeLimits: false,
				testsuites.CapTopology:     true,
			},
		},
	}
}

func (g *gcePdDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &g.driverInfo
}

func (g *gcePdDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("gce", "gke")
	if pattern.FeatureTag == "[sig-windows]" {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	}
}

func (g *gcePdDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	gv, ok := e2evolume.(*gcePdVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to GCE PD test volume")
	volSource := v1.VolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:   gv.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.GCEPersistentDisk.FSType = fsType
	}
	return &volSource
}

func (g *gcePdDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	gv, ok := e2evolume.(*gcePdVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to GCE PD test volume")
	pvSource := v1.PersistentVolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:   gv.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.GCEPersistentDisk.FSType = fsType
	}
	return &pvSource, nil
}

func (g *gcePdDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/gce-pd"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", g.driverInfo.Name)
	delayedBinding := storagev1.VolumeBindingWaitForFirstConsumer

	return testsuites.GetStorageClass(provisioner, parameters, &delayedBinding, ns, suffix)
}

func (g *gcePdDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	config := &testsuites.PerTestConfig{
		Driver:    g,
		Prefix:    "gcepd",
		Framework: f,
	}

	if framework.NodeOSDistroIs("windows") {
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				"kubernetes.io/os": "windows",
			},
		}
	}
	return config, func() {}

}

func (g *gcePdDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	zone := getInlineVolumeZone(config.Framework)
	if volType == testpatterns.InlineVolume {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				v1.LabelZoneFailureDomain: zone,
			},
		}
	}
	ginkgo.By("creating a test gce pd volume")
	vname, err := e2epv.CreatePDWithRetryAndZone(zone)
	framework.ExpectNoError(err)
	return &gcePdVolume{
		volumeName: vname,
	}
}

func (v *gcePdVolume) DeleteVolume() {
	e2epv.DeletePDWithRetry(v.volumeName)
}

// vSphere
type vSphereDriver struct {
	driverInfo testsuites.DriverInfo
}

type vSphereVolume struct {
	volumePath string
	nodeInfo   *vspheretest.NodeInfo
}

var _ testsuites.TestDriver = &vSphereDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &vSphereDriver{}
var _ testsuites.InlineVolumeTestDriver = &vSphereDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &vSphereDriver{}
var _ testsuites.DynamicPVTestDriver = &vSphereDriver{}

// InitVSphereDriver returns vSphereDriver that implements TestDriver interface
func InitVSphereDriver() testsuites.TestDriver {
	return &vSphereDriver{
		driverInfo: testsuites.DriverInfo{
			Name:             "vsphere",
			InTreePluginName: "kubernetes.io/vsphere-volume",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext4",
			),
			TopologyKeys: []string{v1.LabelZoneFailureDomain},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapExec:        true,
				testsuites.CapMultiPODs:   true,
				testsuites.CapTopology:    true,
			},
		},
	}
}
func (v *vSphereDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &v.driverInfo
}

func (v *vSphereDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("vsphere")
}

func (v *vSphereDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	vsv, ok := e2evolume.(*vSphereVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to vSphere test volume")

	// vSphere driver doesn't seem to support readOnly volume
	// TODO: check if it is correct
	if readOnly {
		return nil
	}
	volSource := v1.VolumeSource{
		VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
			VolumePath: vsv.volumePath,
		},
	}
	if fsType != "" {
		volSource.VsphereVolume.FSType = fsType
	}
	return &volSource
}

func (v *vSphereDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	vsv, ok := e2evolume.(*vSphereVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to vSphere test volume")

	// vSphere driver doesn't seem to support readOnly volume
	// TODO: check if it is correct
	if readOnly {
		return nil, nil
	}
	pvSource := v1.PersistentVolumeSource{
		VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
			VolumePath: vsv.volumePath,
		},
	}
	if fsType != "" {
		pvSource.VsphereVolume.FSType = fsType
	}
	return &pvSource, nil
}

func (v *vSphereDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/vsphere-volume"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", v.driverInfo.Name)

	return testsuites.GetStorageClass(provisioner, parameters, nil, ns, suffix)
}

func (v *vSphereDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    v,
		Prefix:    "vsphere",
		Framework: f,
	}, func() {}
}

func (v *vSphereDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	f := config.Framework
	vspheretest.Bootstrap(f)
	nodeInfo := vspheretest.GetReadySchedulableRandomNodeInfo()
	volumePath, err := nodeInfo.VSphere.CreateVolume(&vspheretest.VolumeOptions{}, nodeInfo.DataCenterRef)
	framework.ExpectNoError(err)
	return &vSphereVolume{
		volumePath: volumePath,
		nodeInfo:   nodeInfo,
	}
}

func (v *vSphereVolume) DeleteVolume() {
	v.nodeInfo.VSphere.DeleteVolume(v.volumePath, v.nodeInfo.DataCenterRef)
}

// Azure Disk
type azureDiskDriver struct {
	driverInfo testsuites.DriverInfo
}

type azureDiskVolume struct {
	volumeName string
}

var _ testsuites.TestDriver = &azureDiskDriver{}
var _ testsuites.PreprovisionedVolumeTestDriver = &azureDiskDriver{}
var _ testsuites.InlineVolumeTestDriver = &azureDiskDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &azureDiskDriver{}
var _ testsuites.DynamicPVTestDriver = &azureDiskDriver{}

// InitAzureDiskDriver returns azureDiskDriver that implements TestDriver interface
func InitAzureDiskDriver() testsuites.TestDriver {
	return &azureDiskDriver{
		driverInfo: testsuites.DriverInfo{
			Name:             "azure-disk",
			InTreePluginName: "kubernetes.io/azure-disk",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext3",
				"ext4",
				"xfs",
			),
			TopologyKeys: []string{v1.LabelZoneFailureDomain},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence: true,
				testsuites.CapFsGroup:     true,
				testsuites.CapBlock:       true,
				testsuites.CapExec:        true,
				testsuites.CapMultiPODs:   true,
				// Azure supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				testsuites.CapVolumeLimits: false,
				testsuites.CapTopology:     true,
			},
		},
	}
}

func (a *azureDiskDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &a.driverInfo
}

func (a *azureDiskDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("azure")
}

func (a *azureDiskDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	av, ok := e2evolume.(*azureDiskVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Azure test volume")
	diskName := av.volumeName[(strings.LastIndex(av.volumeName, "/") + 1):]

	kind := v1.AzureManagedDisk
	volSource := v1.VolumeSource{
		AzureDisk: &v1.AzureDiskVolumeSource{
			DiskName:    diskName,
			DataDiskURI: av.volumeName,
			Kind:        &kind,
			ReadOnly:    &readOnly,
		},
	}
	if fsType != "" {
		volSource.AzureDisk.FSType = &fsType
	}
	return &volSource
}

func (a *azureDiskDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	av, ok := e2evolume.(*azureDiskVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to Azure test volume")

	diskName := av.volumeName[(strings.LastIndex(av.volumeName, "/") + 1):]

	kind := v1.AzureManagedDisk
	pvSource := v1.PersistentVolumeSource{
		AzureDisk: &v1.AzureDiskVolumeSource{
			DiskName:    diskName,
			DataDiskURI: av.volumeName,
			Kind:        &kind,
			ReadOnly:    &readOnly,
		},
	}
	if fsType != "" {
		pvSource.AzureDisk.FSType = &fsType
	}
	return &pvSource, nil
}

func (a *azureDiskDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/azure-disk"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", a.driverInfo.Name)
	delayedBinding := storagev1.VolumeBindingWaitForFirstConsumer

	return testsuites.GetStorageClass(provisioner, parameters, &delayedBinding, ns, suffix)
}

func (a *azureDiskDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	return &testsuites.PerTestConfig{
		Driver:    a,
		Prefix:    "azure",
		Framework: f,
	}, func() {}
}

func (a *azureDiskDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	ginkgo.By("creating a test azure disk volume")
	zone := getInlineVolumeZone(config.Framework)
	if volType == testpatterns.InlineVolume {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				v1.LabelZoneFailureDomain: zone,
			},
		}
	}
	volumeName, err := e2epv.CreatePDWithRetryAndZone(zone)
	framework.ExpectNoError(err)
	return &azureDiskVolume{
		volumeName: volumeName,
	}
}

func (v *azureDiskVolume) DeleteVolume() {
	e2epv.DeletePDWithRetry(v.volumeName)
}

// AWS
type awsDriver struct {
	driverInfo testsuites.DriverInfo
}

type awsVolume struct {
	volumeName string
}

var _ testsuites.TestDriver = &awsDriver{}

var _ testsuites.PreprovisionedVolumeTestDriver = &awsDriver{}
var _ testsuites.InlineVolumeTestDriver = &awsDriver{}
var _ testsuites.PreprovisionedPVTestDriver = &awsDriver{}
var _ testsuites.DynamicPVTestDriver = &awsDriver{}

// InitAwsDriver returns awsDriver that implements TestDriver interface
func InitAwsDriver() testsuites.TestDriver {
	return &awsDriver{
		driverInfo: testsuites.DriverInfo{
			Name:             "aws",
			InTreePluginName: "kubernetes.io/aws-ebs",
			MaxFileSize:      testpatterns.FileSizeMedium,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "5Gi",
			},
			SupportedFsType: sets.NewString(
				"", // Default fsType
				"ext2",
				"ext3",
				"ext4",
				"xfs",
				"ntfs",
			),
			SupportedMountOption: sets.NewString("debug", "nouid32"),
			TopologyKeys:         []string{v1.LabelZoneFailureDomain},
			Capabilities: map[testsuites.Capability]bool{
				testsuites.CapPersistence:         true,
				testsuites.CapFsGroup:             true,
				testsuites.CapBlock:               true,
				testsuites.CapExec:                true,
				testsuites.CapMultiPODs:           true,
				testsuites.CapControllerExpansion: true,
				testsuites.CapNodeExpansion:       true,
				// AWS supports volume limits, but the test creates large
				// number of volumes and times out test suites.
				testsuites.CapVolumeLimits: false,
				testsuites.CapTopology:     true,
			},
		},
	}
}

func (a *awsDriver) GetDriverInfo() *testsuites.DriverInfo {
	return &a.driverInfo
}

func (a *awsDriver) SkipUnsupportedTest(pattern testpatterns.TestPattern) {
	e2eskipper.SkipUnlessProviderIs("aws")
}

func (a *awsDriver) GetVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) *v1.VolumeSource {
	av, ok := e2evolume.(*awsVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to AWS test volume")
	volSource := v1.VolumeSource{
		AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
			VolumeID: av.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		volSource.AWSElasticBlockStore.FSType = fsType
	}
	return &volSource
}

func (a *awsDriver) GetPersistentVolumeSource(readOnly bool, fsType string, e2evolume testsuites.TestVolume) (*v1.PersistentVolumeSource, *v1.VolumeNodeAffinity) {
	av, ok := e2evolume.(*awsVolume)
	framework.ExpectEqual(ok, true, "Failed to cast test volume to AWS test volume")
	pvSource := v1.PersistentVolumeSource{
		AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
			VolumeID: av.volumeName,
			ReadOnly: readOnly,
		},
	}
	if fsType != "" {
		pvSource.AWSElasticBlockStore.FSType = fsType
	}
	return &pvSource, nil
}

func (a *awsDriver) GetDynamicProvisionStorageClass(config *testsuites.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := "kubernetes.io/aws-ebs"
	parameters := map[string]string{}
	if fsType != "" {
		parameters["fsType"] = fsType
	}
	ns := config.Framework.Namespace.Name
	suffix := fmt.Sprintf("%s-sc", a.driverInfo.Name)
	delayedBinding := storagev1.VolumeBindingWaitForFirstConsumer

	return testsuites.GetStorageClass(provisioner, parameters, &delayedBinding, ns, suffix)
}

func (a *awsDriver) PrepareTest(f *framework.Framework) (*testsuites.PerTestConfig, func()) {
	config := &testsuites.PerTestConfig{
		Driver:    a,
		Prefix:    "aws",
		Framework: f,
	}

	if framework.NodeOSDistroIs("windows") {
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				"kubernetes.io/os": "windows",
			},
		}
	}
	return config, func() {}
}

func (a *awsDriver) CreateVolume(config *testsuites.PerTestConfig, volType testpatterns.TestVolType) testsuites.TestVolume {
	zone := getInlineVolumeZone(config.Framework)
	if volType == testpatterns.InlineVolume {
		// PD will be created in framework.TestContext.CloudConfig.Zone zone,
		// so pods should be also scheduled there.
		config.ClientNodeSelection = e2epod.NodeSelection{
			Selector: map[string]string{
				v1.LabelZoneFailureDomain: zone,
			},
		}
	}
	ginkgo.By("creating a test aws volume")
	vname, err := e2epv.CreatePDWithRetryAndZone(zone)
	framework.ExpectNoError(err)
	return &awsVolume{
		volumeName: vname,
	}
}

func (v *awsVolume) DeleteVolume() {
	e2epv.DeletePDWithRetry(v.volumeName)
}
