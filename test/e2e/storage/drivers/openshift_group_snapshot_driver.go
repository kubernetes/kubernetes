package drivers

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// Special test driver for volume group snapshots.
//
// Upstream uses a script to install csi-driver-hostpath with group snapshots enabled in its CSI sidecars.
// We can't use that in OCP, so let's create a new test driver based on [Driver: csi-hospath],
// only with the group snapshots enabled.

// The rest of the file is a copy of Kubernete's HostPath test driver from test/e2e/storage/drivers/csi.go
// Differences:
// - the tests driver name is: [Driver: csi-hospath-groupsnapshot].
// - enabled group snapshots in the external-snapshotter sidecar.
// - still use "csi-hostpath" as PatchCSIOptions.OldDriverName, because it's a name of a directory than needs to be replaced in the driver yaml files.

type groupSnapshotHostpathCSIDriver struct {
	driverInfo       storageframework.DriverInfo
	manifests        []string
	volumeAttributes []map[string]string
}

func initGroupSnapshotHostpathCSIDriver(name string, capabilities map[storageframework.Capability]bool, volumeAttributes []map[string]string, manifests ...string) storageframework.TestDriver {
	return &groupSnapshotHostpathCSIDriver{
		driverInfo: storageframework.DriverInfo{
			Name:        name,
			MaxFileSize: storageframework.FileSizeMedium,
			SupportedFsType: sets.NewString(
				"", // Default fsType
			),
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
			Capabilities: capabilities,
			StressTestOptions: &storageframework.StressTestOptions{
				NumPods:     10,
				NumRestarts: 10,
			},
			VolumeSnapshotStressTestOptions: &storageframework.VolumeSnapshotStressTestOptions{
				NumPods:      10,
				NumSnapshots: 10,
			},
			PerformanceTestOptions: &storageframework.PerformanceTestOptions{
				ProvisioningOptions: &storageframework.PerformanceTestProvisioningOptions{
					VolumeSize: "1Mi",
					Count:      300,
					// Volume provisioning metrics are compared to a high baseline.
					// Failure to pass would suggest a performance regression.
					ExpectedMetrics: &storageframework.Metrics{
						AvgLatency: 2 * time.Minute,
						Throughput: 0.5,
					},
				},
			},
			TestTags: []interface{}{"[OCPFeatureGate:VolumeGroupSnapshot]"},
		},
		manifests:        manifests,
		volumeAttributes: volumeAttributes,
	}
}

var _ storageframework.TestDriver = &groupSnapshotHostpathCSIDriver{}
var _ storageframework.DynamicPVTestDriver = &groupSnapshotHostpathCSIDriver{}
var _ storageframework.SnapshottableTestDriver = &groupSnapshotHostpathCSIDriver{}
var _ storageframework.EphemeralTestDriver = &groupSnapshotHostpathCSIDriver{}

// InitgroupSnapshotHostpathCSIDriver returns groupSnapshotHostpathCSIDriver that implements TestDriver interface
func InitGroupSnapshotHostpathCSIDriver() storageframework.TestDriver {
	capabilities := map[storageframework.Capability]bool{
		storageframework.CapPersistence:                    true,
		storageframework.CapSnapshotDataSource:             true,
		storageframework.CapMultiPODs:                      true,
		storageframework.CapBlock:                          true,
		storageframework.CapPVCDataSource:                  true,
		storageframework.CapControllerExpansion:            true,
		storageframework.CapOfflineExpansion:               true,
		storageframework.CapOnlineExpansion:                true,
		storageframework.CapSingleNodeVolume:               true,
		storageframework.CapReadWriteOncePod:               true,
		storageframework.CapMultiplePVsSameID:              true,
		storageframework.CapFSResizeFromSourceNotSupported: true,
		storageframework.CapVolumeGroupSnapshot:            true,

		// This is needed for the
		// testsuites/volumelimits.go `should support volume limits`
		// test. --maxvolumespernode=10 gets
		// added when patching the deployment.
		storageframework.CapVolumeLimits: true,
	}
	// OCP specific code: a different driver name (csi-hostpath-groupsnapshot)
	return initGroupSnapshotHostpathCSIDriver("csi-hostpath-groupsnapshot",
		capabilities,
		// Volume attributes don't matter, but we have to provide at least one map.
		[]map[string]string{
			{"foo": "bar"},
		},
		"test/e2e/testing-manifests/storage-csi/external-attacher/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-provisioner/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-snapshotter/csi-snapshotter/rbac-csi-snapshotter.yaml",
		"test/e2e/testing-manifests/storage-csi/external-health-monitor/external-health-monitor-controller/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/external-resizer/rbac.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-driverinfo.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/csi-hostpath-plugin.yaml",
		"test/e2e/testing-manifests/storage-csi/hostpath/hostpath/e2e-test-rbac.yaml",
	)
}

func (h *groupSnapshotHostpathCSIDriver) GetDriverInfo() *storageframework.DriverInfo {
	return &h.driverInfo
}

func (h *groupSnapshotHostpathCSIDriver) SkipUnsupportedTest(pattern storageframework.TestPattern) {
	if pattern.VolType == storageframework.CSIInlineVolume && len(h.volumeAttributes) == 0 {
		e2eskipper.Skipf("%s has no volume attributes defined, doesn't support ephemeral inline volumes", h.driverInfo.Name)
	}
}

func (h *groupSnapshotHostpathCSIDriver) GetDynamicProvisionStorageClass(ctx context.Context, config *storageframework.PerTestConfig, fsType string) *storagev1.StorageClass {
	provisioner := config.GetUniqueDriverName()
	parameters := map[string]string{}
	ns := config.Framework.Namespace.Name

	return storageframework.GetStorageClass(provisioner, parameters, nil, ns)
}

func (h *groupSnapshotHostpathCSIDriver) GetVolume(config *storageframework.PerTestConfig, volumeNumber int) (map[string]string, bool, bool) {
	return h.volumeAttributes[volumeNumber%len(h.volumeAttributes)], false /* not shared */, false /* read-write */
}

func (h *groupSnapshotHostpathCSIDriver) GetCSIDriverName(config *storageframework.PerTestConfig) string {
	return config.GetUniqueDriverName()
}

func (h *groupSnapshotHostpathCSIDriver) GetSnapshotClass(ctx context.Context, config *storageframework.PerTestConfig, parameters map[string]string) *unstructured.Unstructured {
	snapshotter := config.GetUniqueDriverName()
	ns := config.Framework.Namespace.Name

	return utils.GenerateSnapshotClassSpec(snapshotter, parameters, ns)
}

func (h *groupSnapshotHostpathCSIDriver) GetVolumeAttributesClass(_ context.Context, config *storageframework.PerTestConfig) *storagev1beta1.VolumeAttributesClass {
	return storageframework.CopyVolumeAttributesClass(&storagev1beta1.VolumeAttributesClass{
		DriverName: config.GetUniqueDriverName(),
		Parameters: map[string]string{
			hostpathCSIDriverMutableParameterName: hostpathCSIDriverMutableParameterValue,
		},
	}, config.Framework.Namespace.Name, "e2e-vac-hostpath")
}
func (h *groupSnapshotHostpathCSIDriver) GetVolumeGroupSnapshotClass(ctx context.Context, config *storageframework.PerTestConfig, parameters map[string]string) *unstructured.Unstructured {
	snapshotter := config.GetUniqueDriverName()
	ns := config.Framework.Namespace.Name

	return utils.GenerateVolumeGroupSnapshotClassSpec(snapshotter, parameters, ns)
}

func (h *groupSnapshotHostpathCSIDriver) PrepareTest(ctx context.Context, f *framework.Framework) *storageframework.PerTestConfig {
	// Create secondary namespace which will be used for creating driver
	driverNamespace := utils.CreateDriverNamespace(ctx, f)
	driverns := driverNamespace.Name
	testns := f.Namespace.Name

	ginkgo.By(fmt.Sprintf("deploying %s driver", h.driverInfo.Name))
	cancelLogging := utils.StartPodLogs(ctx, f, driverNamespace)
	cs := f.ClientSet

	// The hostpath CSI driver only works when everything runs on the same node.
	node, err := e2enode.GetRandomReadySchedulableNode(ctx, cs)
	framework.ExpectNoError(err)
	config := &storageframework.PerTestConfig{
		Driver:              h,
		Prefix:              "hostpath",
		Framework:           f,
		ClientNodeSelection: e2epod.NodeSelection{Name: node.Name},
		DriverNamespace:     driverNamespace,
	}

	patches := []utils.PatchCSIOptions{}

	patches = append(patches, utils.PatchCSIOptions{
		OldDriverName:       "csi-hostpath", // OCP: hardcode csi-hostpath here, it specifies directories in yaml files that need to be replaced with the unique driver name.
		NewDriverName:       config.GetUniqueDriverName(),
		DriverContainerName: "hostpath",
		DriverContainerArguments: []string{"--drivername=" + config.GetUniqueDriverName(),
			// This is needed for the
			// testsuites/volumelimits.go `should support volume limits`
			// test.
			"--maxvolumespernode=10",
			// Enable volume lifecycle checks, to report failure if
			// the volume is not unpublished / unstaged correctly.
			"--check-volume-lifecycle=true",
		},
		ProvisionerContainerName: "csi-provisioner",
		SnapshotterContainerName: "csi-snapshotter",
		NodeName:                 node.Name,
	})

	// VAC E2E HostPath patch
	// Enables ModifyVolume support in the hostpath CSI driver, and adds an enabled parameter name
	patches = append(patches, utils.PatchCSIOptions{
		DriverContainerName:      "hostpath",
		DriverContainerArguments: []string{"--enable-controller-modify-volume=true", "--accepted-mutable-parameter-names=e2eVacTest"},
	})

	// VAC E2E FeatureGate patches
	// TODO: These can be removed after the VolumeAttributesClass feature is default enabled
	patches = append(patches, utils.PatchCSIOptions{
		DriverContainerName:      "csi-provisioner",
		DriverContainerArguments: []string{"--feature-gates=VolumeAttributesClass=true"},
	})
	patches = append(patches, utils.PatchCSIOptions{
		DriverContainerName:      "csi-resizer",
		DriverContainerArguments: []string{"--feature-gates=VolumeAttributesClass=true"},
	})

	// OCP specific code: enable group snapshot
	patches = append(patches, utils.PatchCSIOptions{
		DriverContainerName:      "csi-snapshotter",
		DriverContainerArguments: []string{"--feature-gates=CSIVolumeGroupSnapshot=true"},
	})

	err = utils.CreateFromManifests(ctx, config.Framework, driverNamespace, func(item interface{}) error {
		for _, o := range patches {
			if err := utils.PatchCSIDeployment(config.Framework, o, item); err != nil {
				return err
			}
		}

		// Remove csi-external-health-monitor-agent and
		// csi-external-health-monitor-controller
		// containers. The agent is obsolete.
		// The controller is not needed for any of the
		// tests and is causing too much overhead when
		// running in a large cluster (see
		// https://github.com/kubernetes/kubernetes/issues/102452#issuecomment-856991009).
		switch item := item.(type) {
		case *appsv1.StatefulSet:
			var containers []v1.Container
			for _, container := range item.Spec.Template.Spec.Containers {
				switch container.Name {
				case "csi-external-health-monitor-agent", "csi-external-health-monitor-controller":
					// Remove these containers.
				default:
					// Keep the others.
					containers = append(containers, container)
				}
			}
			item.Spec.Template.Spec.Containers = containers
		}
		return nil
	}, h.manifests...)

	if err != nil {
		framework.Failf("deploying %s driver: %v", h.driverInfo.Name, err)
	}

	cleanupFunc := generateDriverCleanupFunc(
		f,
		h.driverInfo.Name,
		testns,
		driverns,
		cancelLogging)
	ginkgo.DeferCleanup(cleanupFunc)

	return config
}
