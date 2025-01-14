/*
Copyright 2019 The Kubernetes Authors.

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
	"context"
	"fmt"
	"reflect"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

type multiVolumeTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

var _ storageframework.TestSuite = &multiVolumeTestSuite{}

// InitCustomMultiVolumeTestSuite returns multiVolumeTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomMultiVolumeTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &multiVolumeTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "multiVolume",
			TestTags:     []interface{}{framework.WithSlow()},
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

// InitMultiVolumeTestSuite returns multiVolumeTestSuite that implements TestSuite interface
// using test suite default patterns
func InitMultiVolumeTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.FsVolModePreprovisionedPV,
		storageframework.FsVolModeDynamicPV,
		storageframework.BlockVolModePreprovisionedPV,
		storageframework.BlockVolModeDynamicPV,
		storageframework.Ext4DynamicPV,
		storageframework.XfsDynamicPV,
		storageframework.NtfsDynamicPV,
	}
	return InitCustomMultiVolumeTestSuite(patterns)
}

func (t *multiVolumeTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *multiVolumeTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	dInfo := driver.GetDriverInfo()
	skipVolTypePatterns(pattern, driver, storageframework.NewVolTypeMap(storageframework.PreprovisionedPV))
	if pattern.VolMode == v1.PersistentVolumeBlock && !dInfo.Capabilities[storageframework.CapBlock] {
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolMode)
	}
}

func (t *multiVolumeTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config *storageframework.PerTestConfig

		cs        clientset.Interface
		ns        *v1.Namespace
		driver    storageframework.TestDriver
		resources []*storageframework.VolumeResource

		migrationCheck *migrationOpCheck
	}
	var (
		dInfo = driver.GetDriverInfo()
		l     local
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("multivolume", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) {
		l = local{}
		l.ns = f.Namespace
		l.cs = f.ClientSet
		l.driver = driver

		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)
		l.migrationCheck = newMigrationOpCheck(ctx, f.ClientSet, f.ClientConfig(), dInfo.InTreePluginName)
	}

	cleanup := func(ctx context.Context) {
		var errs []error
		for _, resource := range l.resources {
			errs = append(errs, resource.CleanupResource(ctx))
		}

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleanup resource")
		l.migrationCheck.validateMigrationVolumeOpCounts(ctx)
	}

	// This tests below configuration:
	//          [pod1]                            same node       [pod2]
	//      [   node1   ]                           ==>        [   node1   ]
	//          /    \      <- same volume mode                   /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	ginkgo.It("should access to two volumes with the same volume mode and retain data across pod recreation on the same node", func(ctx context.Context) {
		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == storageframework.PreprovisionedPV {
			e2eskipper.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		var pvcs []*v1.PersistentVolumeClaim
		numVols := 2

		for i := 0; i < numVols; i++ {
			testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
			resource := storageframework.CreateVolumeResource(ctx, driver, l.config, pattern, testVolumeSizeRange)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.Pvc)
		}
		TestAccessMultipleVolumesAcrossPodRecreation(ctx, l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, pvcs, true /* sameNode */)
	})

	// This tests below configuration:
	//          [pod1]                       different node       [pod2]
	//      [   node1   ]                           ==>        [   node2   ]
	//          /    \      <- same volume mode                   /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	ginkgo.It("should access to two volumes with the same volume mode and retain data across pod recreation on different node", func(ctx context.Context) {
		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == storageframework.PreprovisionedPV {
			e2eskipper.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Check different-node test requirement
		if l.driver.GetDriverInfo().Capabilities[storageframework.CapSingleNodeVolume] {
			e2eskipper.Skipf("Driver %s only supports %v -- skipping", l.driver.GetDriverInfo().Name, storageframework.CapSingleNodeVolume)
		}
		if l.config.ClientNodeSelection.Name != "" {
			e2eskipper.Skipf("Driver %q requires to deploy on a specific node - skipping", l.driver.GetDriverInfo().Name)
		}
		if err := ensureTopologyRequirements(ctx, &l.config.ClientNodeSelection, l.cs, dInfo, 2); err != nil {
			framework.Failf("Error setting topology requirements: %v", err)
		}

		var pvcs []*v1.PersistentVolumeClaim
		numVols := 2

		for i := 0; i < numVols; i++ {
			testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
			resource := storageframework.CreateVolumeResource(ctx, driver, l.config, pattern, testVolumeSizeRange)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.Pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(ctx, l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, pvcs, false /* sameNode */)
	})

	// This tests below configuration (only <block, filesystem> pattern is tested):
	//          [pod1]                            same node       [pod2]
	//      [   node1   ]                          ==>        [   node1   ]
	//          /    \      <- different volume mode             /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	ginkgo.It("should access to two volumes with different volume mode and retain data across pod recreation on the same node", func(ctx context.Context) {
		if pattern.VolMode == v1.PersistentVolumeFilesystem {
			e2eskipper.Skipf("Filesystem volume case should be covered by block volume case -- skipping")
		}

		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == storageframework.PreprovisionedPV {
			e2eskipper.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		var pvcs []*v1.PersistentVolumeClaim
		numVols := 2

		for i := 0; i < numVols; i++ {
			curPattern := pattern
			if i != 0 {
				// 1st volume should be block and set filesystem for 2nd and later volumes
				curPattern.VolMode = v1.PersistentVolumeFilesystem
			}
			testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
			resource := storageframework.CreateVolumeResource(ctx, driver, l.config, curPattern, testVolumeSizeRange)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.Pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(ctx, l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, pvcs, true /* sameNode */)
	})

	// This tests below configuration (only <block, filesystem> pattern is tested):
	//          [pod1]                      different node       [pod2]
	//      [   node1   ]                          ==>        [   node2   ]
	//          /    \      <- different volume mode             /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	ginkgo.It("should access to two volumes with different volume mode and retain data across pod recreation on different node", func(ctx context.Context) {
		if pattern.VolMode == v1.PersistentVolumeFilesystem {
			e2eskipper.Skipf("Filesystem volume case should be covered by block volume case -- skipping")
		}

		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == storageframework.PreprovisionedPV {
			e2eskipper.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		// Check different-node test requirement
		if l.driver.GetDriverInfo().Capabilities[storageframework.CapSingleNodeVolume] {
			e2eskipper.Skipf("Driver %s only supports %v -- skipping", l.driver.GetDriverInfo().Name, storageframework.CapSingleNodeVolume)
		}
		if l.config.ClientNodeSelection.Name != "" {
			e2eskipper.Skipf("Driver %q requires to deploy on a specific node - skipping", l.driver.GetDriverInfo().Name)
		}
		if err := ensureTopologyRequirements(ctx, &l.config.ClientNodeSelection, l.cs, dInfo, 2); err != nil {
			framework.Failf("Error setting topology requirements: %v", err)
		}

		var pvcs []*v1.PersistentVolumeClaim
		numVols := 2

		for i := 0; i < numVols; i++ {
			curPattern := pattern
			if i != 0 {
				// 1st volume should be block and set filesystem for 2nd and later volumes
				curPattern.VolMode = v1.PersistentVolumeFilesystem
			}
			testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
			resource := storageframework.CreateVolumeResource(ctx, driver, l.config, curPattern, testVolumeSizeRange)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.Pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(ctx, l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, pvcs, false /* sameNode */)
	})

	// This tests below configuration:
	// [pod1] [pod2]
	// [   node1   ]
	//   \      /     <- same volume mode
	//   [volume1]
	ginkgo.It("should concurrently access the single volume from pods on the same node", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		numPods := 2

		if !l.driver.GetDriverInfo().Capabilities[storageframework.CapMultiPODs] {
			e2eskipper.Skipf("Driver %q does not support multiple concurrent pods - skipping", dInfo.Name)
		}

		// Create volume
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		resource := storageframework.CreateVolumeResource(ctx, l.driver, l.config, pattern, testVolumeSizeRange)
		l.resources = append(l.resources, resource)

		// Test access to the volume from pods on different node
		TestConcurrentAccessToSingleVolume(ctx, l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, resource.Pvc, numPods, true /* sameNode */, false /* readOnly */)
	})

	// This tests below configuration:
	// [pod1]           [pod2]
	// [        node1        ]
	//   |                 |     <- same volume mode
	// [volume1]   ->  [restored volume1 snapshot]
	f.It("should concurrently access the volume and restored snapshot from pods on the same node [LinuxOnly]", feature.VolumeSnapshotDataSource, feature.VolumeSourceXFS, func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		if !l.driver.GetDriverInfo().Capabilities[storageframework.CapSnapshotDataSource] {
			e2eskipper.Skipf("Driver %q does not support volume snapshots - skipping", dInfo.Name)
		}
		if pattern.SnapshotType == "" {
			e2eskipper.Skipf("Driver %q does not support snapshots - skipping", dInfo.Name)
		}

		// Create a volume
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		resource := storageframework.CreateVolumeResource(ctx, l.driver, l.config, pattern, testVolumeSizeRange)
		l.resources = append(l.resources, resource)
		pvcs := []*v1.PersistentVolumeClaim{resource.Pvc}

		// Create snapshot of it
		expectedContent := fmt.Sprintf("volume content %d", time.Now().UTC().UnixNano())
		sDriver, ok := driver.(storageframework.SnapshottableTestDriver)
		if !ok {
			framework.Failf("Driver %q has CapSnapshotDataSource but does not implement SnapshottableTestDriver", dInfo.Name)
		}
		testConfig := storageframework.ConvertTestConfig(l.config)
		dc := l.config.Framework.DynamicClient
		dataSourceRef := prepareSnapshotDataSourceForProvisioning(ctx, f, testConfig, l.config, pattern, l.cs, dc, resource.Pvc, resource.Sc, sDriver, pattern.VolMode, expectedContent)

		// Create 2nd PVC for testing
		pvc2 := &v1.PersistentVolumeClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name:      resource.Pvc.Name + "-restored",
				Namespace: resource.Pvc.Namespace,
			},
		}
		resource.Pvc.Spec.DeepCopyInto(&pvc2.Spec)
		pvc2.Spec.VolumeName = ""
		pvc2.Spec.DataSourceRef = dataSourceRef

		pvc2, err := l.cs.CoreV1().PersistentVolumeClaims(pvc2.Namespace).Create(ctx, pvc2, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		pvcs = append(pvcs, pvc2)
		ginkgo.DeferCleanup(framework.IgnoreNotFound(l.cs.CoreV1().PersistentVolumeClaims(pvc2.Namespace).Delete), pvc2.Name, metav1.DeleteOptions{})

		// Test access to both volumes on the same node.
		TestConcurrentAccessToRelatedVolumes(ctx, l.config.Framework, l.cs, l.ns.Name, l.config.ClientNodeSelection, pvcs, expectedContent)
	})

	// This tests below configuration:
	// [pod1]           [pod2]
	// [        node1        ]
	//   |                 |     <- same volume mode
	// [volume1]   ->  [cloned volume1]
	f.It("should concurrently access the volume and its clone from pods on the same node [LinuxOnly]", feature.VolumeSourceXFS, func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		if !l.driver.GetDriverInfo().Capabilities[storageframework.CapPVCDataSource] {
			e2eskipper.Skipf("Driver %q does not support volume clone - skipping", dInfo.Name)
		}

		// Create a volume
		expectedContent := fmt.Sprintf("volume content %d", time.Now().UTC().UnixNano())
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		resource := storageframework.CreateVolumeResource(ctx, l.driver, l.config, pattern, testVolumeSizeRange)
		l.resources = append(l.resources, resource)
		pvcs := []*v1.PersistentVolumeClaim{resource.Pvc}
		testConfig := storageframework.ConvertTestConfig(l.config)
		dataSourceRef := preparePVCDataSourceForProvisioning(ctx, f, testConfig, l.cs, resource.Pvc, resource.Sc, pattern.VolMode, expectedContent)

		// Create 2nd PVC for testing
		pvc2 := &v1.PersistentVolumeClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name:      resource.Pvc.Name + "-cloned",
				Namespace: resource.Pvc.Namespace,
			},
		}
		resource.Pvc.Spec.DeepCopyInto(&pvc2.Spec)
		pvc2.Spec.VolumeName = ""
		pvc2.Spec.DataSourceRef = dataSourceRef

		pvc2, err := l.cs.CoreV1().PersistentVolumeClaims(pvc2.Namespace).Create(ctx, pvc2, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		pvcs = append(pvcs, pvc2)
		ginkgo.DeferCleanup(framework.IgnoreNotFound(l.cs.CoreV1().PersistentVolumeClaims(pvc2.Namespace).Delete), pvc2.Name, metav1.DeleteOptions{})

		// Test access to both volumes on the same node.
		TestConcurrentAccessToRelatedVolumes(ctx, l.config.Framework, l.cs, l.ns.Name, l.config.ClientNodeSelection, pvcs, expectedContent)
	})

	// This tests below configuration:
	// [pod1] [pod2]
	// [   node1   ]
	//   \      /     <- same volume mode (read only)
	//   [volume1]
	ginkgo.It("should concurrently access the single read-only volume from pods on the same node", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		numPods := 2

		if !l.driver.GetDriverInfo().Capabilities[storageframework.CapMultiPODs] {
			e2eskipper.Skipf("Driver %q does not support multiple concurrent pods - skipping", dInfo.Name)
		}

		if l.driver.GetDriverInfo().Name == "vsphere" && reflect.DeepEqual(pattern, storageframework.BlockVolModeDynamicPV) {
			e2eskipper.Skipf("Driver %q does not support read only raw block volumes - skipping", dInfo.Name)
		}

		// Create volume
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		resource := storageframework.CreateVolumeResource(ctx, l.driver, l.config, pattern, testVolumeSizeRange)
		l.resources = append(l.resources, resource)

		// Initialize the volume with a filesystem - it's going to be mounted as read-only below.
		initializeVolume(ctx, l.cs, f.Timeouts, l.ns.Name, resource.Pvc, l.config.ClientNodeSelection)

		// Test access to the volume from pods on a single node
		TestConcurrentAccessToSingleVolume(ctx, l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, resource.Pvc, numPods, true /* sameNode */, true /* readOnly */)
	})

	// This tests below configuration:
	//        [pod1] [pod2]
	// [   node1   ] [   node2   ]
	//         \      /     <- same volume mode
	//         [volume1]
	ginkgo.It("should concurrently access the single volume from pods on different node", func(ctx context.Context) {
		init(ctx)
		ginkgo.DeferCleanup(cleanup)

		numPods := 2

		if !l.driver.GetDriverInfo().Capabilities[storageframework.CapRWX] {
			e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", l.driver.GetDriverInfo().Name, storageframework.CapRWX)
		}

		// Check different-node test requirement
		if l.config.ClientNodeSelection.Name != "" {
			e2eskipper.Skipf("Driver %q requires to deploy on a specific node - skipping", l.driver.GetDriverInfo().Name)
		}
		// For multi-node tests there must be enough nodes with the same toopology to schedule the pods
		if err := ensureTopologyRequirements(ctx, &l.config.ClientNodeSelection, l.cs, dInfo, 2); err != nil {
			framework.Failf("Error setting topology requirements: %v", err)
		}

		// Create volume
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		resource := storageframework.CreateVolumeResource(ctx, l.driver, l.config, pattern, testVolumeSizeRange)
		l.resources = append(l.resources, resource)

		// Test access to the volume from pods on different node
		TestConcurrentAccessToSingleVolume(ctx, l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, resource.Pvc, numPods, false /* sameNode */, false /* readOnly */)
	})
}

// testAccessMultipleVolumes tests access to multiple volumes from single pod on the specified node
// If readSeedBase > 0, read test are done before write/read test assuming that there is already data written.
func testAccessMultipleVolumes(ctx context.Context, f *framework.Framework, cs clientset.Interface, ns string,
	node e2epod.NodeSelection, pvcs []*v1.PersistentVolumeClaim, readSeedBase int64, writeSeedBase int64) string {
	ginkgo.By(fmt.Sprintf("Creating pod on %+v with multiple volumes", node))
	podConfig := e2epod.Config{
		NS:            ns,
		PVCs:          pvcs,
		SeLinuxLabel:  e2epod.GetLinuxLabel(),
		NodeSelection: node,
		ImageID:       e2epod.GetDefaultTestImageID(),
	}
	pod, err := e2epod.CreateSecPodWithNodeSelection(ctx, cs, &podConfig, f.Timeouts.PodStart)
	defer func() {
		framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, cs, pod))
	}()
	framework.ExpectNoError(err)

	byteLen := 64
	for i, pvc := range pvcs {
		// CreateSecPodWithNodeSelection make volumes accessible via /mnt/volume({i} + 1)
		index := i + 1
		path := fmt.Sprintf("/mnt/volume%d", index)
		ginkgo.By(fmt.Sprintf("Checking if the volume%d exists as expected volume mode (%s)", index, *pvc.Spec.VolumeMode))
		e2evolume.CheckVolumeModeOfPath(ctx, f, pod, *pvc.Spec.VolumeMode, path)

		if readSeedBase > 0 {
			ginkgo.By(fmt.Sprintf("Checking if read from the volume%d works properly", index))
			storageutils.CheckReadFromPath(ctx, f, pod, *pvc.Spec.VolumeMode, false, path, byteLen, readSeedBase+int64(i))
		}

		ginkgo.By(fmt.Sprintf("Checking if write to the volume%d works properly", index))
		storageutils.CheckWriteToPath(ctx, f, pod, *pvc.Spec.VolumeMode, false, path, byteLen, writeSeedBase+int64(i))

		ginkgo.By(fmt.Sprintf("Checking if read from the volume%d works properly", index))
		storageutils.CheckReadFromPath(ctx, f, pod, *pvc.Spec.VolumeMode, false, path, byteLen, writeSeedBase+int64(i))
	}

	pod, err = cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	return pod.Spec.NodeName
}

// TestAccessMultipleVolumesAcrossPodRecreation tests access to multiple volumes from single pod,
// then recreate pod on the same or different node depending on requiresSameNode,
// and recheck access to the volumes from the recreated pod
func TestAccessMultipleVolumesAcrossPodRecreation(ctx context.Context, f *framework.Framework, cs clientset.Interface, ns string,
	node e2epod.NodeSelection, pvcs []*v1.PersistentVolumeClaim, requiresSameNode bool) {

	// No data is written in volume, so passing negative value
	readSeedBase := int64(-1)
	writeSeedBase := time.Now().UTC().UnixNano()
	// Test access to multiple volumes on the specified node
	nodeName := testAccessMultipleVolumes(ctx, f, cs, ns, node, pvcs, readSeedBase, writeSeedBase)

	// Set affinity depending on requiresSameNode
	if requiresSameNode {
		e2epod.SetAffinity(&node, nodeName)
	} else {
		e2epod.SetAntiAffinity(&node, nodeName)
	}

	// Test access to multiple volumes again on the node updated above
	// Setting previous writeSeed to current readSeed to check previous data is retained
	readSeedBase = writeSeedBase
	// Update writeSeed with new value
	writeSeedBase = time.Now().UTC().UnixNano()
	_ = testAccessMultipleVolumes(ctx, f, cs, ns, node, pvcs, readSeedBase, writeSeedBase)
}

// TestConcurrentAccessToSingleVolume tests access to a single volume from multiple pods,
// then delete the last pod, and recheck access to the volume after pod deletion to check if other
// pod deletion doesn't affect. Pods are deployed on the same node or different nodes depending on requiresSameNode.
// Read/write check are done across pod, by check reading both what pod{n-1} and pod{n} wrote from pod{n}.
func TestConcurrentAccessToSingleVolume(ctx context.Context, f *framework.Framework, cs clientset.Interface, ns string,
	node e2epod.NodeSelection, pvc *v1.PersistentVolumeClaim, numPods int, requiresSameNode bool,
	readOnly bool) {

	var pods []*v1.Pod

	// Create each pod with pvc
	for i := 0; i < numPods; i++ {
		index := i + 1
		ginkgo.By(fmt.Sprintf("Creating pod%d with a volume on %+v", index, node))
		podConfig := e2epod.Config{
			NS:            ns,
			PVCs:          []*v1.PersistentVolumeClaim{pvc},
			SeLinuxLabel:  e2epod.GetLinuxLabel(),
			NodeSelection: node,
			PVCsReadOnly:  readOnly,
			ImageID:       e2epod.GetTestImageID(imageutils.JessieDnsutils),
		}
		pod, err := e2epod.CreateSecPodWithNodeSelection(ctx, cs, &podConfig, f.Timeouts.PodStart)
		framework.ExpectNoError(err)
		// The pod must get deleted before this function returns because the caller may try to
		// delete volumes as part of the tests. Keeping the pod running would block that.
		// If the test times out, then the namespace deletion will take care of it.
		defer func() {
			framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, cs, pod))
		}()
		pod, err = cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
		pods = append(pods, pod)
		framework.ExpectNoError(err, fmt.Sprintf("get pod%d", index))
		actualNodeName := pod.Spec.NodeName

		// Set affinity depending on requiresSameNode
		if requiresSameNode {
			e2epod.SetAffinity(&node, actualNodeName)
		} else {
			e2epod.SetAntiAffinity(&node, actualNodeName)
		}
	}

	path := "/mnt/volume1"

	var seed int64
	byteLen := 64
	directIO := false
	// direct IO is needed for Block-mode PVs
	if *pvc.Spec.VolumeMode == v1.PersistentVolumeBlock {
		if len(pods) < 1 {
			framework.Failf("Number of pods shouldn't be less than 1, but got %d", len(pods))
		}
		// byteLen should be the size of a sector to enable direct I/O
		byteLen = storageutils.GetSectorSize(ctx, f, pods[0], path)
		directIO = true
	}

	// Check if volume can be accessed from each pod
	for i, pod := range pods {
		index := i + 1
		ginkgo.By(fmt.Sprintf("Checking if the volume in pod%d exists as expected volume mode (%s)", index, *pvc.Spec.VolumeMode))
		e2evolume.CheckVolumeModeOfPath(ctx, f, pod, *pvc.Spec.VolumeMode, path)

		if readOnly {
			ginkgo.By("Skipping volume content checks, volume is read-only")
			continue
		}

		if i != 0 {
			ginkgo.By(fmt.Sprintf("From pod%d, checking if reading the data that pod%d write works properly", index, index-1))
			// For 1st pod, no one has written data yet, so pass the read check
			storageutils.CheckReadFromPath(ctx, f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)
		}

		// Update the seed and check if write/read works properly
		seed = time.Now().UTC().UnixNano()

		ginkgo.By(fmt.Sprintf("Checking if write to the volume in pod%d works properly", index))
		storageutils.CheckWriteToPath(ctx, f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)

		ginkgo.By(fmt.Sprintf("Checking if read from the volume in pod%d works properly", index))
		storageutils.CheckReadFromPath(ctx, f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)
	}

	if len(pods) < 2 {
		framework.Failf("Number of pods shouldn't be less than 2, but got %d", len(pods))
	}
	// Delete the last pod and remove from slice of pods
	lastPod := pods[len(pods)-1]
	framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, cs, lastPod))
	pods = pods[:len(pods)-1]

	// Recheck if pv can be accessed from each pod after the last pod deletion
	for i, pod := range pods {
		index := i + 1
		// index of pod and index of pvc match, because pods are created above way
		ginkgo.By(fmt.Sprintf("Rechecking if the volume in pod%d exists as expected volume mode (%s)", index, *pvc.Spec.VolumeMode))
		e2evolume.CheckVolumeModeOfPath(ctx, f, pod, *pvc.Spec.VolumeMode, "/mnt/volume1")

		if readOnly {
			ginkgo.By("Skipping volume content checks, volume is read-only")
			continue
		}

		if i == 0 {
			// This time there should be data that last pod wrote, for 1st pod
			ginkgo.By(fmt.Sprintf("From pod%d, rechecking if reading the data that last pod write works properly", index))
		} else {
			ginkgo.By(fmt.Sprintf("From pod%d, rechecking if reading the data that pod%d write works properly", index, index-1))
		}
		storageutils.CheckReadFromPath(ctx, f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)

		// Update the seed and check if write/read works properly
		seed = time.Now().UTC().UnixNano()

		ginkgo.By(fmt.Sprintf("Rechecking if write to the volume in pod%d works properly", index))
		storageutils.CheckWriteToPath(ctx, f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)

		ginkgo.By(fmt.Sprintf("Rechecking if read from the volume in pod%d works properly", index))
		storageutils.CheckReadFromPath(ctx, f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)
	}
}

// TestConcurrentAccessToRelatedVolumes tests access to multiple volumes from multiple pods.
// Each provided PVC is used by a single pod. The test ensures that volumes created from
// another volume (=clone) or volume snapshot can be used together with the original volume.
func TestConcurrentAccessToRelatedVolumes(ctx context.Context, f *framework.Framework, cs clientset.Interface, ns string,
	node e2epod.NodeSelection, pvcs []*v1.PersistentVolumeClaim, expectedContent string) {

	var pods []*v1.Pod

	// Create each pod with pvc
	for i := range pvcs {
		index := i + 1
		ginkgo.By(fmt.Sprintf("Creating pod%d with a volume on %+v", index, node))
		podConfig := e2epod.Config{
			NS:            ns,
			PVCs:          []*v1.PersistentVolumeClaim{pvcs[i]},
			SeLinuxLabel:  e2epod.GetLinuxLabel(),
			NodeSelection: node,
			PVCsReadOnly:  false,
			ImageID:       e2epod.GetTestImageID(imageutils.JessieDnsutils),
		}
		pod, err := e2epod.CreateSecPodWithNodeSelection(ctx, cs, &podConfig, f.Timeouts.PodStart)
		defer func() {
			framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, cs, pod))
		}()
		framework.ExpectNoError(err)
		pods = append(pods, pod)
		actualNodeName := pod.Spec.NodeName

		// Always run the subsequent pods on the same node.
		e2epod.SetAffinity(&node, actualNodeName)
	}

	for i, pvc := range pvcs {
		var commands []string

		if *pvc.Spec.VolumeMode == v1.PersistentVolumeBlock {
			fileName := "/mnt/volume1"
			commands = e2evolume.GenerateReadBlockCmd(fileName, len(expectedContent))
			// Check that all pods have the same content
			index := i + 1
			ginkgo.By(fmt.Sprintf("Checking if the volume in pod%d has expected initial content", index))
			_, err := e2eoutput.LookForStringInPodExec(pods[i].Namespace, pods[i].Name, commands, expectedContent, time.Minute)
			framework.ExpectNoError(err, "failed: finding the contents of the block volume %s.", fileName)
		} else {
			fileName := "/mnt/volume1/index.html"
			commands = e2evolume.GenerateReadFileCmd(fileName)
			// Check that all pods have the same content
			index := i + 1
			ginkgo.By(fmt.Sprintf("Checking if the volume in pod%d has expected initial content", index))
			_, err := e2eoutput.LookForStringInPodExec(pods[i].Namespace, pods[i].Name, commands, expectedContent, time.Minute)
			framework.ExpectNoError(err, "failed: finding the contents of the mounted file %s.", fileName)
		}
	}
}

// getCurrentTopologies() goes through all Nodes and returns unique driver topologies and count of Nodes per topology
func getCurrentTopologiesNumber(cs clientset.Interface, nodes *v1.NodeList, keys []string) ([]topology, []int, error) {
	topos := []topology{}
	topoCount := []int{}

	// TODO: scale?
	for _, n := range nodes.Items {
		topo := map[string]string{}
		for _, k := range keys {
			v, ok := n.Labels[k]
			if ok {
				topo[k] = v
			}
		}

		found := false
		for i, existingTopo := range topos {
			if topologyEqual(existingTopo, topo) {
				found = true
				topoCount[i]++
				break
			}
		}
		if !found && len(topo) > 0 {
			framework.Logf("found topology %v", topo)
			topos = append(topos, topo)
			topoCount = append(topoCount, 1)
		}
	}
	return topos, topoCount, nil
}

// ensureTopologyRequirements check that there are enough nodes in the cluster for a test and
// sets nodeSelection affinity according to given topology keys for drivers that provide them.
func ensureTopologyRequirements(ctx context.Context, nodeSelection *e2epod.NodeSelection, cs clientset.Interface, driverInfo *storageframework.DriverInfo, minCount int) error {
	nodes, err := e2enode.GetReadySchedulableNodes(ctx, cs)
	framework.ExpectNoError(err)
	if len(nodes.Items) < minCount {
		e2eskipper.Skipf("Number of available nodes is less than %d - skipping", minCount)
	}

	topologyKeys := driverInfo.TopologyKeys
	if len(topologyKeys) == 0 {
		// The driver does not have any topology restrictions
		return nil
	}

	topologyList, topologyCount, err := getCurrentTopologiesNumber(cs, nodes, topologyKeys)
	if err != nil {
		return err
	}
	suitableTopologies := []topology{}
	for i, topo := range topologyList {
		if topologyCount[i] >= minCount {
			suitableTopologies = append(suitableTopologies, topo)
		}
	}
	if len(suitableTopologies) == 0 {
		e2eskipper.Skipf("No topology with at least %d nodes found - skipping", minCount)
	}
	// Take the first suitable topology
	e2epod.SetNodeAffinityTopologyRequirement(nodeSelection, suitableTopologies[0])
	return nil
}

// initializeVolume creates a filesystem on given volume, so it can be used as read-only later
func initializeVolume(ctx context.Context, cs clientset.Interface, t *framework.TimeoutContext, ns string, pvc *v1.PersistentVolumeClaim, node e2epod.NodeSelection) {
	if pvc.Spec.VolumeMode != nil && *pvc.Spec.VolumeMode == v1.PersistentVolumeBlock {
		// Block volumes do not need to be initialized.
		return
	}

	ginkgo.By(fmt.Sprintf("Initializing a filesystem on PVC %s", pvc.Name))
	// Just create a pod with the volume as read-write. Kubernetes will create a filesystem there
	// if it does not exist yet.
	podConfig := e2epod.Config{
		NS:            ns,
		PVCs:          []*v1.PersistentVolumeClaim{pvc},
		SeLinuxLabel:  e2epod.GetLinuxLabel(),
		NodeSelection: node,
		ImageID:       e2epod.GetDefaultTestImageID(),
	}
	pod, err := e2epod.CreateSecPod(ctx, cs, &podConfig, t.PodStart)
	defer func() {
		framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, cs, pod))
	}()
	framework.ExpectNoError(err)
}
