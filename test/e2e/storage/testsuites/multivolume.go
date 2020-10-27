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
	"time"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

type multiVolumeTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &multiVolumeTestSuite{}

// InitMultiVolumeTestSuite returns multiVolumeTestSuite that implements TestSuite interface
func InitMultiVolumeTestSuite() TestSuite {
	return &multiVolumeTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "multiVolume [Slow]",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.FsVolModePreprovisionedPV,
				testpatterns.FsVolModeDynamicPV,
				testpatterns.BlockVolModePreprovisionedPV,
				testpatterns.BlockVolModeDynamicPV,
			},
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

func (t *multiVolumeTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *multiVolumeTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
	skipVolTypePatterns(pattern, driver, testpatterns.NewVolTypeMap(testpatterns.PreprovisionedPV))
}

func (t *multiVolumeTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config        *PerTestConfig
		driverCleanup func()

		cs        clientset.Interface
		ns        *v1.Namespace
		driver    TestDriver
		resources []*VolumeResource

		migrationCheck *migrationOpCheck
	}
	var (
		dInfo = driver.GetDriverInfo()
		l     local
	)

	ginkgo.BeforeEach(func() {
		// Check preconditions.
		if pattern.VolMode == v1.PersistentVolumeBlock && !dInfo.Capabilities[CapBlock] {
			e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolMode)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("multivolume")

	init := func() {
		l = local{}
		l.ns = f.Namespace
		l.cs = f.ClientSet
		l.driver = driver

		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)
		l.migrationCheck = newMigrationOpCheck(f.ClientSet, dInfo.InTreePluginName)
	}

	cleanup := func() {
		var errs []error
		for _, resource := range l.resources {
			errs = append(errs, resource.CleanupResource())
		}

		errs = append(errs, tryFunc(l.driverCleanup))
		l.driverCleanup = nil
		framework.ExpectNoError(errors.NewAggregate(errs), "while cleanup resource")
		l.migrationCheck.validateMigrationVolumeOpCounts()
	}

	// This tests below configuration:
	//          [pod1]                            same node       [pod2]
	//      [   node1   ]                           ==>        [   node1   ]
	//          /    \      <- same volume mode                   /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	ginkgo.It("should access to two volumes with the same volume mode and retain data across pod recreation on the same node", func() {
		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == testpatterns.PreprovisionedPV {
			e2eskipper.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init()
		defer cleanup()

		var pvcs []*v1.PersistentVolumeClaim
		numVols := 2

		for i := 0; i < numVols; i++ {
			testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
			resource := CreateVolumeResource(driver, l.config, pattern, testVolumeSizeRange)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.Pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, pvcs, true /* sameNode */)
	})

	// This tests below configuration:
	//          [pod1]                       different node       [pod2]
	//      [   node1   ]                           ==>        [   node2   ]
	//          /    \      <- same volume mode                   /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	ginkgo.It("should access to two volumes with the same volume mode and retain data across pod recreation on different node", func() {
		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == testpatterns.PreprovisionedPV {
			e2eskipper.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init()
		defer cleanup()

		// Check different-node test requirement
		if l.driver.GetDriverInfo().Capabilities[CapSingleNodeVolume] {
			e2eskipper.Skipf("Driver %s only supports %v -- skipping", l.driver.GetDriverInfo().Name, CapSingleNodeVolume)
		}
		nodes, err := e2enode.GetReadySchedulableNodes(l.cs)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf("Number of available nodes is less than 2 - skipping")
		}
		if l.config.ClientNodeSelection.Name != "" {
			e2eskipper.Skipf("Driver %q requires to deploy on a specific node - skipping", l.driver.GetDriverInfo().Name)
		}
		// For multi-node tests there must be enough nodes with the same toopology to schedule the pods
		topologyKeys := dInfo.TopologyKeys
		if len(topologyKeys) != 0 {
			if err = ensureTopologyRequirements(&l.config.ClientNodeSelection, nodes, l.cs, topologyKeys, 2); err != nil {
				framework.Failf("Error setting topology requirements: %v", err)
			}
		}

		var pvcs []*v1.PersistentVolumeClaim
		numVols := 2

		for i := 0; i < numVols; i++ {
			testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
			resource := CreateVolumeResource(driver, l.config, pattern, testVolumeSizeRange)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.Pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, pvcs, false /* sameNode */)
	})

	// This tests below configuration (only <block, filesystem> pattern is tested):
	//          [pod1]                            same node       [pod2]
	//      [   node1   ]                          ==>        [   node1   ]
	//          /    \      <- different volume mode             /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	ginkgo.It("should access to two volumes with different volume mode and retain data across pod recreation on the same node", func() {
		if pattern.VolMode == v1.PersistentVolumeFilesystem {
			e2eskipper.Skipf("Filesystem volume case should be covered by block volume case -- skipping")
		}

		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == testpatterns.PreprovisionedPV {
			e2eskipper.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init()
		defer cleanup()

		var pvcs []*v1.PersistentVolumeClaim
		numVols := 2

		for i := 0; i < numVols; i++ {
			curPattern := pattern
			if i != 0 {
				// 1st volume should be block and set filesystem for 2nd and later volumes
				curPattern.VolMode = v1.PersistentVolumeFilesystem
			}
			testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
			resource := CreateVolumeResource(driver, l.config, curPattern, testVolumeSizeRange)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.Pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, pvcs, true /* sameNode */)
	})

	// This tests below configuration (only <block, filesystem> pattern is tested):
	//          [pod1]                      different node       [pod2]
	//      [   node1   ]                          ==>        [   node2   ]
	//          /    \      <- different volume mode             /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	ginkgo.It("should access to two volumes with different volume mode and retain data across pod recreation on different node", func() {
		if pattern.VolMode == v1.PersistentVolumeFilesystem {
			e2eskipper.Skipf("Filesystem volume case should be covered by block volume case -- skipping")
		}

		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == testpatterns.PreprovisionedPV {
			e2eskipper.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init()
		defer cleanup()

		// Check different-node test requirement
		if l.driver.GetDriverInfo().Capabilities[CapSingleNodeVolume] {
			e2eskipper.Skipf("Driver %s only supports %v -- skipping", l.driver.GetDriverInfo().Name, CapSingleNodeVolume)
		}
		nodes, err := e2enode.GetReadySchedulableNodes(l.cs)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf("Number of available nodes is less than 2 - skipping")
		}
		if l.config.ClientNodeSelection.Name != "" {
			e2eskipper.Skipf("Driver %q requires to deploy on a specific node - skipping", l.driver.GetDriverInfo().Name)
		}
		// For multi-node tests there must be enough nodes with the same toopology to schedule the pods
		topologyKeys := dInfo.TopologyKeys
		if len(topologyKeys) != 0 {
			if err = ensureTopologyRequirements(&l.config.ClientNodeSelection, nodes, l.cs, topologyKeys, 2); err != nil {
				framework.Failf("Error setting topology requirements: %v", err)
			}
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
			resource := CreateVolumeResource(driver, l.config, curPattern, testVolumeSizeRange)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.Pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, pvcs, false /* sameNode */)
	})

	// This tests below configuration:
	// [pod1] [pod2]
	// [   node1   ]
	//   \      /     <- same volume mode
	//   [volume1]
	ginkgo.It("should concurrently access the single volume from pods on the same node", func() {
		init()
		defer cleanup()

		numPods := 2

		if !l.driver.GetDriverInfo().Capabilities[CapMultiPODs] {
			e2eskipper.Skipf("Driver %q does not support multiple concurrent pods - skipping", dInfo.Name)
		}

		// Create volume
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		resource := CreateVolumeResource(l.driver, l.config, pattern, testVolumeSizeRange)
		l.resources = append(l.resources, resource)

		// Test access to the volume from pods on different node
		TestConcurrentAccessToSingleVolume(l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, resource.Pvc, numPods, true /* sameNode */, false /* readOnly */)
	})

	// This tests below configuration:
	// [pod1] [pod2]
	// [   node1   ]
	//   \      /     <- same volume mode (read only)
	//   [volume1]
	ginkgo.It("should concurrently access the single read-only volume from pods on the same node", func() {
		init()
		defer cleanup()

		numPods := 2

		if !l.driver.GetDriverInfo().Capabilities[CapMultiPODs] {
			e2eskipper.Skipf("Driver %q does not support multiple concurrent pods - skipping", dInfo.Name)
		}

		// Create volume
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		resource := CreateVolumeResource(l.driver, l.config, pattern, testVolumeSizeRange)
		l.resources = append(l.resources, resource)

		// Initialize the volume with a filesystem - it's going to be mounted as read-only below.
		initializeVolume(l.cs, l.ns.Name, resource.Pvc, l.config.ClientNodeSelection)

		// Test access to the volume from pods on a single node
		TestConcurrentAccessToSingleVolume(l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, resource.Pvc, numPods, true /* sameNode */, true /* readOnly */)
	})

	// This tests below configuration:
	//        [pod1] [pod2]
	// [   node1   ] [   node2   ]
	//         \      /     <- same volume mode
	//         [volume1]
	ginkgo.It("should concurrently access the single volume from pods on different node", func() {
		init()
		defer cleanup()

		numPods := 2

		if !l.driver.GetDriverInfo().Capabilities[CapRWX] {
			e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", l.driver.GetDriverInfo().Name, CapRWX)
		}

		// Check different-node test requirement
		nodes, err := e2enode.GetReadySchedulableNodes(l.cs)
		framework.ExpectNoError(err)
		if len(nodes.Items) < numPods {
			e2eskipper.Skipf(fmt.Sprintf("Number of available nodes is less than %d - skipping", numPods))
		}
		if l.config.ClientNodeSelection.Name != "" {
			e2eskipper.Skipf("Driver %q requires to deploy on a specific node - skipping", l.driver.GetDriverInfo().Name)
		}
		// For multi-node tests there must be enough nodes with the same toopology to schedule the pods
		topologyKeys := dInfo.TopologyKeys
		if len(topologyKeys) != 0 {
			if err = ensureTopologyRequirements(&l.config.ClientNodeSelection, nodes, l.cs, topologyKeys, 2); err != nil {
				framework.Failf("Error setting topology requirements: %v", err)
			}
		}

		// Create volume
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		resource := CreateVolumeResource(l.driver, l.config, pattern, testVolumeSizeRange)
		l.resources = append(l.resources, resource)

		// Test access to the volume from pods on different node
		TestConcurrentAccessToSingleVolume(l.config.Framework, l.cs, l.ns.Name,
			l.config.ClientNodeSelection, resource.Pvc, numPods, false /* sameNode */, false /* readOnly */)
	})
}

// testAccessMultipleVolumes tests access to multiple volumes from single pod on the specified node
// If readSeedBase > 0, read test are done before write/read test assuming that there is already data written.
func testAccessMultipleVolumes(f *framework.Framework, cs clientset.Interface, ns string,
	node e2epod.NodeSelection, pvcs []*v1.PersistentVolumeClaim, readSeedBase int64, writeSeedBase int64) string {
	ginkgo.By(fmt.Sprintf("Creating pod on %+v with multiple volumes", node))
	podConfig := e2epod.Config{
		NS:            ns,
		PVCs:          pvcs,
		SeLinuxLabel:  e2epv.SELinuxLabel,
		NodeSelection: node,
	}
	pod, err := e2epod.CreateSecPodWithNodeSelection(cs, &podConfig, framework.PodStartTimeout)
	defer func() {
		framework.ExpectNoError(e2epod.DeletePodWithWait(cs, pod))
	}()
	framework.ExpectNoError(err)

	byteLen := 64
	for i, pvc := range pvcs {
		// CreateSecPodWithNodeSelection make volumes accessible via /mnt/volume({i} + 1)
		index := i + 1
		path := fmt.Sprintf("/mnt/volume%d", index)
		ginkgo.By(fmt.Sprintf("Checking if the volume%d exists as expected volume mode (%s)", index, *pvc.Spec.VolumeMode))
		utils.CheckVolumeModeOfPath(f, pod, *pvc.Spec.VolumeMode, path)

		if readSeedBase > 0 {
			ginkgo.By(fmt.Sprintf("Checking if read from the volume%d works properly", index))
			utils.CheckReadFromPath(f, pod, *pvc.Spec.VolumeMode, false, path, byteLen, readSeedBase+int64(i))
		}

		ginkgo.By(fmt.Sprintf("Checking if write to the volume%d works properly", index))
		utils.CheckWriteToPath(f, pod, *pvc.Spec.VolumeMode, false, path, byteLen, writeSeedBase+int64(i))

		ginkgo.By(fmt.Sprintf("Checking if read from the volume%d works properly", index))
		utils.CheckReadFromPath(f, pod, *pvc.Spec.VolumeMode, false, path, byteLen, writeSeedBase+int64(i))
	}

	pod, err = cs.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	return pod.Spec.NodeName
}

// TestAccessMultipleVolumesAcrossPodRecreation tests access to multiple volumes from single pod,
// then recreate pod on the same or different node depending on requiresSameNode,
// and recheck access to the volumes from the recreated pod
func TestAccessMultipleVolumesAcrossPodRecreation(f *framework.Framework, cs clientset.Interface, ns string,
	node e2epod.NodeSelection, pvcs []*v1.PersistentVolumeClaim, requiresSameNode bool) {

	// No data is written in volume, so passing negative value
	readSeedBase := int64(-1)
	writeSeedBase := time.Now().UTC().UnixNano()
	// Test access to multiple volumes on the specified node
	nodeName := testAccessMultipleVolumes(f, cs, ns, node, pvcs, readSeedBase, writeSeedBase)

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
	_ = testAccessMultipleVolumes(f, cs, ns, node, pvcs, readSeedBase, writeSeedBase)
}

// TestConcurrentAccessToSingleVolume tests access to a single volume from multiple pods,
// then delete the last pod, and recheck access to the volume after pod deletion to check if other
// pod deletion doesn't affect. Pods are deployed on the same node or different nodes depending on requiresSameNode.
// Read/write check are done across pod, by check reading both what pod{n-1} and pod{n} wrote from pod{n}.
func TestConcurrentAccessToSingleVolume(f *framework.Framework, cs clientset.Interface, ns string,
	node e2epod.NodeSelection, pvc *v1.PersistentVolumeClaim, numPods int, requiresSameNode bool,
	readOnly bool) {

	var pods []*v1.Pod

	// Create each pod with pvc
	for i := 0; i < numPods; i++ {
		index := i + 1
		ginkgo.By(fmt.Sprintf("Creating pod%d with a volume on %+v", index, node))
		podConfig := e2epod.Config{
			NS:            ns,
			ImageID:       imageutils.DebianIptables,
			PVCs:          []*v1.PersistentVolumeClaim{pvc},
			SeLinuxLabel:  e2epv.SELinuxLabel,
			NodeSelection: node,
			PVCsReadOnly:  readOnly,
		}
		pod, err := e2epod.CreateSecPodWithNodeSelection(cs, &podConfig, framework.PodStartTimeout)
		defer func() {
			framework.ExpectNoError(e2epod.DeletePodWithWait(cs, pod))
		}()
		framework.ExpectNoError(err)
		pod, err = cs.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
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

	var seed int64
	byteLen := 64
	directIO := false
	// direct IO is needed for Block-mode PVs
	if *pvc.Spec.VolumeMode == v1.PersistentVolumeBlock {
		// byteLen should be the size of a sector to enable direct I/O
		byteLen = 512
		directIO = true
	}

	path := "/mnt/volume1"
	// Check if volume can be accessed from each pod
	for i, pod := range pods {
		index := i + 1
		ginkgo.By(fmt.Sprintf("Checking if the volume in pod%d exists as expected volume mode (%s)", index, *pvc.Spec.VolumeMode))
		utils.CheckVolumeModeOfPath(f, pod, *pvc.Spec.VolumeMode, path)

		if readOnly {
			ginkgo.By("Skipping volume content checks, volume is read-only")
			continue
		}

		if i != 0 {
			ginkgo.By(fmt.Sprintf("From pod%d, checking if reading the data that pod%d write works properly", index, index-1))
			// For 1st pod, no one has written data yet, so pass the read check
			utils.CheckReadFromPath(f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)
		}

		// Update the seed and check if write/read works properly
		seed = time.Now().UTC().UnixNano()

		ginkgo.By(fmt.Sprintf("Checking if write to the volume in pod%d works properly", index))
		utils.CheckWriteToPath(f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)

		ginkgo.By(fmt.Sprintf("Checking if read from the volume in pod%d works properly", index))
		utils.CheckReadFromPath(f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)
	}

	// Delete the last pod and remove from slice of pods
	if len(pods) < 2 {
		framework.Failf("Number of pods shouldn't be less than 2, but got %d", len(pods))
	}
	lastPod := pods[len(pods)-1]
	framework.ExpectNoError(e2epod.DeletePodWithWait(cs, lastPod))
	pods = pods[:len(pods)-1]

	// Recheck if pv can be accessed from each pod after the last pod deletion
	for i, pod := range pods {
		index := i + 1
		// index of pod and index of pvc match, because pods are created above way
		ginkgo.By(fmt.Sprintf("Rechecking if the volume in pod%d exists as expected volume mode (%s)", index, *pvc.Spec.VolumeMode))
		utils.CheckVolumeModeOfPath(f, pod, *pvc.Spec.VolumeMode, "/mnt/volume1")

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
		utils.CheckReadFromPath(f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)

		// Update the seed and check if write/read works properly
		seed = time.Now().UTC().UnixNano()

		ginkgo.By(fmt.Sprintf("Rechecking if write to the volume in pod%d works properly", index))
		utils.CheckWriteToPath(f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)

		ginkgo.By(fmt.Sprintf("Rechecking if read from the volume in pod%d works properly", index))
		utils.CheckReadFromPath(f, pod, *pvc.Spec.VolumeMode, directIO, path, byteLen, seed)
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
		if !found {
			framework.Logf("found topology %v", topo)
			topos = append(topos, topo)
			topoCount = append(topoCount, 1)
		}
	}
	return topos, topoCount, nil
}

// ensureTopologyRequirements sets nodeSelection affinity according to given topology keys for drivers that provide them
func ensureTopologyRequirements(nodeSelection *e2epod.NodeSelection, nodes *v1.NodeList, cs clientset.Interface, topologyKeys []string, minCount int) error {
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
func initializeVolume(cs clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim, node e2epod.NodeSelection) {
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
		SeLinuxLabel:  e2epv.SELinuxLabel,
		NodeSelection: node,
	}
	pod, err := e2epod.CreateSecPod(cs, &podConfig, framework.PodStartTimeout)
	defer func() {
		framework.ExpectNoError(e2epod.DeletePodWithWait(cs, pod))
	}()
	framework.ExpectNoError(err)
}
