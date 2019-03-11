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
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

type multiVolumeTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &multiVolumeTestSuite{}

// InitMultiVolumeTestSuite returns multiVolumeTestSuite that implements TestSuite interface
func InitMultiVolumeTestSuite() TestSuite {
	return &multiVolumeTestSuite{
		tsInfo: TestSuiteInfo{
			name: "multiVolume",
			testPatterns: []testpatterns.TestPattern{
				testpatterns.FsVolModePreprovisionedPV,
				testpatterns.FsVolModeDynamicPV,
				testpatterns.BlockVolModePreprovisionedPV,
				testpatterns.BlockVolModeDynamicPV,
			},
		},
	}
}

func (t *multiVolumeTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *multiVolumeTestSuite) defineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config      *PerTestConfig
		testCleanup func()

		cs        clientset.Interface
		ns        *v1.Namespace
		driver    TestDriver
		resources []*genericVolumeTestResource
	}
	var (
		dInfo = driver.GetDriverInfo()
		l     local
	)

	BeforeEach(func() {
		// Check preconditions.
		if pattern.VolMode == v1.PersistentVolumeBlock && !dInfo.Capabilities[CapBlock] {
			framework.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolMode)
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
		l.config, l.testCleanup = driver.PrepareTest(f)
	}

	cleanup := func() {
		for _, resource := range l.resources {
			resource.cleanupResource()
		}

		if l.testCleanup != nil {
			l.testCleanup()
			l.testCleanup = nil
		}
	}

	// This tests below configuration:
	//          [pod1]                            same node       [pod2]
	//      [   node1   ]                           ==>        [   node1   ]
	//          /    \      <- same volume mode                   /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	It("should access to two volumes with the same volume mode and retain data across pod recreation on the same node", func() {
		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == testpatterns.PreprovisionedPV {
			framework.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init()
		defer cleanup()

		var pvcs []*v1.PersistentVolumeClaim
		numVols := 2

		for i := 0; i < numVols; i++ {
			resource := createGenericVolumeTestResource(driver, l.config, pattern)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(l.config.Framework, l.cs, l.ns.Name,
			framework.NodeSelection{Name: l.config.ClientNodeName}, pvcs, true /* sameNode */)
	})

	// This tests below configuration:
	//          [pod1]                       different node       [pod2]
	//      [   node1   ]                           ==>        [   node2   ]
	//          /    \      <- same volume mode                   /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	It("should access to two volumes with the same volume mode and retain data across pod recreation on different node", func() {
		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == testpatterns.PreprovisionedPV {
			framework.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init()
		defer cleanup()

		// Check different-node test requirement
		nodes := framework.GetReadySchedulableNodesOrDie(l.cs)
		if len(nodes.Items) < 2 {
			framework.Skipf("Number of available nodes is less than 2 - skipping")
		}
		if l.config.ClientNodeName != "" {
			framework.Skipf("Driver %q requires to deploy on a specific node - skipping", l.driver.GetDriverInfo().Name)
		}

		var pvcs []*v1.PersistentVolumeClaim
		numVols := 2

		for i := 0; i < numVols; i++ {
			resource := createGenericVolumeTestResource(driver, l.config, pattern)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(l.config.Framework, l.cs, l.ns.Name,
			framework.NodeSelection{Name: l.config.ClientNodeName}, pvcs, false /* sameNode */)
	})

	// This tests below configuration (only <block, filesystem> pattern is tested):
	//          [pod1]                            same node       [pod2]
	//      [   node1   ]                          ==>        [   node1   ]
	//          /    \      <- different volume mode             /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	It("should access to two volumes with different volume mode and retain data across pod recreation on the same node", func() {
		if pattern.VolMode == v1.PersistentVolumeFilesystem {
			framework.Skipf("Filesystem volume case should be covered by block volume case -- skipping")
		}

		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == testpatterns.PreprovisionedPV {
			framework.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
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
			resource := createGenericVolumeTestResource(driver, l.config, curPattern)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(l.config.Framework, l.cs, l.ns.Name,
			framework.NodeSelection{Name: l.config.ClientNodeName}, pvcs, true /* sameNode */)
	})

	// This tests below configuration (only <block, filesystem> pattern is tested):
	//          [pod1]                      different node       [pod2]
	//      [   node1   ]                          ==>        [   node2   ]
	//          /    \      <- different volume mode             /    \
	//   [volume1]  [volume2]                              [volume1]  [volume2]
	It("should access to two volumes with different volume mode and retain data across pod recreation on different node", func() {
		if pattern.VolMode == v1.PersistentVolumeFilesystem {
			framework.Skipf("Filesystem volume case should be covered by block volume case -- skipping")
		}

		// Currently, multiple volumes are not generally available for pre-provisoined volume,
		// because containerized storage servers, such as iSCSI and rbd, are just returning
		// a static volume inside container, not actually creating a new volume per request.
		if pattern.VolType == testpatterns.PreprovisionedPV {
			framework.Skipf("This test doesn't work with pre-provisioned volume -- skipping")
		}

		init()
		defer cleanup()

		// Check different-node test requirement
		nodes := framework.GetReadySchedulableNodesOrDie(l.cs)
		if len(nodes.Items) < 2 {
			framework.Skipf("Number of available nodes is less than 2 - skipping")
		}
		if l.config.ClientNodeName != "" {
			framework.Skipf("Driver %q requires to deploy on a specific node - skipping", l.driver.GetDriverInfo().Name)
		}

		var pvcs []*v1.PersistentVolumeClaim
		numVols := 2

		for i := 0; i < numVols; i++ {
			curPattern := pattern
			if i != 0 {
				// 1st volume should be block and set filesystem for 2nd and later volumes
				curPattern.VolMode = v1.PersistentVolumeFilesystem
			}
			resource := createGenericVolumeTestResource(driver, l.config, curPattern)
			l.resources = append(l.resources, resource)
			pvcs = append(pvcs, resource.pvc)
		}

		TestAccessMultipleVolumesAcrossPodRecreation(l.config.Framework, l.cs, l.ns.Name,
			framework.NodeSelection{Name: l.config.ClientNodeName}, pvcs, false /* sameNode */)
	})

	// This tests below configuration:
	// [pod1] [pod2]
	// [   node1   ]
	//   \      /     <- same volume mode
	//   [volume1]
	It("should concurrently access the single volume from pods on the same node", func() {
		init()
		defer cleanup()

		numPods := 2

		if !l.driver.GetDriverInfo().Capabilities[CapMultiPODs] {
			framework.Skipf("Driver %q does not support multiple concurrent pods - skipping", dInfo.Name)
		}

		// Create volume
		resource := createGenericVolumeTestResource(l.driver, l.config, pattern)
		l.resources = append(l.resources, resource)

		// Test access to the volume from pods on different node
		TestConcurrentAccessToSingleVolume(l.config.Framework, l.cs, l.ns.Name,
			framework.NodeSelection{Name: l.config.ClientNodeName}, resource.pvc, numPods, true /* sameNode */)
	})

	// This tests below configuration:
	//        [pod1] [pod2]
	// [   node1   ] [   node2   ]
	//         \      /     <- same volume mode
	//         [volume1]
	It("should concurrently access the single volume from pods on different node", func() {
		init()
		defer cleanup()

		numPods := 2

		if !l.driver.GetDriverInfo().Capabilities[CapRWX] {
			framework.Skipf("Driver %s doesn't support %v -- skipping", l.driver.GetDriverInfo().Name, CapRWX)
		}

		// Check different-node test requirement
		nodes := framework.GetReadySchedulableNodesOrDie(l.cs)
		if len(nodes.Items) < numPods {
			framework.Skipf(fmt.Sprintf("Number of available nodes is less than %d - skipping", numPods))
		}
		if l.config.ClientNodeName != "" {
			framework.Skipf("Driver %q requires to deploy on a specific node - skipping", l.driver.GetDriverInfo().Name)
		}

		// Create volume
		resource := createGenericVolumeTestResource(l.driver, l.config, pattern)
		l.resources = append(l.resources, resource)

		// Test access to the volume from pods on different node
		TestConcurrentAccessToSingleVolume(l.config.Framework, l.cs, l.ns.Name,
			framework.NodeSelection{Name: l.config.ClientNodeName}, resource.pvc, numPods, false /* sameNode */)
	})

	// The following two iSCSI tests test that the in-tree iSCSI plugin does not
	// prematurely logout from a target (portal+IQN). This issue is unique to
	// iSCSI because logging in/out is a node-wide operation, so any time kubelet
	// detaches a volume, the plugin must ensure that it doesn't log out while
	// other volumes depending on the same target are in use, otherwise they will
	// return i/o errors (same target but different lun are unique from kubelet's
	// perspective).
	// See #74313 for more information: previously, the plugin always logged out
	// when a block volume was detached, and the plugin logged out when a file
	// volume was detached even if a block volume was still in use.

	// This tests below configuration:
	//      [pod1] [pod2]                         same node    [pod1]
	//      [   node1   ]                           ==>        [   node1   ]
	//          /    \      <- same volume mode                    /
	//   [volume1]  [volume2]                               [volume1]
	It("should access an iSCSI volume after a different pod on the same node using a volume with the same volume mode, same target, & different lun is deleted", func() {
		// in-tree driver drivers.iSCSIDriver has name "iscsi"
		if dInfo.Name != "iscsi" {
			framework.Skipf("This test doesn't work with non-iscsi volume -- skipping")
		}
		if pattern.VolType != testpatterns.PreprovisionedPV {
			framework.Skipf("This test only works with pre-provisioned volume -- skipping")
		}

		init()
		defer cleanup()

		testIscsiDifferentLunSameNode(driver, pattern, l.config, l.resources, l.cs, l.ns, true)
	})

	// This tests below configuration:
	//      [pod1] [pod2]                         same node    [pod1]
	//      [   node1   ]                           ==>        [   node1   ]
	//          /    \      <- different volume mode               /
	//   [volume1]  [volume2]                               [volume1]
	It("should access an iSCSI volume after a different pod on the same node using a volume with a different volume mode, same target, & different lun is deleted", func() {
		// in-tree driver drivers.iSCSIDriver has name "iscsi"
		if dInfo.Name != "iscsi" {
			framework.Skipf("This test doesn't work with non-iscsi volume -- skipping")
		}
		if pattern.VolType != testpatterns.PreprovisionedPV {
			framework.Skipf("This test only works with pre-provisioned volume -- skipping")
		}

		init()
		defer cleanup()

		testIscsiDifferentLunSameNode(driver, pattern, l.config, l.resources, l.cs, l.ns, false)
	})
}

// testAccessMultipleVolumes tests access to multiple volumes from single pod on the specified node
// If readSeedBase > 0, read test are done before write/read test assuming that there is already data written.
func testAccessMultipleVolumes(f *framework.Framework, cs clientset.Interface, ns string,
	node framework.NodeSelection, pvcs []*v1.PersistentVolumeClaim, readSeedBase int64, writeSeedBase int64) string {
	By(fmt.Sprintf("Creating pod on %+v with multiple volumes", node))
	pod, err := framework.CreateSecPodWithNodeSelection(cs, ns, pvcs,
		false, "", false, false, framework.SELinuxLabel,
		nil, node, framework.PodStartTimeout)
	defer func() {
		framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))
	}()
	Expect(err).NotTo(HaveOccurred())

	byteLen := 64
	for i, pvc := range pvcs {
		// CreateSecPodWithNodeSelection make volumes accessible via /mnt/volume({i} + 1)
		index := i + 1
		path := fmt.Sprintf("/mnt/volume%d", index)
		By(fmt.Sprintf("Checking if the volume%d exists as expected volume mode (%s)", index, *pvc.Spec.VolumeMode))
		utils.CheckVolumeModeOfPath(pod, *pvc.Spec.VolumeMode, path)

		if readSeedBase > 0 {
			By(fmt.Sprintf("Checking if read from the volume%d works properly", index))
			utils.CheckReadFromPath(pod, *pvc.Spec.VolumeMode, path, byteLen, readSeedBase+int64(i))
		}

		By(fmt.Sprintf("Checking if write to the volume%d works properly", index))
		utils.CheckWriteToPath(pod, *pvc.Spec.VolumeMode, path, byteLen, writeSeedBase+int64(i))

		By(fmt.Sprintf("Checking if read from the volume%d works properly", index))
		utils.CheckReadFromPath(pod, *pvc.Spec.VolumeMode, path, byteLen, writeSeedBase+int64(i))
	}

	pod, err = cs.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "get pod")
	return pod.Spec.NodeName
}

// TestAccessMultipleVolumesAcrossPodRecreation tests access to multiple volumes from single pod,
// then recreate pod on the same or different node depending on requiresSameNode,
// and recheck access to the volumes from the recreated pod
func TestAccessMultipleVolumesAcrossPodRecreation(f *framework.Framework, cs clientset.Interface, ns string,
	node framework.NodeSelection, pvcs []*v1.PersistentVolumeClaim, requiresSameNode bool) {

	// No data is written in volume, so passing negative value
	readSeedBase := int64(-1)
	writeSeedBase := time.Now().UTC().UnixNano()
	// Test access to multiple volumes on the specified node
	nodeName := testAccessMultipleVolumes(f, cs, ns, node, pvcs, readSeedBase, writeSeedBase)

	// Set affinity depending on requiresSameNode
	if requiresSameNode {
		framework.SetAffinity(&node, nodeName)
	} else {
		framework.SetAntiAffinity(&node, nodeName)
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
	node framework.NodeSelection, pvc *v1.PersistentVolumeClaim, numPods int, requiresSameNode bool) {

	var pods []*v1.Pod

	// Create each pod with pvc
	for i := 0; i < numPods; i++ {
		index := i + 1
		By(fmt.Sprintf("Creating pod%d with a volume on %+v", index, node))
		pod, err := framework.CreateSecPodWithNodeSelection(cs, ns,
			[]*v1.PersistentVolumeClaim{pvc},
			false, "", false, false, framework.SELinuxLabel,
			nil, node, framework.PodStartTimeout)
		defer func() {
			framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))
		}()
		Expect(err).NotTo(HaveOccurred())
		pod, err = cs.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
		pods = append(pods, pod)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("get pod%d", index))
		actualNodeName := pod.Spec.NodeName

		// Set affinity depending on requiresSameNode
		if requiresSameNode {
			framework.SetAffinity(&node, actualNodeName)
		} else {
			framework.SetAntiAffinity(&node, actualNodeName)
		}
	}

	var seed int64
	byteLen := 64
	path := "/mnt/volume1"
	// Check if volume can be accessed from each pod
	for i, pod := range pods {
		index := i + 1
		By(fmt.Sprintf("Checking if the volume in pod%d exists as expected volume mode (%s)", index, *pvc.Spec.VolumeMode))
		utils.CheckVolumeModeOfPath(pod, *pvc.Spec.VolumeMode, path)

		if i != 0 {
			By(fmt.Sprintf("From pod%d, checking if reading the data that pod%d write works properly", index, index-1))
			// For 1st pod, no one has written data yet, so pass the read check
			utils.CheckReadFromPath(pod, *pvc.Spec.VolumeMode, path, byteLen, seed)
		}

		// Update the seed and check if write/read works properly
		seed = time.Now().UTC().UnixNano()

		By(fmt.Sprintf("Checking if write to the volume in pod%d works properly", index))
		utils.CheckWriteToPath(pod, *pvc.Spec.VolumeMode, path, byteLen, seed)

		By(fmt.Sprintf("Checking if read from the volume in pod%d works properly", index))
		utils.CheckReadFromPath(pod, *pvc.Spec.VolumeMode, path, byteLen, seed)
	}

	// Delete the last pod and remove from slice of pods
	if len(pods) < 2 {
		framework.Failf("Number of pods shouldn't be less than 2, but got %d", len(pods))
	}
	lastPod := pods[len(pods)-1]
	framework.ExpectNoError(framework.DeletePodWithWait(f, cs, lastPod))
	pods = pods[:len(pods)-1]

	// Recheck if pv can be accessed from each pod after the last pod deletion
	for i, pod := range pods {
		index := i + 1
		// index of pod and index of pvc match, because pods are created above way
		By(fmt.Sprintf("Rechecking if the volume in pod%d exists as expected volume mode (%s)", index, *pvc.Spec.VolumeMode))
		utils.CheckVolumeModeOfPath(pod, *pvc.Spec.VolumeMode, "/mnt/volume1")

		if i == 0 {
			// This time there should be data that last pod wrote, for 1st pod
			By(fmt.Sprintf("From pod%d, rechecking if reading the data that last pod write works properly", index))
		} else {
			By(fmt.Sprintf("From pod%d, rechecking if reading the data that pod%d write works properly", index, index-1))
		}
		utils.CheckReadFromPath(pod, *pvc.Spec.VolumeMode, path, byteLen, seed)

		// Update the seed and check if write/read works properly
		seed = time.Now().UTC().UnixNano()

		By(fmt.Sprintf("Rechecking if write to the volume in pod%d works properly", index))
		utils.CheckWriteToPath(pod, *pvc.Spec.VolumeMode, path, byteLen, seed)

		By(fmt.Sprintf("Rechecking if read from the volume in pod%d works properly", index))
		utils.CheckReadFromPath(pod, *pvc.Spec.VolumeMode, path, byteLen, seed)
	}
}

func testIscsiDifferentLunSameNode(driver TestDriver, pattern testpatterns.TestPattern, config *PerTestConfig, resources []*genericVolumeTestResource, cs clientset.Interface, ns *v1.Namespace, sameVolumeMode bool) {
	var pvcs []*v1.PersistentVolumeClaim
	var pods []*v1.Pod
	numVols := 2
	nodeSelection := framework.NodeSelection{}
	if config.ClientNodeName != "" {
		nodeSelection.Name = config.ClientNodeName
	}
	path := "/mnt/volume1"

	// The iscsi driver creates 1 volume server with 2 luns. Subsequent calls to
	// its GetPersistentVolumeSource cycle through lun 0 and 1
	// So create a pod 0 using lun 0 and and pod 1 using lun 1
	for i := 0; i < numVols; i++ {
		if i > 0 && !sameVolumeMode {
			if pattern.VolMode == v1.PersistentVolumeBlock {
				pattern.VolMode = v1.PersistentVolumeFilesystem
			} else {
				pattern.VolMode = v1.PersistentVolumeBlock
			}
		}

		resource := createGenericVolumeTestResource(driver, config, pattern)
		resources = append(resources, resource)
		pvcs = append(pvcs, resource.pvc)

		description := fmt.Sprintf("Creating pod %d on %+v using volume lun %d", i, nodeSelection, i)
		By(description)
		pod, err := framework.CreateSecPodWithNodeSelection(cs, ns.Name, []*v1.PersistentVolumeClaim{resource.pvc},
			false, "", false, false, framework.SELinuxLabel,
			nil, nodeSelection, framework.PodStartTimeout)
		defer func() {
			framework.ExpectNoError(framework.DeletePodWithWait(config.Framework, cs, pod))
		}()
		Expect(err).NotTo(HaveOccurred(), description)

		By(fmt.Sprintf("Checking in pod %d if volume1 exists as expected volume mode (%s)", i, *resource.pvc.Spec.VolumeMode))
		utils.CheckVolumeModeOfPath(pod, *resource.pvc.Spec.VolumeMode, path)

		By(fmt.Sprintf("Checking in pod %d if read and write to volume1 works properly", i))
		utils.CheckReadWriteToPath(pod, *resource.pvc.Spec.VolumeMode, path)

		pod, err = cs.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "get pod")
		nodeSelection.Name = pod.Spec.NodeName
		pods = append(pods, pod)
	}

	description := fmt.Sprintf("Deleting pod 1 %q on %+v using volume lun 1", pods[1].Name, nodeSelection)
	By(description)
	framework.ExpectNoError(framework.DeletePodWithWait(config.Framework, cs, pods[1]), description)

	By("Checking in pod 0 if read and write to the volume1 still works properly")
	utils.CheckReadWriteToPath(pods[0], *pvcs[0].Spec.VolumeMode, path)
}
