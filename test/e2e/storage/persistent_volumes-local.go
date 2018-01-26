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

package storage

import (
	"fmt"
	"math/rand"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ghodss/yaml"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	extv1beta1 "k8s.io/api/extensions/v1beta1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

type localTestConfig struct {
	ns       string
	nodes    []v1.Node
	node0    *v1.Node
	client   clientset.Interface
	scName   string
	ssTester *framework.StatefulSetTester
}

type localVolumeType string

const (
	// default local volume type, aka a directory
	DirectoryLocalVolumeType localVolumeType = "dir"
	// creates a tmpfs and mounts it
	TmpfsLocalVolumeType localVolumeType = "tmpfs"
	// tests based on local ssd at /mnt/disks/by-uuid/
	GCELocalSSDVolumeType localVolumeType = "gce-localssd-scsi-fs"
)

var setupLocalVolumeMap = map[localVolumeType]func(*localTestConfig, *v1.Node) *localTestVolume{
	GCELocalSSDVolumeType:    setupLocalVolumeGCELocalSSD,
	TmpfsLocalVolumeType:     setupLocalVolumeTmpfs,
	DirectoryLocalVolumeType: setupLocalVolumeDirectory,
}

var cleanupLocalVolumeMap = map[localVolumeType]func(*localTestConfig, *localTestVolume){
	GCELocalSSDVolumeType:    cleanupLocalVolumeGCELocalSSD,
	TmpfsLocalVolumeType:     cleanupLocalVolumeTmpfs,
	DirectoryLocalVolumeType: cleanupLocalVolumeDirectory,
}

type localTestVolume struct {
	// Node that the volume is on
	node *v1.Node
	// Path to the volume on the host node
	hostDir string
	// PVC for this volume
	pvc *v1.PersistentVolumeClaim
	// PV for this volume
	pv *v1.PersistentVolume
	// Type of local volume
	localVolumeType localVolumeType
}

const (
	// TODO: This may not be available/writable on all images.
	hostBase      = "/tmp"
	containerBase = "/myvol"
	// 'hostBase + discoveryDir' is the path for volume discovery.
	discoveryDir = "disks"
	// Path to the first volume in the test containers
	// created via createLocalPod or makeLocalPod
	// leveraging pv_util.MakePod
	volumeDir = "/mnt/volume1"
	// testFile created in setupLocalVolume
	testFile = "test-file"
	// testFileContent written into testFile
	testFileContent = "test-file-content"
	testSCPrefix    = "local-volume-test-storageclass"

	// Following are constants used for provisioner e2e tests.
	//
	// testServiceAccount is the service account for bootstrapper
	testServiceAccount = "local-storage-admin"
	// volumeConfigName is the configmap passed to bootstrapper and provisioner
	volumeConfigName = "local-volume-config"
	// provisioner image used for e2e tests
	provisionerImageName = "quay.io/external_storage/local-volume-provisioner:v2.0.0"
	// provisioner daemonSetName name
	daemonSetName = "local-volume-provisioner"
	// provisioner default mount point folder
	provisionerDefaultMountRoot = "/mnt/local-storage"
	// provisioner node/pv cluster role binding
	nodeBindingName         = "local-storage:provisioner-node-binding"
	pvBindingName           = "local-storage:provisioner-pv-binding"
	systemRoleNode          = "system:node"
	systemRolePVProvisioner = "system:persistent-volume-provisioner"

	// A sample request size
	testRequestSize = "10Mi"
)

var (
	// storage class volume binding modes
	waitMode      = storagev1.VolumeBindingWaitForFirstConsumer
	immediateMode = storagev1.VolumeBindingImmediate

	// Common selinux labels
	selinuxLabel = &v1.SELinuxOptions{
		Level: "s0:c0,c1"}
)

var _ = utils.SIGDescribe("PersistentVolumes-local [Feature:LocalPersistentVolumes] [Serial]", func() {
	f := framework.NewDefaultFramework("persistent-local-volumes-test")

	var (
		config *localTestConfig
		scName string
	)

	BeforeEach(func() {
		// Get all the schedulable nodes
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodes.Items)).NotTo(BeZero(), "No available nodes for scheduling")
		scName = fmt.Sprintf("%v-%v-%v", testSCPrefix, f.Namespace.Name, rand.Int())
		// Choose the first node
		node0 := &nodes.Items[0]

		ssTester := framework.NewStatefulSetTester(f.ClientSet)
		config = &localTestConfig{
			ns:       f.Namespace.Name,
			client:   f.ClientSet,
			nodes:    nodes.Items,
			node0:    node0,
			scName:   scName,
			ssTester: ssTester,
		}
	})

	Context("when one pod requests one prebound PVC", func() {

		var testVol *localTestVolume

		BeforeEach(func() {
			setupStorageClass(config, &waitMode)
			testVols := setupLocalVolumesPVCsPVs(config, DirectoryLocalVolumeType, config.node0, 1, waitMode)
			testVol = testVols[0]
		})

		AfterEach(func() {
			cleanupLocalVolumes(config, []*localTestVolume{testVol})
			cleanupStorageClass(config)
		})

		It("should be able to mount volume and read from pod1", func() {
			By("Creating pod1")
			pod1, pod1Err := createLocalPod(config, testVol)
			Expect(pod1Err).NotTo(HaveOccurred())
			verifyLocalPod(config, testVol, pod1, config.node0.Name)

			By("Reading in pod1")
			// testFileContent was written during setupLocalVolume
			testReadFileContent(volumeDir, testFile, testFileContent, pod1)

			By("Deleting pod1")
			framework.DeletePodOrFail(config.client, config.ns, pod1.Name)
		})

		It("should be able to mount volume and write from pod1", func() {
			By("Creating pod1")
			pod1, pod1Err := createLocalPod(config, testVol)
			Expect(pod1Err).NotTo(HaveOccurred())
			verifyLocalPod(config, testVol, pod1, config.node0.Name)

			// testFileContent was written during setupLocalVolume
			testReadFileContent(volumeDir, testFile, testFileContent, pod1)

			By("Writing in pod1")
			writeCmd, _ := createWriteAndReadCmds(volumeDir, testFile, testVol.hostDir /*writeTestFileContent*/)
			podRWCmdExec(pod1, writeCmd)

			By("Deleting pod1")
			framework.DeletePodOrFail(config.client, config.ns, pod1.Name)
		})
	})

	localVolumeTypes := []localVolumeType{DirectoryLocalVolumeType, TmpfsLocalVolumeType, GCELocalSSDVolumeType}
	for _, tempTestVolType := range localVolumeTypes {

		// New variable required for gingko test closures
		testVolType := tempTestVolType
		ctxString := fmt.Sprintf("[Volume type: %s]", testVolType)
		testMode := immediateMode

		Context(ctxString, func() {

			BeforeEach(func() {
				if testVolType == GCELocalSSDVolumeType {
					SkipUnlessLocalSSDExists("scsi", "fs", config.node0)
				}
				setupStorageClass(config, &testMode)

			})

			AfterEach(func() {
				cleanupStorageClass(config)
			})

			Context("when two pods mount a local volume at the same time", func() {
				It("should be able to write from pod1 and read from pod2", func() {
					var testVol *localTestVolume
					testVols := setupLocalVolumesPVCsPVs(config, testVolType, config.node0, 1, testMode)
					testVol = testVols[0]
					twoPodsReadWriteTest(config, testVol)
					cleanupLocalVolumes(config, testVols)
				})
			})

			Context("when two pods mount a local volume one after the other", func() {
				It("should be able to write from pod1 and read from pod2", func() {
					var testVol *localTestVolume
					testVols := setupLocalVolumesPVCsPVs(config, testVolType, config.node0, 1, testMode)
					testVol = testVols[0]
					twoPodsReadWriteSerialTest(config, testVol)
					cleanupLocalVolumes(config, testVols)
				})
			})

			Context("when pod using local volume with non-existant path", func() {

				ep := &eventPatterns{
					reason:  "FailedMount",
					pattern: make([]string, 2)}
				ep.pattern = append(ep.pattern, "MountVolume.SetUp failed")
				ep.pattern = append(ep.pattern, "does not exist")

				It("should not be able to mount", func() {
					testVol := &localTestVolume{
						node:            config.node0,
						hostDir:         "/non-existent/location/nowhere",
						localVolumeType: testVolType,
					}
					By("Creating local PVC and PV")
					createLocalPVCsPVs(config, []*localTestVolume{testVol}, testMode)
					pod, err := createLocalPod(config, testVol)
					Expect(err).To(HaveOccurred())
					checkPodEvents(config, pod.Name, ep)
					verifyLocalVolume(config, testVol)
					cleanupLocalPVCsPVs(config, []*localTestVolume{testVol})
				})
			})

			Context("when pod's node is different from PV's NodeAffinity", func() {

				BeforeEach(func() {
					if len(config.nodes) < 2 {
						framework.Skipf("Runs only when number of nodes >= 2")
					}
				})

				ep := &eventPatterns{
					reason:  "FailedScheduling",
					pattern: make([]string, 2)}
				ep.pattern = append(ep.pattern, "MatchNodeSelector")
				ep.pattern = append(ep.pattern, "VolumeNodeAffinityConflict")

				It("should not be able to mount due to different NodeAffinity", func() {
					testPodWithNodeName(config, testVolType, ep, config.nodes[1].Name, makeLocalPodWithNodeAffinity, testMode)
				})

				It("should not be able to mount due to different NodeSelector", func() {
					testPodWithNodeName(config, testVolType, ep, config.nodes[1].Name, makeLocalPodWithNodeSelector, testMode)
				})

			})

			Context("when pod's node is different from PV's NodeName", func() {

				BeforeEach(func() {
					if len(config.nodes) < 2 {
						framework.Skipf("Runs only when number of nodes >= 2")
					}
				})

				ep := &eventPatterns{
					reason:  "FailedMount",
					pattern: make([]string, 2)}
				ep.pattern = append(ep.pattern, "NodeSelectorTerm")
				ep.pattern = append(ep.pattern, "MountVolume.NodeAffinity check failed")

				It("should not be able to mount due to different NodeName", func() {
					testPodWithNodeName(config, testVolType, ep, config.nodes[1].Name, makeLocalPodWithNodeName, testMode)
				})
			})
		})
	}

	Context("when using local volume provisioner", func() {
		var volumePath string

		BeforeEach(func() {
			setupStorageClass(config, &immediateMode)
			setupLocalVolumeProvisioner(config)
			volumePath = path.Join(
				hostBase, discoveryDir, fmt.Sprintf("vol-%v", string(uuid.NewUUID())))
			setupLocalVolumeProvisionerMountPoint(config, volumePath)
		})

		AfterEach(func() {
			cleanupLocalVolumeProvisionerMountPoint(config, volumePath)
			cleanupLocalVolumeProvisioner(config, volumePath)
			cleanupStorageClass(config)
		})

		It("should create and recreate local persistent volume", func() {
			By("Starting a provisioner daemonset")
			createProvisionerDaemonset(config)
			kind := schema.GroupKind{Group: "extensions", Kind: "DaemonSet"}
			framework.WaitForControlledPodsRunning(config.client, config.ns, daemonSetName, kind)

			By("Waiting for a PersitentVolume to be created")
			oldPV, err := waitForLocalPersistentVolume(config.client, volumePath)
			Expect(err).NotTo(HaveOccurred())

			// Create a persistent volume claim for local volume: the above volume will be bound.
			By("Creating a persistent volume claim")
			claim, err := config.client.CoreV1().PersistentVolumeClaims(config.ns).Create(newLocalClaim(config))
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForPersistentVolumeClaimPhase(
				v1.ClaimBound, config.client, claim.Namespace, claim.Name, framework.Poll, 1*time.Minute)
			Expect(err).NotTo(HaveOccurred())

			claim, err = config.client.CoreV1().PersistentVolumeClaims(config.ns).Get(claim.Name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			Expect(claim.Spec.VolumeName).To(Equal(oldPV.Name))

			// Delete the persistent volume claim: file will be cleaned up and volume be re-created.
			By("Deleting the persistent volume claim to clean up persistent volume and re-create one")
			writeCmd, _ := createWriteAndReadCmds(volumePath, testFile, testFileContent)
			err = framework.IssueSSHCommand(writeCmd, framework.TestContext.Provider, config.node0)
			Expect(err).NotTo(HaveOccurred())
			err = config.client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(claim.Name, &metav1.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred())

			By("Waiting for a new PersistentVolume to be re-created")
			newPV, err := waitForLocalPersistentVolume(config.client, volumePath)
			Expect(err).NotTo(HaveOccurred())
			Expect(newPV.UID).NotTo(Equal(oldPV.UID))
			fileDoesntExistCmd := createFileDoesntExistCmd(volumePath, testFile)
			err = framework.IssueSSHCommand(fileDoesntExistCmd, framework.TestContext.Provider, config.node0)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("when StatefulSet has pod anti-affinity", func() {
		var testVols map[string][]*localTestVolume
		const (
			ssReplicas  = 3
			volsPerNode = 2
		)

		BeforeEach(func() {
			if len(config.nodes) < ssReplicas {
				framework.Skipf("Runs only when number of nodes >= %v", ssReplicas)
			}
			setupStorageClass(config, &waitMode)

			testVols = map[string][]*localTestVolume{}
			for i, node := range config.nodes {
				// The PVCs created here won't be used
				By(fmt.Sprintf("Setting up local volumes on node %q", node.Name))
				vols := setupLocalVolumesPVCsPVs(config, DirectoryLocalVolumeType, &config.nodes[i], volsPerNode, waitMode)
				testVols[node.Name] = vols
			}
		})

		AfterEach(func() {
			for _, vols := range testVols {
				cleanupLocalVolumes(config, vols)
			}
			cleanupStorageClass(config)
		})

		It("should use volumes spread across nodes", func() {
			By("Creating a StatefulSet with pod anti-affinity on nodes")
			ss := createStatefulSet(config, ssReplicas, volsPerNode)
			validateStatefulSet(config, ss)
		})
	})

	// TODO: add stress test that creates many pods in parallel across multiple nodes
})

type makeLocalPodWith func(config *localTestConfig, volume *localTestVolume, nodeName string) *v1.Pod

func testPodWithNodeName(config *localTestConfig, testVolType localVolumeType, ep *eventPatterns, nodeName string, makeLocalPodFunc makeLocalPodWith, bindingMode storagev1.VolumeBindingMode) {
	By(fmt.Sprintf("local-volume-type: %s", testVolType))
	testVols := setupLocalVolumesPVCsPVs(config, testVolType, config.node0, 1, bindingMode)
	testVol := testVols[0]

	pod := makeLocalPodFunc(config, testVol, nodeName)
	pod, err := config.client.CoreV1().Pods(config.ns).Create(pod)
	Expect(err).NotTo(HaveOccurred())
	err = framework.WaitForPodRunningInNamespace(config.client, pod)
	Expect(err).To(HaveOccurred())
	checkPodEvents(config, pod.Name, ep)

	cleanupLocalVolumes(config, []*localTestVolume{testVol})
}

type eventPatterns struct {
	reason  string
	pattern []string
}

func checkPodEvents(config *localTestConfig, podName string, ep *eventPatterns) {
	var events *v1.EventList
	selector := fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      podName,
		"involvedObject.namespace": config.ns,
		"reason":                   ep.reason,
	}.AsSelector().String()
	options := metav1.ListOptions{FieldSelector: selector}
	events, err := config.client.CoreV1().Events(config.ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(events.Items)).NotTo(Equal(0))
	for _, p := range ep.pattern {
		Expect(events.Items[0].Message).To(ContainSubstring(p))
	}
}

// The tests below are run against multiple mount point types

// Test two pods at the same time, write from pod1, and read from pod2
func twoPodsReadWriteTest(config *localTestConfig, testVol *localTestVolume) {
	By("Creating pod1 to write to the PV")
	pod1, pod1Err := createLocalPod(config, testVol)
	Expect(pod1Err).NotTo(HaveOccurred())
	verifyLocalPod(config, testVol, pod1, config.node0.Name)

	// testFileContent was written during setupLocalVolume
	testReadFileContent(volumeDir, testFile, testFileContent, pod1)

	By("Creating pod2 to read from the PV")
	pod2, pod2Err := createLocalPod(config, testVol)
	Expect(pod2Err).NotTo(HaveOccurred())
	verifyLocalPod(config, testVol, pod2, config.node0.Name)

	// testFileContent was written during setupLocalVolume
	testReadFileContent(volumeDir, testFile, testFileContent, pod2)

	writeCmd := createWriteCmd(volumeDir, testFile, testVol.hostDir /*writeTestFileContent*/)

	By("Writing in pod1")
	podRWCmdExec(pod1, writeCmd)

	By("Reading in pod2")
	testReadFileContent(volumeDir, testFile, testVol.hostDir, pod2)

	By("Deleting pod1")
	framework.DeletePodOrFail(config.client, config.ns, pod1.Name)
	By("Deleting pod2")
	framework.DeletePodOrFail(config.client, config.ns, pod2.Name)
}

// Test two pods one after other, write from pod1, and read from pod2
func twoPodsReadWriteSerialTest(config *localTestConfig, testVol *localTestVolume) {
	By("Creating pod1")
	pod1, pod1Err := createLocalPod(config, testVol)
	Expect(pod1Err).NotTo(HaveOccurred())
	verifyLocalPod(config, testVol, pod1, config.node0.Name)

	// testFileContent was written during setupLocalVolume
	testReadFileContent(volumeDir, testFile, testFileContent, pod1)

	writeCmd := createWriteCmd(volumeDir, testFile, testVol.hostDir /*writeTestFileContent*/)

	By("Writing in pod1")
	podRWCmdExec(pod1, writeCmd)

	By("Deleting pod1")
	framework.DeletePodOrFail(config.client, config.ns, pod1.Name)

	By("Creating pod2")
	pod2, pod2Err := createLocalPod(config, testVol)
	Expect(pod2Err).NotTo(HaveOccurred())
	verifyLocalPod(config, testVol, pod2, config.node0.Name)

	By("Reading in pod2")
	testReadFileContent(volumeDir, testFile, testVol.hostDir, pod2)

	By("Deleting pod2")
	framework.DeletePodOrFail(config.client, config.ns, pod2.Name)
}

func setupStorageClass(config *localTestConfig, mode *storagev1.VolumeBindingMode) {
	sc := &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: config.scName,
		},
		Provisioner:       "kubernetes.io/no-provisioner",
		VolumeBindingMode: mode,
	}

	sc, err := config.client.StorageV1().StorageClasses().Create(sc)
	Expect(err).NotTo(HaveOccurred())
}

func cleanupStorageClass(config *localTestConfig) {
	framework.ExpectNoError(config.client.StorageV1().StorageClasses().Delete(config.scName, nil))
}

// podNode wraps RunKubectl to get node where pod is running
func podNodeName(config *localTestConfig, pod *v1.Pod) (string, error) {
	runtimePod, runtimePodErr := config.client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	return runtimePod.Spec.NodeName, runtimePodErr
}

// setupLocalVolumes sets up directories to use for local PV
func setupLocalVolumes(config *localTestConfig, localVolumeType localVolumeType, node *v1.Node, count int) []*localTestVolume {
	vols := []*localTestVolume{}
	for i := 0; i < count; i++ {
		setupLocalVolume, ok := setupLocalVolumeMap[localVolumeType]
		Expect(ok).To(BeTrue())
		testVol := setupLocalVolume(config, node)
		vols = append(vols, testVol)
	}
	return vols
}

func cleanupLocalPVCsPVs(config *localTestConfig, volumes []*localTestVolume) {
	for _, volume := range volumes {
		By("Cleaning up PVC and PV")
		errs := framework.PVPVCCleanup(config.client, config.ns, volume.pv, volume.pvc)
		if len(errs) > 0 {
			framework.Failf("Failed to delete PV and/or PVC: %v", utilerrors.NewAggregate(errs))
		}
	}
}

// Deletes the PVC/PV, and launches a pod with hostpath volume to remove the test directory
func cleanupLocalVolumes(config *localTestConfig, volumes []*localTestVolume) {
	cleanupLocalPVCsPVs(config, volumes)

	for _, volume := range volumes {
		cleanup := cleanupLocalVolumeMap[volume.localVolumeType]
		cleanup(config, volume)
	}
}

func setupWriteTestFile(hostDir string, config *localTestConfig, localVolumeType localVolumeType, node *v1.Node) *localTestVolume {
	writeCmd, _ := createWriteAndReadCmds(hostDir, testFile, testFileContent)
	By(fmt.Sprintf("Creating local volume on node %q at path %q", node.Name, hostDir))
	err := framework.IssueSSHCommand(writeCmd, framework.TestContext.Provider, node)
	Expect(err).NotTo(HaveOccurred())
	return &localTestVolume{
		node:            node,
		hostDir:         hostDir,
		localVolumeType: localVolumeType,
	}
}

func setupLocalVolumeTmpfs(config *localTestConfig, node *v1.Node) *localTestVolume {
	testDirName := "local-volume-test-" + string(uuid.NewUUID())
	hostDir := filepath.Join(hostBase, testDirName)
	createAndMountTmpfsLocalVolume(config, hostDir, node)
	// populate volume with testFile containing testFileContent
	return setupWriteTestFile(hostDir, config, TmpfsLocalVolumeType, node)
}

func setupLocalVolumeGCELocalSSD(config *localTestConfig, node *v1.Node) *localTestVolume {
	res, err := framework.IssueSSHCommandWithResult("ls /mnt/disks/by-uuid/google-local-ssds-scsi-fs/", framework.TestContext.Provider, node)
	Expect(err).NotTo(HaveOccurred())
	dirName := strings.Fields(res.Stdout)[0]
	hostDir := "/mnt/disks/by-uuid/google-local-ssds-scsi-fs/" + dirName
	// populate volume with testFile containing testFileContent
	return setupWriteTestFile(hostDir, config, GCELocalSSDVolumeType, node)
}

func setupLocalVolumeDirectory(config *localTestConfig, node *v1.Node) *localTestVolume {
	testDirName := "local-volume-test-" + string(uuid.NewUUID())
	hostDir := filepath.Join(hostBase, testDirName)
	// populate volume with testFile containing testFileContent
	return setupWriteTestFile(hostDir, config, DirectoryLocalVolumeType, node)
}

func verifyLocalVolume(config *localTestConfig, volume *localTestVolume) {
	framework.ExpectNoError(framework.WaitOnPVandPVC(config.client, config.ns, volume.pv, volume.pvc))
}

func verifyLocalPod(config *localTestConfig, volume *localTestVolume, pod *v1.Pod, expectedNodeName string) {
	podNodeName, err := podNodeName(config, pod)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("pod %q created on Node %q", pod.Name, podNodeName)
	Expect(podNodeName).To(Equal(expectedNodeName))
}

// Deletes the PVC/PV, and launches a pod with hostpath volume to remove the test directory
func cleanupLocalVolumeGCELocalSSD(config *localTestConfig, volume *localTestVolume) {
	By("Removing the test directory")
	removeCmd := fmt.Sprintf("rm %s", volume.hostDir+"/"+testFile)
	err := framework.IssueSSHCommand(removeCmd, framework.TestContext.Provider, volume.node)
	Expect(err).NotTo(HaveOccurred())
}

// Deletes the PVC/PV, and launches a pod with hostpath volume to remove the test directory
func cleanupLocalVolumeTmpfs(config *localTestConfig, volume *localTestVolume) {
	unmountTmpfsLocalVolume(config, volume.hostDir, volume.node)

	By("Removing the test directory")
	removeCmd := fmt.Sprintf("rm -r %s", volume.hostDir)
	err := framework.IssueSSHCommand(removeCmd, framework.TestContext.Provider, volume.node)
	Expect(err).NotTo(HaveOccurred())
}

// Deletes the PVC/PV, and launches a pod with hostpath volume to remove the test directory
func cleanupLocalVolumeDirectory(config *localTestConfig, volume *localTestVolume) {
	By("Removing the test directory")
	removeCmd := fmt.Sprintf("rm -r %s", volume.hostDir)
	err := framework.IssueSSHCommand(removeCmd, framework.TestContext.Provider, volume.node)
	Expect(err).NotTo(HaveOccurred())
}

func makeLocalPVCConfig(config *localTestConfig) framework.PersistentVolumeClaimConfig {
	return framework.PersistentVolumeClaimConfig{
		AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		StorageClassName: &config.scName,
	}
}

func makeLocalPVConfig(config *localTestConfig, volume *localTestVolume) framework.PersistentVolumeConfig {
	// TODO: hostname may not be the best option
	nodeKey := "kubernetes.io/hostname"
	if volume.node.Labels == nil {
		framework.Failf("Node does not have labels")
	}
	nodeValue, found := volume.node.Labels[nodeKey]
	if !found {
		framework.Failf("Node does not have required label %q", nodeKey)
	}

	return framework.PersistentVolumeConfig{
		PVSource: v1.PersistentVolumeSource{
			Local: &v1.LocalVolumeSource{
				Path: volume.hostDir,
			},
		},
		NamePrefix:       "local-pv",
		StorageClassName: config.scName,
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      nodeKey,
								Operator: v1.NodeSelectorOpIn,
								Values:   []string{nodeValue},
							},
						},
					},
				},
			},
		},
	}
}

// Creates a PVC and PV with prebinding
func createLocalPVCsPVs(config *localTestConfig, volumes []*localTestVolume, mode storagev1.VolumeBindingMode) {
	var err error

	for _, volume := range volumes {
		pvcConfig := makeLocalPVCConfig(config)
		pvConfig := makeLocalPVConfig(config, volume)
		volume.pv, volume.pvc, err = framework.CreatePVPVC(config.client, pvConfig, pvcConfig, config.ns, false)
		framework.ExpectNoError(err)
	}

	if mode == storagev1.VolumeBindingImmediate {
		for _, volume := range volumes {
			verifyLocalVolume(config, volume)
		}
	} else {
		// Verify PVCs are not bound
		// There isn't really a great way to verify this without making the test be slow...
		err = wait.PollImmediate(time.Second, 10*time.Second, func() (done bool, err error) {
			for _, volume := range volumes {
				pvc, err := config.client.CoreV1().PersistentVolumeClaims(volume.pvc.Namespace).Get(volume.pvc.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				Expect(pvc.Status.Phase).To(Equal(v1.ClaimPending))
			}
			return false, nil
		})
		Expect(err).To(HaveOccurred())
	}
}

func makeLocalPod(config *localTestConfig, volume *localTestVolume, cmd string) *v1.Pod {
	return framework.MakeSecPod(config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, cmd, false, false, selinuxLabel)
}

func makeLocalPodWithNodeAffinity(config *localTestConfig, volume *localTestVolume, nodeName string) (pod *v1.Pod) {
	pod = framework.MakeSecPod(config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, "", false, false, selinuxLabel)
	if pod == nil {
		return
	}
	affinity := &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      "kubernetes.io/hostname",
								Operator: v1.NodeSelectorOpIn,
								Values:   []string{nodeName},
							},
						},
					},
				},
			},
		},
	}
	pod.Spec.Affinity = affinity
	return
}

func makeLocalPodWithNodeSelector(config *localTestConfig, volume *localTestVolume, nodeName string) (pod *v1.Pod) {
	pod = framework.MakeSecPod(config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, "", false, false, selinuxLabel)
	if pod == nil {
		return
	}
	ns := map[string]string{
		"kubernetes.io/hostname": nodeName,
	}
	pod.Spec.NodeSelector = ns
	return
}

func makeLocalPodWithNodeName(config *localTestConfig, volume *localTestVolume, nodeName string) (pod *v1.Pod) {
	pod = framework.MakeSecPod(config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, "", false, false, selinuxLabel)
	if pod == nil {
		return
	}
	pod.Spec.NodeName = nodeName
	return
}

// createSecPod should be used when Pod requires non default SELinux labels
func createSecPod(config *localTestConfig, volume *localTestVolume, hostIPC bool, hostPID bool, seLinuxLabel *v1.SELinuxOptions) (*v1.Pod, error) {
	pod, err := framework.CreateSecPod(config.client, config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, "", hostIPC, hostPID, seLinuxLabel)
	podNodeName, podNodeNameErr := podNodeName(config, pod)
	Expect(podNodeNameErr).NotTo(HaveOccurred())
	framework.Logf("Security Context POD %q created on Node %q", pod.Name, podNodeName)
	Expect(podNodeName).To(Equal(config.node0.Name))
	return pod, err
}

func createLocalPod(config *localTestConfig, volume *localTestVolume) (*v1.Pod, error) {
	return framework.CreateSecPod(config.client, config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, "", false, false, selinuxLabel)
}

func createAndMountTmpfsLocalVolume(config *localTestConfig, dir string, node *v1.Node) {
	By(fmt.Sprintf("Creating tmpfs mount point on node %q at path %q", node.Name, dir))
	err := framework.IssueSSHCommand(fmt.Sprintf("mkdir -p %q && sudo mount -t tmpfs -o size=1m tmpfs-%q %q", dir, dir, dir), framework.TestContext.Provider, node)
	Expect(err).NotTo(HaveOccurred())
}

func unmountTmpfsLocalVolume(config *localTestConfig, dir string, node *v1.Node) {
	By(fmt.Sprintf("Unmount tmpfs mount point on node %q at path %q", node.Name, dir))
	err := framework.IssueSSHCommand(fmt.Sprintf("sudo umount %q", dir), framework.TestContext.Provider, node)
	Expect(err).NotTo(HaveOccurred())
}

// Create corresponding write and read commands
// to be executed via SSH on the node with the local PV
func createWriteAndReadCmds(testFileDir string, testFile string, writeTestFileContent string) (writeCmd string, readCmd string) {
	writeCmd = createWriteCmd(testFileDir, testFile, writeTestFileContent)
	readCmd = createReadCmd(testFileDir, testFile)
	return writeCmd, readCmd
}

func createWriteCmd(testFileDir string, testFile string, writeTestFileContent string) string {
	testFilePath := filepath.Join(testFileDir, testFile)
	return fmt.Sprintf("mkdir -p %s; echo %s > %s", testFileDir, writeTestFileContent, testFilePath)
}
func createReadCmd(testFileDir string, testFile string) string {
	testFilePath := filepath.Join(testFileDir, testFile)
	return fmt.Sprintf("cat %s", testFilePath)
}

// Read testFile and evaluate whether it contains the testFileContent
func testReadFileContent(testFileDir string, testFile string, testFileContent string, pod *v1.Pod) {
	readCmd := createReadCmd(volumeDir, testFile)
	readOut := podRWCmdExec(pod, readCmd)
	Expect(readOut).To(ContainSubstring(testFileContent))
}

// Create command to verify that the file doesn't exist
// to be executed via SSH on the node with the local PV
func createFileDoesntExistCmd(testFileDir string, testFile string) string {
	testFilePath := filepath.Join(testFileDir, testFile)
	return fmt.Sprintf("[ ! -e %s ]", testFilePath)
}

// Execute a read or write command in a pod.
// Fail on error
func podRWCmdExec(pod *v1.Pod, cmd string) string {
	out, err := utils.PodExec(pod, cmd)
	Expect(err).NotTo(HaveOccurred())
	return out
}

// Initialize test volume on node
// and create local PVC and PV
func setupLocalVolumesPVCsPVs(
	config *localTestConfig,
	localVolumeType localVolumeType,
	node *v1.Node,
	count int,
	mode storagev1.VolumeBindingMode) []*localTestVolume {

	By("Initializing test volumes")
	testVols := setupLocalVolumes(config, localVolumeType, node, count)

	By("Creating local PVCs and PVs")
	createLocalPVCsPVs(config, testVols, mode)

	return testVols
}

func setupLocalVolumeProvisioner(config *localTestConfig) {
	By("Bootstrapping local volume provisioner")
	createServiceAccount(config)
	createProvisionerClusterRoleBinding(config)
	createVolumeConfigMap(config)

	By("Initializing local volume discovery base path")
	mkdirCmd := fmt.Sprintf("mkdir -p %v -m 777", path.Join(hostBase, discoveryDir))
	err := framework.IssueSSHCommand(mkdirCmd, framework.TestContext.Provider, config.node0)
	Expect(err).NotTo(HaveOccurred())
}

func cleanupLocalVolumeProvisioner(config *localTestConfig, volumePath string) {
	By("Cleaning up cluster role binding")
	deleteClusterRoleBinding(config)

	By("Removing the test discovery directory")
	removeCmd := fmt.Sprintf("rm -r %s", path.Join(hostBase, discoveryDir))
	err := framework.IssueSSHCommand(removeCmd, framework.TestContext.Provider, config.node0)
	Expect(err).NotTo(HaveOccurred())

	By("Cleaning up persistent volume")
	pv, err := findLocalPersistentVolume(config.client, volumePath)
	Expect(err).NotTo(HaveOccurred())
	err = config.client.CoreV1().PersistentVolumes().Delete(pv.Name, &metav1.DeleteOptions{})
	Expect(err).NotTo(HaveOccurred())
}

func setupLocalVolumeProvisionerMountPoint(config *localTestConfig, volumePath string) {
	By(fmt.Sprintf("Creating local directory at path %q", volumePath))
	mkdirCmd := fmt.Sprintf("mkdir %v -m 777", volumePath)
	err := framework.IssueSSHCommand(mkdirCmd, framework.TestContext.Provider, config.node0)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Mounting local directory at path %q", volumePath))
	mntCmd := fmt.Sprintf("sudo mount --bind %v %v", volumePath, volumePath)
	err = framework.IssueSSHCommand(mntCmd, framework.TestContext.Provider, config.node0)
	Expect(err).NotTo(HaveOccurred())
}

func cleanupLocalVolumeProvisionerMountPoint(config *localTestConfig, volumePath string) {
	By(fmt.Sprintf("Unmounting the test mount point from %q", volumePath))
	umountCmd := fmt.Sprintf("sudo umount %v", volumePath)
	err := framework.IssueSSHCommand(umountCmd, framework.TestContext.Provider, config.node0)
	Expect(err).NotTo(HaveOccurred())

	By("Removing the test mount point")
	removeCmd := fmt.Sprintf("rm -r %s", volumePath)
	err = framework.IssueSSHCommand(removeCmd, framework.TestContext.Provider, config.node0)

}

func createServiceAccount(config *localTestConfig) {
	serviceAccount := v1.ServiceAccount{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "ServiceAccount"},
		ObjectMeta: metav1.ObjectMeta{Name: testServiceAccount, Namespace: config.ns},
	}
	_, err := config.client.CoreV1().ServiceAccounts(config.ns).Create(&serviceAccount)
	Expect(err).NotTo(HaveOccurred())
}

// createProvisionerClusterRoleBinding creates two cluster role bindings for local volume provisioner's
// service account: systemRoleNode and systemRolePVProvisioner. These are required for
// provisioner to get node information and create persistent volumes.
func createProvisionerClusterRoleBinding(config *localTestConfig) {
	subjects := []rbacv1beta1.Subject{
		{
			Kind:      rbacv1beta1.ServiceAccountKind,
			Name:      testServiceAccount,
			Namespace: config.ns,
		},
	}

	pvBinding := rbacv1beta1.ClusterRoleBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "rbac.authorization.k8s.io/v1beta1",
			Kind:       "ClusterRoleBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: pvBindingName,
		},
		RoleRef: rbacv1beta1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     systemRolePVProvisioner,
		},
		Subjects: subjects,
	}
	nodeBinding := rbacv1beta1.ClusterRoleBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "rbac.authorization.k8s.io/v1beta1",
			Kind:       "ClusterRoleBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeBindingName,
		},
		RoleRef: rbacv1beta1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     systemRoleNode,
		},
		Subjects: subjects,
	}

	_, err := config.client.RbacV1beta1().ClusterRoleBindings().Create(&pvBinding)
	Expect(err).NotTo(HaveOccurred())
	_, err = config.client.RbacV1beta1().ClusterRoleBindings().Create(&nodeBinding)
	Expect(err).NotTo(HaveOccurred())
}

func deleteClusterRoleBinding(config *localTestConfig) {
	// These role bindings are created in provisioner; we just ensure it's
	// deleted and do not panic on error.
	config.client.RbacV1beta1().ClusterRoleBindings().Delete(nodeBindingName, metav1.NewDeleteOptions(0))
	config.client.RbacV1beta1().ClusterRoleBindings().Delete(pvBindingName, metav1.NewDeleteOptions(0))
}

func createVolumeConfigMap(config *localTestConfig) {
	// MountConfig and ProvisionerConfiguration from
	// https://github.com/kubernetes-incubator/external-storage/blob/master/local-volume/provisioner/pkg/common/common.go
	type MountConfig struct {
		// The hostpath directory
		HostDir  string `json:"hostDir" yaml:"hostDir"`
		MountDir string `json:"mountDir" yaml:"mountDir"`
	}
	type ProvisionerConfiguration struct {
		// StorageClassConfig defines configuration of Provisioner's storage classes
		StorageClassConfig map[string]MountConfig `json:"storageClassMap" yaml:"storageClassMap"`
	}
	var provisionerConfig ProvisionerConfiguration
	provisionerConfig.StorageClassConfig = map[string]MountConfig{
		config.scName: {
			HostDir:  path.Join(hostBase, discoveryDir),
			MountDir: provisionerDefaultMountRoot,
		},
	}

	data, err := yaml.Marshal(&provisionerConfig.StorageClassConfig)
	Expect(err).NotTo(HaveOccurred())

	configMap := v1.ConfigMap{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "ConfigMap",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      volumeConfigName,
			Namespace: config.ns,
		},
		Data: map[string]string{
			"storageClassMap": string(data),
		},
	}
	_, err = config.client.CoreV1().ConfigMaps(config.ns).Create(&configMap)
	Expect(err).NotTo(HaveOccurred())
}

func createProvisionerDaemonset(config *localTestConfig) {
	provisionerPrivileged := true
	provisioner := &extv1beta1.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "DaemonSet",
			APIVersion: "extensions/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: daemonSetName,
		},
		Spec: extv1beta1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "local-volume-provisioner"},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "local-volume-provisioner"},
				},
				Spec: v1.PodSpec{
					ServiceAccountName: testServiceAccount,
					Containers: []v1.Container{
						{
							Name:            "provisioner",
							Image:           provisionerImageName,
							ImagePullPolicy: "Always",
							SecurityContext: &v1.SecurityContext{
								Privileged: &provisionerPrivileged,
							},
							Env: []v1.EnvVar{
								{
									Name: "MY_NODE_NAME",
									ValueFrom: &v1.EnvVarSource{
										FieldRef: &v1.ObjectFieldSelector{
											FieldPath: "spec.nodeName",
										},
									},
								},
								{
									Name: "MY_NAMESPACE",
									ValueFrom: &v1.EnvVarSource{
										FieldRef: &v1.ObjectFieldSelector{
											FieldPath: "metadata.namespace",
										},
									},
								},
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      volumeConfigName,
									MountPath: "/etc/provisioner/config/",
								},
								{
									Name:      "local-disks",
									MountPath: provisionerDefaultMountRoot,
								},
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: volumeConfigName,
							VolumeSource: v1.VolumeSource{
								ConfigMap: &v1.ConfigMapVolumeSource{
									LocalObjectReference: v1.LocalObjectReference{
										Name: volumeConfigName,
									},
								},
							},
						},
						{
							Name: "local-disks",
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{
									Path: path.Join(hostBase, discoveryDir),
								},
							},
						},
					},
				},
			},
		},
	}
	_, err := config.client.ExtensionsV1beta1().DaemonSets(config.ns).Create(provisioner)
	Expect(err).NotTo(HaveOccurred())
}

// newLocalClaim creates a new persistent volume claim.
func newLocalClaim(config *localTestConfig) *v1.PersistentVolumeClaim {
	claim := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "local-pvc-",
			Namespace:    config.ns,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			StorageClassName: &config.scName,
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(testRequestSize),
				},
			},
		},
	}

	return &claim
}

// newLocalClaim creates a new persistent volume claim.
func newLocalClaimWithName(config *localTestConfig, name string) *v1.PersistentVolumeClaim {
	claim := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: config.ns,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			StorageClassName: &config.scName,
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(testRequestSize),
				},
			},
		},
	}

	return &claim
}

// waitForLocalPersistentVolume waits a local persistent volume with 'volumePath' to be available.
func waitForLocalPersistentVolume(c clientset.Interface, volumePath string) (*v1.PersistentVolume, error) {
	var pv *v1.PersistentVolume

	for start := time.Now(); time.Since(start) < 10*time.Minute && pv == nil; time.Sleep(5 * time.Second) {
		pvs, err := c.CoreV1().PersistentVolumes().List(metav1.ListOptions{})
		if err != nil {
			return nil, err
		}
		if len(pvs.Items) == 0 {
			continue
		}
		for _, p := range pvs.Items {
			if p.Spec.PersistentVolumeSource.Local == nil || p.Spec.PersistentVolumeSource.Local.Path != volumePath {
				continue
			}
			if p.Status.Phase != v1.VolumeAvailable {
				continue
			}
			pv = &p
			break
		}
	}
	if pv == nil {
		return nil, fmt.Errorf("Timeout while waiting for local persistent volume with path %v to be available", volumePath)
	}
	return pv, nil
}

// findLocalPersistentVolume finds persistent volume with 'spec.local.path' equals 'volumePath'.
func findLocalPersistentVolume(c clientset.Interface, volumePath string) (*v1.PersistentVolume, error) {
	pvs, err := c.CoreV1().PersistentVolumes().List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	for _, p := range pvs.Items {
		if p.Spec.PersistentVolumeSource.Local != nil && p.Spec.PersistentVolumeSource.Local.Path == volumePath {
			return &p, nil
		}
	}
	return nil, fmt.Errorf("Unable to find local persistent volume with path %v", volumePath)
}

func createStatefulSet(config *localTestConfig, ssReplicas int32, volumeCount int) *appsv1.StatefulSet {
	mounts := []v1.VolumeMount{}
	claims := []v1.PersistentVolumeClaim{}
	for i := 0; i < volumeCount; i++ {
		name := fmt.Sprintf("vol%v", i+1)
		pvc := newLocalClaimWithName(config, name)
		mounts = append(mounts, v1.VolumeMount{Name: name, MountPath: "/" + name})
		claims = append(claims, *pvc)
	}

	affinity := v1.Affinity{
		PodAntiAffinity: &v1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "app",
								Operator: metav1.LabelSelectorOpIn,
								Values:   []string{"local-volume-test"},
							},
						},
					},
					TopologyKey: "kubernetes.io/hostname",
				},
			},
		},
	}

	labels := map[string]string{"app": "local-volume-test"}
	spec := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "local-volume-statefulset",
			Namespace: config.ns,
		},
		Spec: appsv1.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "local-volume-test"},
			},
			Replicas: &ssReplicas,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:         "nginx",
							Image:        imageutils.GetE2EImage(imageutils.NginxSlim),
							VolumeMounts: mounts,
						},
					},
					Affinity: &affinity,
				},
			},
			VolumeClaimTemplates: claims,
			ServiceName:          "test-service",
		},
	}

	ss, err := config.client.AppsV1().StatefulSets(config.ns).Create(spec)
	Expect(err).NotTo(HaveOccurred())

	config.ssTester.WaitForRunningAndReady(ssReplicas, ss)
	return ss
}

func validateStatefulSet(config *localTestConfig, ss *appsv1.StatefulSet) {
	pods := config.ssTester.GetPodList(ss)

	// Verify that each pod is on a different node
	nodes := sets.NewString()
	for _, pod := range pods.Items {
		nodes.Insert(pod.Spec.NodeName)
	}

	Expect(nodes.Len()).To(Equal(len(pods.Items)))

	// TODO: validate all PVCs are bound
}

// SkipUnlessLocalSSDExists takes in an ssdInterface (scsi/nvme) and a filesystemType (fs/block)
// and skips if a disk of that type does not exist on the node
func SkipUnlessLocalSSDExists(ssdInterface, filesystemType string, node *v1.Node) {
	ssdCmd := fmt.Sprintf("ls -1 /mnt/disks/by-uuid/google-local-ssds-%s-%s/ | wc -l", ssdInterface, filesystemType)
	res, err := framework.IssueSSHCommandWithResult(ssdCmd, framework.TestContext.Provider, node)
	Expect(err).NotTo(HaveOccurred())
	num, err := strconv.Atoi(strings.TrimSpace(res.Stdout))
	Expect(err).NotTo(HaveOccurred())
	if num < 1 {
		framework.Skipf("Requires at least 1 %s %s localSSD ", ssdInterface, filesystemType)
	}
}
