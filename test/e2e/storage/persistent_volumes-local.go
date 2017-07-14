/*
Copyright 2017 The Kubernetes Authors.

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
	"path/filepath"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
)

type localTestConfig struct {
	ns     string
	nodes  *v1.NodeList
	client clientset.Interface
}

type localTestVolume struct {
	// Node that the volume is on
	node *v1.Node
	// Path to the volume on the host node
	hostDir string
	// Path to the volume in the local util container
	containerDir string
	// PVC for this volume
	pvc *v1.PersistentVolumeClaim
	// PV for this volume
	pv *v1.PersistentVolume
}

const (
	// TODO: This may not be available/writable on all images.
	hostBase      = "/tmp"
	containerBase = "/myvol"
	// Path to the first volume in the test containers
	// created via createLocalPod or makeLocalPod
	// leveraging pv_util.MakePod
	volumeDir = "/mnt/volume1"
	// testFile created in setupLocalVolume
	testFile = "test-file"
	// testFileContent writtent into testFile
	testFileContent = "test-file-content"
	testSC          = "local-test-storageclass"
)

var _ = SIGDescribe("PersistentVolumes-local [Feature:LocalPersistentVolumes] [Serial]", func() {
	f := framework.NewDefaultFramework("persistent-local-volumes-test")

	var (
		config *localTestConfig
		node0  *v1.Node
	)

	BeforeEach(func() {

		// Get all the schedulable nodes
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodes.Items)).NotTo(BeZero(), "No available nodes for scheduling")

		config = &localTestConfig{
			ns:     f.Namespace.Name,
			client: f.ClientSet,
			nodes:  nodes,
		}

		// Choose the first node
		node0 = &config.nodes.Items[0]
	})

	Context("when one pod requests one prebound PVC", func() {

		var testVol *localTestVolume

		BeforeEach(func() {
			testVol = setupLocalVolumePVCPV(config, node0)
		})

		AfterEach(func() {
			cleanupLocalVolume(config, testVol)
		})

		It("should be able to mount and read from the volume using one-command containers", func() {
			By("Creating a pod to read from the PV")
			//testFileContent was written during setupLocalVolume
			_, readCmd := createWriteAndReadCmds(volumeDir, testFile, "" /*writeTestFileContent*/)
			podSpec := makeLocalPod(config, testVol, readCmd)
			f.TestContainerOutput("pod reads PV", podSpec, 0, []string{testFileContent})
		})

		It("should be able to mount and write to the volume using one-command containers", func() {
			By("Creating a pod to write to the PV")
			writeCmd, readCmd := createWriteAndReadCmds(volumeDir, testFile, testVol.hostDir /*writeTestFileContent*/)
			writeThenReadCmd := fmt.Sprintf("%s;%s", writeCmd, readCmd)
			podSpec := makeLocalPod(config, testVol, writeThenReadCmd)
			f.TestContainerOutput("pod writes to PV", podSpec, 0, []string{testVol.hostDir})
		})

		It("should be able to mount volume and read from pod1", func() {
			By("Creating pod1")
			pod1, pod1Err := createLocalPod(config, testVol)
			Expect(pod1Err).NotTo(HaveOccurred())

			pod1NodeName, pod1NodeNameErr := podNodeName(config, pod1)
			Expect(pod1NodeNameErr).NotTo(HaveOccurred())
			framework.Logf("pod1 %q created on Node %q", pod1.Name, pod1NodeName)
			Expect(pod1NodeName).To(Equal(node0.Name))

			By("Reading in pod1")
			//testFileContent was written during setupLocalVolume
			_, readCmd := createWriteAndReadCmds(volumeDir, testFile, "" /*writeTestFileContent*/)
			readOut := podRWCmdExec(pod1, readCmd)
			Expect(readOut).To(ContainSubstring(testFileContent)) /*aka writeTestFileContents*/

			By("Deleting pod1")
			framework.DeletePodOrFail(config.client, config.ns, pod1.Name)
		})

		It("should be able to mount volume and write from pod1", func() {
			By("Creating pod1")
			pod1, pod1Err := createLocalPod(config, testVol)
			Expect(pod1Err).NotTo(HaveOccurred())

			pod1NodeName, pod1NodeNameErr := podNodeName(config, pod1)
			Expect(pod1NodeNameErr).NotTo(HaveOccurred())
			framework.Logf("pod1 %q created on Node %q", pod1.Name, pod1NodeName)
			Expect(pod1NodeName).To(Equal(node0.Name))

			By("Writing in pod1")
			writeCmd, _ := createWriteAndReadCmds(volumeDir, testFile, testVol.hostDir /*writeTestFileContent*/)
			podRWCmdExec(pod1, writeCmd)

			By("Deleting pod1")
			framework.DeletePodOrFail(config.client, config.ns, pod1.Name)
		})
	})

	Context("when two pods request one prebound PVC one after other", func() {

		var testVol *localTestVolume

		BeforeEach(func() {
			testVol = setupLocalVolumePVCPV(config, node0)
		})

		AfterEach(func() {
			cleanupLocalVolume(config, testVol)
		})

		It("should be able to mount volume, write from pod1, and read from pod2 using one-command containers", func() {
			By("Creating pod1 to write to the PV")
			writeCmd, readCmd := createWriteAndReadCmds(volumeDir, testFile, testVol.hostDir /*writeTestFileContent*/)
			writeThenReadCmd := fmt.Sprintf("%s;%s", writeCmd, readCmd)
			podSpec1 := makeLocalPod(config, testVol, writeThenReadCmd)
			f.TestContainerOutput("pod writes to PV", podSpec1, 0, []string{testVol.hostDir})

			By("Creating pod2 to read from the PV")
			podSpec2 := makeLocalPod(config, testVol, readCmd)
			f.TestContainerOutput("pod reads PV", podSpec2, 0, []string{testVol.hostDir})
		})

		It("should be able to mount volume in two pods one after other, write from pod1, and read from pod2", func() {
			By("Creating pod1")
			pod1, pod1Err := createLocalPod(config, testVol)
			Expect(pod1Err).NotTo(HaveOccurred())

			framework.ExpectNoError(framework.WaitForPodRunningInNamespace(config.client, pod1))
			pod1NodeName, pod1NodeNameErr := podNodeName(config, pod1)
			Expect(pod1NodeNameErr).NotTo(HaveOccurred())
			framework.Logf("Pod1 %q created on Node %q", pod1.Name, pod1NodeName)
			Expect(pod1NodeName).To(Equal(node0.Name))

			writeCmd, readCmd := createWriteAndReadCmds(volumeDir, testFile, testVol.hostDir /*writeTestFileContent*/)

			By("Writing in pod1")
			podRWCmdExec(pod1, writeCmd)

			By("Deleting pod1")
			framework.DeletePodOrFail(config.client, config.ns, pod1.Name)

			By("Creating pod2")
			pod2, pod2Err := createLocalPod(config, testVol)
			Expect(pod2Err).NotTo(HaveOccurred())

			framework.ExpectNoError(framework.WaitForPodRunningInNamespace(config.client, pod2))
			pod2NodeName, pod2NodeNameErr := podNodeName(config, pod2)
			Expect(pod2NodeNameErr).NotTo(HaveOccurred())
			framework.Logf("Pod2 %q created on Node %q", pod2.Name, pod2NodeName)
			Expect(pod2NodeName).To(Equal(node0.Name))

			By("Reading in pod2")
			readOut := podRWCmdExec(pod2, readCmd)
			Expect(readOut).To(ContainSubstring(testVol.hostDir)) /*aka writeTestFileContents*/

			By("Deleting pod2")
			framework.DeletePodOrFail(config.client, config.ns, pod2.Name)
		})
	})

	Context("when two pods request one prebound PVC at the same time", func() {

		var testVol *localTestVolume

		BeforeEach(func() {
			testVol = setupLocalVolumePVCPV(config, node0)
		})

		AfterEach(func() {
			cleanupLocalVolume(config, testVol)
		})

		It("should be able to mount volume in two pods at the same time, write from pod1, and read from pod2", func() {
			By("Creating pod1 to write to the PV")
			pod1, pod1Err := createLocalPod(config, testVol)
			Expect(pod1Err).NotTo(HaveOccurred())

			framework.ExpectNoError(framework.WaitForPodRunningInNamespace(config.client, pod1))
			pod1NodeName, pod1NodeNameErr := podNodeName(config, pod1)
			Expect(pod1NodeNameErr).NotTo(HaveOccurred())
			framework.Logf("Pod1 %q created on Node %q", pod1.Name, pod1NodeName)
			Expect(pod1NodeName).To(Equal(node0.Name))

			By("Creating pod2 to read from the PV")
			pod2, pod2Err := createLocalPod(config, testVol)
			Expect(pod2Err).NotTo(HaveOccurred())

			framework.ExpectNoError(framework.WaitForPodRunningInNamespace(config.client, pod2))
			pod2NodeName, pod2NodeNameErr := podNodeName(config, pod2)
			Expect(pod2NodeNameErr).NotTo(HaveOccurred())
			framework.Logf("Pod2 %q created on Node %q", pod2.Name, pod2NodeName)
			Expect(pod2NodeName).To(Equal(node0.Name))

			writeCmd, readCmd := createWriteAndReadCmds(volumeDir, testFile, testVol.hostDir /*writeTestFileContent*/)

			By("Writing in pod1")
			podRWCmdExec(pod1, writeCmd)
			By("Reading in pod2")
			readOut := podRWCmdExec(pod2, readCmd)

			Expect(readOut).To(ContainSubstring(testVol.hostDir)) /*aka writeTestFileContents*/

			By("Deleting pod1")
			framework.DeletePodOrFail(config.client, config.ns, pod1.Name)
			By("Deleting pod2")
			framework.DeletePodOrFail(config.client, config.ns, pod2.Name)
		})
	})
})

// podNode wraps RunKubectl to get node where pod is running
func podNodeName(config *localTestConfig, pod *v1.Pod) (string, error) {
	runtimePod, runtimePodErr := config.client.Core().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	return runtimePod.Spec.NodeName, runtimePodErr
}

// Launches a pod with hostpath volume on a specific node to setup a directory to use
// for the local PV
func setupLocalVolume(config *localTestConfig, node *v1.Node) *localTestVolume {
	testDirName := "local-volume-test-" + string(uuid.NewUUID())
	testDir := filepath.Join(containerBase, testDirName)
	hostDir := filepath.Join(hostBase, testDirName)
	//populate volume with testFile containing testFileContent
	writeCmd, _ := createWriteAndReadCmds(testDir, testFile, testFileContent)
	By(fmt.Sprintf("Creating local volume on node %q at path %q", node.Name, hostDir))

	runLocalUtil(config, node.Name, writeCmd)
	return &localTestVolume{
		node:         node,
		hostDir:      hostDir,
		containerDir: testDir,
	}
}

// Deletes the PVC/PV, and launches a pod with hostpath volume to remove the test directory
func cleanupLocalVolume(config *localTestConfig, volume *localTestVolume) {
	if volume == nil {
		return
	}

	By("Cleaning up PVC and PV")
	errs := framework.PVPVCCleanup(config.client, config.ns, volume.pv, volume.pvc)
	if len(errs) > 0 {
		framework.Failf("Failed to delete PV and/or PVC: %v", utilerrors.NewAggregate(errs))
	}

	By("Removing the test directory")
	removeCmd := fmt.Sprintf("rm -r %s", volume.containerDir)
	runLocalUtil(config, volume.node.Name, removeCmd)
}

func runLocalUtil(config *localTestConfig, nodeName, cmd string) {
	framework.StartVolumeServer(config.client, framework.VolumeTestConfig{
		Namespace:   config.ns,
		Prefix:      "local-volume-init",
		ServerImage: framework.BusyBoxImage,
		ServerCmds:  []string{"/bin/sh"},
		ServerArgs:  []string{"-c", cmd},
		ServerVolumes: map[string]string{
			hostBase: containerBase,
		},
		WaitForCompletion: true,
		NodeName:          nodeName,
	})
}

func makeLocalPVCConfig() framework.PersistentVolumeClaimConfig {
	sc := testSC
	return framework.PersistentVolumeClaimConfig{
		AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		StorageClassName: &sc,
	}
}

func makeLocalPVConfig(volume *localTestVolume) framework.PersistentVolumeConfig {
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
		StorageClassName: testSC,
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
func createLocalPVCPV(config *localTestConfig, volume *localTestVolume) {
	pvcConfig := makeLocalPVCConfig()
	pvConfig := makeLocalPVConfig(volume)

	var err error
	volume.pv, volume.pvc, err = framework.CreatePVPVC(config.client, pvConfig, pvcConfig, config.ns, true)
	framework.ExpectNoError(err)
	framework.ExpectNoError(framework.WaitOnPVandPVC(config.client, config.ns, volume.pv, volume.pvc))
}

func makeLocalPod(config *localTestConfig, volume *localTestVolume, cmd string) *v1.Pod {
	return framework.MakePod(config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, cmd)
}

func createLocalPod(config *localTestConfig, volume *localTestVolume) (*v1.Pod, error) {
	return framework.CreatePod(config.client, config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, "")
}

// Create corresponding write and read commands
// to be executed inside containers with local PV attached
func createWriteAndReadCmds(testFileDir string, testFile string, writeTestFileContent string) (writeCmd string, readCmd string) {
	testFilePath := filepath.Join(testFileDir, testFile)
	writeCmd = fmt.Sprintf("mkdir -p %s; echo %s > %s", testFileDir, writeTestFileContent, testFilePath)
	readCmd = fmt.Sprintf("cat %s", testFilePath)
	return writeCmd, readCmd
}

// Execute a read or write command in a pod.
// Fail on error
func podRWCmdExec(pod *v1.Pod, cmd string) string {
	out, err := podExec(pod, cmd)
	Expect(err).NotTo(HaveOccurred())
	return out
}

// Initialize test volume on node
// and create local PVC and PV
func setupLocalVolumePVCPV(config *localTestConfig, node *v1.Node) *localTestVolume {
	By("Initializing test volume")
	testVol := setupLocalVolume(config, node)

	By("Creating local PVC and PV")
	createLocalPVCPV(config, testVol)

	return testVol
}
