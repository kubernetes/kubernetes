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
	"encoding/json"
	"fmt"
	"path"
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
)

type localTestConfig struct {
	ns     string
	nodes  *v1.NodeList
	node0  *v1.Node
	client clientset.Interface
	scName string
}

type LocalVolumeType string

const (
	// default local volume type, aka a directory
	DirectoryLocalVolumeType LocalVolumeType = "dir"
	// creates a tmpfs and mounts it
	TmpfsLocalVolumeType LocalVolumeType = "tmpfs"
)

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
	localVolumeType LocalVolumeType
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
	testServiceAccount = "local-storage-bootstrapper"
	// testRoleBinding is the cluster-admin rolebinding for bootstrapper
	testRoleBinding = "local-storage:bootstrapper"
	// volumeConfigName is the configmap passed to bootstrapper and provisioner
	volumeConfigName = "local-volume-config"
	// bootstrapper and provisioner images used for e2e tests
	bootstrapperImageName = "quay.io/external_storage/local-volume-provisioner-bootstrap:v1.0.1"
	provisionerImageName  = "quay.io/external_storage/local-volume-provisioner:v1.0.1"
	// provisioner daemonSetName name, must match the one defined in bootstrapper
	daemonSetName = "local-volume-provisioner"
	// provisioner node/pv cluster role binding, must match the one defined in bootstrapper
	nodeBindingName = "local-storage:provisioner-node-binding"
	pvBindingName   = "local-storage:provisioner-pv-binding"
	// A sample request size
	testRequestSize = "10Mi"
)

// Common selinux labels
var selinuxLabel = &v1.SELinuxOptions{
	Level: "s0:c0,c1"}

var _ = SIGDescribe("PersistentVolumes-local [Feature:LocalPersistentVolumes] [Serial]", func() {
	f := framework.NewDefaultFramework("persistent-local-volumes-test")

	var (
		config *localTestConfig
		scName string
	)

	BeforeEach(func() {
		// Get all the schedulable nodes
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodes.Items)).NotTo(BeZero(), "No available nodes for scheduling")
		scName = fmt.Sprintf("%v-%v", testSCPrefix, f.Namespace.Name)
		// Choose the first node
		node0 := &nodes.Items[0]

		config = &localTestConfig{
			ns:     f.Namespace.Name,
			client: f.ClientSet,
			nodes:  nodes,
			node0:  node0,
			scName: scName,
		}
	})

	Context("when one pod requests one prebound PVC", func() {

		var testVol *localTestVolume

		BeforeEach(func() {
			testVol = setupLocalVolumePVCPV(config, DirectoryLocalVolumeType)
		})

		AfterEach(func() {
			cleanupLocalVolume(config, testVol)
		})

		It("should be able to mount volume and read from pod1", func() {
			By("Creating pod1")
			pod1, pod1Err := createLocalPod(config, testVol)
			Expect(pod1Err).NotTo(HaveOccurred())

			pod1NodeName, pod1NodeNameErr := podNodeName(config, pod1)
			Expect(pod1NodeNameErr).NotTo(HaveOccurred())
			framework.Logf("pod1 %q created on Node %q", pod1.Name, pod1NodeName)
			Expect(pod1NodeName).To(Equal(config.node0.Name))

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

			pod1NodeName, pod1NodeNameErr := podNodeName(config, pod1)
			Expect(pod1NodeNameErr).NotTo(HaveOccurred())
			framework.Logf("pod1 %q created on Node %q", pod1.Name, pod1NodeName)
			Expect(pod1NodeName).To(Equal(config.node0.Name))

			// testFileContent was written during setupLocalVolume
			testReadFileContent(volumeDir, testFile, testFileContent, pod1)

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
			testVol = setupLocalVolumePVCPV(config, DirectoryLocalVolumeType)
		})

		AfterEach(func() {
			cleanupLocalVolume(config, testVol)
		})
	})

	LocalVolumeTypes := []LocalVolumeType{DirectoryLocalVolumeType, TmpfsLocalVolumeType}

	Context("when two pods mount a local volume at the same time", func() {
		It("should be able to write from pod1 and read from pod2", func() {
			for _, testVolType := range LocalVolumeTypes {
				var testVol *localTestVolume
				By(fmt.Sprintf("local-volume-type: %s", testVolType))
				testVol = setupLocalVolumePVCPV(config, testVolType)
				twoPodsReadWriteTest(config, testVol)
				cleanupLocalVolume(config, testVol)
			}
		})
	})

	Context("when two pods mount a local volume one after the other", func() {
		It("should be able to write from pod1 and read from pod2", func() {
			for _, testVolType := range LocalVolumeTypes {
				var testVol *localTestVolume
				By(fmt.Sprintf("local-volume-type: %s", testVolType))
				testVol = setupLocalVolumePVCPV(config, testVolType)
				twoPodsReadWriteSerialTest(config, testVol)
				cleanupLocalVolume(config, testVol)
			}
		})
	})

	Context("when pod using local volume with non-existant path", func() {
		ep := &eventPatterns{
			reason:  "FailedMount",
			pattern: make([]string, 2)}
		ep.pattern = append(ep.pattern, "MountVolume.SetUp failed")
		ep.pattern = append(ep.pattern, "does not exist")

		It("should not be able to mount", func() {
			for _, testVolType := range LocalVolumeTypes {
				By(fmt.Sprintf("local-volume-type: %s", testVolType))
				testVol := &localTestVolume{
					node:            config.node0,
					hostDir:         "/non-existent/location/nowhere",
					localVolumeType: testVolType,
				}
				By("Creating local PVC and PV")
				createLocalPVCPV(config, testVol)
				pod, err := createLocalPod(config, testVol)
				Expect(err).To(HaveOccurred())
				checkPodEvents(config, pod.Name, ep)
			}
		})
	})

	Context("when pod's node is different from PV's NodeAffinity", func() {

		BeforeEach(func() {
			if len(config.nodes.Items) < 2 {
				framework.Skipf("Runs only when number of nodes >= 2")
			}
		})

		ep := &eventPatterns{
			reason:  "FailedScheduling",
			pattern: make([]string, 2)}
		ep.pattern = append(ep.pattern, "MatchNodeSelector")
		ep.pattern = append(ep.pattern, "NoVolumeNodeConflict")
		for _, testVolType := range LocalVolumeTypes {

			It("should not be able to mount due to different NodeAffinity", func() {

				testPodWithNodeName(config, testVolType, ep, config.nodes.Items[1].Name, makeLocalPodWithNodeAffinity)
			})

			It("should not be able to mount due to different NodeSelector", func() {

				testPodWithNodeName(config, testVolType, ep, config.nodes.Items[1].Name, makeLocalPodWithNodeSelector)
			})

		}
	})

	Context("when pod's node is different from PV's NodeName", func() {

		BeforeEach(func() {
			if len(config.nodes.Items) < 2 {
				framework.Skipf("Runs only when number of nodes >= 2")
			}
		})

		ep := &eventPatterns{
			reason:  "FailedMount",
			pattern: make([]string, 2)}
		ep.pattern = append(ep.pattern, "NodeSelectorTerm")
		ep.pattern = append(ep.pattern, "Storage node affinity check failed")
		for _, testVolType := range LocalVolumeTypes {

			It("should not be able to mount due to different NodeName", func() {

				testPodWithNodeName(config, testVolType, ep, config.nodes.Items[1].Name, makeLocalPodWithNodeName)
			})
		}
	})

	Context("when using local volume provisioner", func() {
		var volumePath string

		BeforeEach(func() {
			setupLocalVolumeProvisioner(config)
			volumePath = path.Join(
				hostBase, discoveryDir, fmt.Sprintf("vol-%v", string(uuid.NewUUID())))
		})

		AfterEach(func() {
			cleanupLocalVolumeProvisioner(config, volumePath)
		})

		It("should create and recreate local persistent volume", func() {
			By("Creating bootstrapper pod to start provisioner daemonset")
			createBootstrapperJob(config)
			kind := schema.GroupKind{Group: "extensions", Kind: "DaemonSet"}
			framework.WaitForControlledPodsRunning(config.client, config.ns, daemonSetName, kind)

			By("Creating a directory under discovery path")
			framework.Logf("creating local volume under path %q", volumePath)
			mkdirCmd := fmt.Sprintf("mkdir %v -m 777", volumePath)
			err := framework.IssueSSHCommand(mkdirCmd, framework.TestContext.Provider, config.node0)
			Expect(err).NotTo(HaveOccurred())
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
})

type makeLocalPodWith func(config *localTestConfig, volume *localTestVolume, nodeName string) *v1.Pod

func testPodWithNodeName(config *localTestConfig, testVolType LocalVolumeType, ep *eventPatterns, nodeName string, makeLocalPodFunc makeLocalPodWith) {
	var testVol *localTestVolume
	By(fmt.Sprintf("local-volume-type: %s", testVolType))
	testVol = setupLocalVolumePVCPV(config, testVolType)

	pod := makeLocalPodFunc(config, testVol, nodeName)
	pod, err := config.client.CoreV1().Pods(config.ns).Create(pod)
	Expect(err).NotTo(HaveOccurred())
	err = framework.WaitForPodRunningInNamespace(config.client, pod)
	Expect(err).To(HaveOccurred())
	checkPodEvents(config, pod.Name, ep)
	cleanupLocalVolume(config, testVol)
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

	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(config.client, pod1))
	pod1NodeName, pod1NodeNameErr := podNodeName(config, pod1)
	Expect(pod1NodeNameErr).NotTo(HaveOccurred())
	framework.Logf("Pod1 %q created on Node %q", pod1.Name, pod1NodeName)
	Expect(pod1NodeName).To(Equal(config.node0.Name))

	// testFileContent was written during setupLocalVolume
	testReadFileContent(volumeDir, testFile, testFileContent, pod1)

	By("Creating pod2 to read from the PV")
	pod2, pod2Err := createLocalPod(config, testVol)
	Expect(pod2Err).NotTo(HaveOccurred())

	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(config.client, pod2))
	pod2NodeName, pod2NodeNameErr := podNodeName(config, pod2)
	Expect(pod2NodeNameErr).NotTo(HaveOccurred())
	framework.Logf("Pod2 %q created on Node %q", pod2.Name, pod2NodeName)
	Expect(pod2NodeName).To(Equal(config.node0.Name))

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

	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(config.client, pod1))
	pod1NodeName, pod1NodeNameErr := podNodeName(config, pod1)
	Expect(pod1NodeNameErr).NotTo(HaveOccurred())
	framework.Logf("Pod1 %q created on Node %q", pod1.Name, pod1NodeName)
	Expect(pod1NodeName).To(Equal(config.node0.Name))

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

	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(config.client, pod2))
	pod2NodeName, pod2NodeNameErr := podNodeName(config, pod2)
	Expect(pod2NodeNameErr).NotTo(HaveOccurred())
	framework.Logf("Pod2 %q created on Node %q", pod2.Name, pod2NodeName)
	Expect(pod2NodeName).To(Equal(config.node0.Name))

	By("Reading in pod2")
	testReadFileContent(volumeDir, testFile, testVol.hostDir, pod2)

	By("Deleting pod2")
	framework.DeletePodOrFail(config.client, config.ns, pod2.Name)
}

// podNode wraps RunKubectl to get node where pod is running
func podNodeName(config *localTestConfig, pod *v1.Pod) (string, error) {
	runtimePod, runtimePodErr := config.client.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	return runtimePod.Spec.NodeName, runtimePodErr
}

// setupLocalVolume setups a directory to user for local PV
func setupLocalVolume(config *localTestConfig, localVolumeType LocalVolumeType) *localTestVolume {
	testDirName := "local-volume-test-" + string(uuid.NewUUID())
	hostDir := filepath.Join(hostBase, testDirName)

	if localVolumeType == TmpfsLocalVolumeType {
		createAndMountTmpfsLocalVolume(config, hostDir)
	}

	// populate volume with testFile containing testFileContent
	writeCmd, _ := createWriteAndReadCmds(hostDir, testFile, testFileContent)
	By(fmt.Sprintf("Creating local volume on node %q at path %q", config.node0.Name, hostDir))
	err := framework.IssueSSHCommand(writeCmd, framework.TestContext.Provider, config.node0)
	Expect(err).NotTo(HaveOccurred())
	return &localTestVolume{
		node:            config.node0,
		hostDir:         hostDir,
		localVolumeType: localVolumeType,
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

	if volume.localVolumeType == TmpfsLocalVolumeType {
		unmountTmpfsLocalVolume(config, volume.hostDir)
	}

	By("Removing the test directory")
	removeCmd := fmt.Sprintf("rm -r %s", volume.hostDir)
	err := framework.IssueSSHCommand(removeCmd, framework.TestContext.Provider, config.node0)
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
func createLocalPVCPV(config *localTestConfig, volume *localTestVolume) {
	pvcConfig := makeLocalPVCConfig(config)
	pvConfig := makeLocalPVConfig(config, volume)
	var err error
	volume.pv, volume.pvc, err = framework.CreatePVPVC(config.client, pvConfig, pvcConfig, config.ns, true)
	framework.ExpectNoError(err)
	framework.ExpectNoError(framework.WaitOnPVandPVC(config.client, config.ns, volume.pv, volume.pvc))
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

func createAndMountTmpfsLocalVolume(config *localTestConfig, dir string) {
	By(fmt.Sprintf("Creating tmpfs mount point on node %q at path %q", config.node0.Name, dir))
	err := framework.IssueSSHCommand(fmt.Sprintf("mkdir -p %q && sudo mount -t tmpfs -o size=1m tmpfs-%q %q", dir, dir, dir), framework.TestContext.Provider, config.node0)
	Expect(err).NotTo(HaveOccurred())
}

func unmountTmpfsLocalVolume(config *localTestConfig, dir string) {
	By(fmt.Sprintf("Unmount tmpfs mount point on node %q at path %q", config.node0.Name, dir))
	err := framework.IssueSSHCommand(fmt.Sprintf("sudo umount %q", dir), framework.TestContext.Provider, config.node0)
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
	out, err := podExec(pod, cmd)
	Expect(err).NotTo(HaveOccurred())
	return out
}

// Initialize test volume on node
// and create local PVC and PV
func setupLocalVolumePVCPV(config *localTestConfig, localVolumeType LocalVolumeType) *localTestVolume {
	By("Initializing test volume")
	testVol := setupLocalVolume(config, localVolumeType)

	By("Creating local PVC and PV")
	createLocalPVCPV(config, testVol)

	return testVol
}

func setupLocalVolumeProvisioner(config *localTestConfig) {
	By("Bootstrapping local volume provisioner")
	createServiceAccount(config)
	createClusterRoleBinding(config)
	createVolumeConfigMap(config)

	By("Initializing local volume discovery base path")
	mkdirCmd := fmt.Sprintf("mkdir %v -m 777", path.Join(hostBase, discoveryDir))
	err := framework.IssueSSHCommand(mkdirCmd, framework.TestContext.Provider, config.node0)
	Expect(err).NotTo(HaveOccurred())
}

func cleanupLocalVolumeProvisioner(config *localTestConfig, volumePath string) {
	By("Cleaning up cluster role binding")
	deleteClusterRoleBinding(config)

	By("Removing the test directory")
	removeCmd := fmt.Sprintf("rm -r %s", path.Join(hostBase, discoveryDir))
	err := framework.IssueSSHCommand(removeCmd, framework.TestContext.Provider, config.node0)
	Expect(err).NotTo(HaveOccurred())

	By("Cleaning up persistent volume")
	pv, err := findLocalPersistentVolume(config.client, volumePath)
	Expect(err).NotTo(HaveOccurred())
	err = config.client.CoreV1().PersistentVolumes().Delete(pv.Name, &metav1.DeleteOptions{})
	Expect(err).NotTo(HaveOccurred())
}

func createServiceAccount(config *localTestConfig) {
	serviceAccount := v1.ServiceAccount{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "ServiceAccount"},
		ObjectMeta: metav1.ObjectMeta{Name: testServiceAccount, Namespace: config.ns},
	}
	_, err := config.client.CoreV1().ServiceAccounts(config.ns).Create(&serviceAccount)
	Expect(err).NotTo(HaveOccurred())
}

func createClusterRoleBinding(config *localTestConfig) {
	subjects := []rbacv1beta1.Subject{
		{
			Kind:      rbacv1beta1.ServiceAccountKind,
			Name:      testServiceAccount,
			Namespace: config.ns,
		},
	}

	binding := rbacv1beta1.ClusterRoleBinding{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "rbac.authorization.k8s.io/v1beta1",
			Kind:       "ClusterRoleBinding",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: testRoleBinding,
		},
		RoleRef: rbacv1beta1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "cluster-admin",
		},
		Subjects: subjects,
	}

	_, err := config.client.RbacV1beta1().ClusterRoleBindings().Create(&binding)
	Expect(err).NotTo(HaveOccurred())
}

func deleteClusterRoleBinding(config *localTestConfig) {
	err := config.client.RbacV1beta1().ClusterRoleBindings().Delete(testRoleBinding, metav1.NewDeleteOptions(0))
	Expect(err).NotTo(HaveOccurred())
	// These role bindings are created in provisioner; we just ensure it's
	// deleted and do not panic on error.
	config.client.RbacV1beta1().ClusterRoleBindings().Delete(nodeBindingName, metav1.NewDeleteOptions(0))
	config.client.RbacV1beta1().ClusterRoleBindings().Delete(pvBindingName, metav1.NewDeleteOptions(0))
}

func createVolumeConfigMap(config *localTestConfig) {
	mountConfig := struct {
		HostDir string `json:"hostDir"`
	}{
		HostDir: path.Join(hostBase, discoveryDir),
	}
	data, err := json.Marshal(&mountConfig)
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
			config.scName: string(data),
		},
	}
	_, err = config.client.CoreV1().ConfigMaps(config.ns).Create(&configMap)
	Expect(err).NotTo(HaveOccurred())
}

func createBootstrapperJob(config *localTestConfig) {
	bootJob := &batchv1.Job{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Job",
			APIVersion: "batch/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "local-volume-tester-",
		},
		Spec: batchv1.JobSpec{
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					RestartPolicy:      v1.RestartPolicyNever,
					ServiceAccountName: testServiceAccount,
					Containers: []v1.Container{
						{
							Name:  "volume-tester",
							Image: bootstrapperImageName,
							Env: []v1.EnvVar{
								{
									Name: "MY_NAMESPACE",
									ValueFrom: &v1.EnvVarSource{
										FieldRef: &v1.ObjectFieldSelector{
											FieldPath: "metadata.namespace",
										},
									},
								},
							},
							Args: []string{
								fmt.Sprintf("--image=%v", provisionerImageName),
								fmt.Sprintf("--volume-config=%v", volumeConfigName),
							},
						},
					},
				},
			},
		},
	}
	job, err := config.client.Batch().Jobs(config.ns).Create(bootJob)
	Expect(err).NotTo(HaveOccurred())
	err = framework.WaitForJobFinish(config.client, config.ns, job.Name, 1)
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
