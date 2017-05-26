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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
)

type localTestConfig struct {
	ns     string
	nodes  []v1.Node
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
	testFile      = "test-file"
	testContents  = "testdata"
	testSC        = "local-test-storagclass"
)

var _ = framework.KubeDescribe("[Volume] PersistentVolumes-local [Feature:LocalPersistentVolumes] [Serial]", func() {
	f := framework.NewDefaultFramework("persistent-local-volumes-test")

	var (
		config *localTestConfig
	)

	BeforeEach(func() {
		config = &localTestConfig{
			ns:     f.Namespace.Name,
			client: f.ClientSet,
			nodes:  []v1.Node{},
		}

		// Get all the schedulable nodes
		nodes, err := config.client.CoreV1().Nodes().List(metav1.ListOptions{})
		if err != nil {
			framework.Failf("Failed to get nodes: %v", err)
		}

		for _, node := range nodes.Items {
			if !node.Spec.Unschedulable {
				// TODO: does this need to be a deep copy
				config.nodes = append(config.nodes, node)
			}
		}
		if len(config.nodes) == 0 {
			framework.Failf("No available nodes for scheduling")
		}
	})

	Context("when one pod requests one prebound PVC", func() {
		var (
			testVol *localTestVolume
			node    *v1.Node
		)

		BeforeEach(func() {
			// Choose the first node
			node = &config.nodes[0]
		})

		AfterEach(func() {
			cleanupLocalVolume(config, testVol)
			testVol = nil
		})

		It("should be able to mount and read from the volume", func() {
			By("Initializing test volume")
			testVol = setupLocalVolume(config, node)

			By("Creating local PVC and PV")
			createLocalPVCPV(config, testVol)

			By("Creating a pod to consume the PV")
			readCmd := fmt.Sprintf("cat /mnt/volume1/%s", testFile)
			podSpec := createLocalPod(config, testVol, readCmd)
			f.TestContainerOutput("pod consumes PV", podSpec, 0, []string{testContents})
		})

		It("should be able to mount and write to the volume", func() {
			By("Initializing test volume")
			testVol = setupLocalVolume(config, node)

			By("Creating local PVC and PV")
			createLocalPVCPV(config, testVol)

			By("Creating a pod to write to the PV")
			testFilePath := filepath.Join("/mnt/volume1", testFile)
			cmd := fmt.Sprintf("echo %s > %s; cat %s", testVol.hostDir, testFilePath, testFilePath)
			podSpec := createLocalPod(config, testVol, cmd)
			f.TestContainerOutput("pod writes to PV", podSpec, 0, []string{testVol.hostDir})
		})
	})
})

// Launches a pod with hostpath volume on a specific node to setup a directory to use
// for the local PV
func setupLocalVolume(config *localTestConfig, node *v1.Node) *localTestVolume {
	testDirName := "local-volume-test-" + string(uuid.NewUUID())
	testDir := filepath.Join(containerBase, testDirName)
	hostDir := filepath.Join(hostBase, testDirName)
	testFilePath := filepath.Join(testDir, testFile)
	writeCmd := fmt.Sprintf("mkdir %s; echo %s > %s", testDir, testContents, testFilePath)
	framework.Logf("Creating local volume on node %q at path %q", node.Name, hostDir)

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
		framework.Logf("AfterEach: Failed to delete PV and/or PVC: %v", utilerrors.NewAggregate(errs))
	}

	By("Removing the test directory")
	removeCmd := fmt.Sprintf("rm -r %s", volume.containerDir)
	runLocalUtil(config, volume.node.Name, removeCmd)
}

func runLocalUtil(config *localTestConfig, nodeName, cmd string) {
	framework.StartVolumeServer(config.client, framework.VolumeTestConfig{
		Namespace:   config.ns,
		Prefix:      "local-volume-init",
		ServerImage: "gcr.io/google_containers/busybox:1.24",
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
	framework.WaitOnPVandPVC(config.client, config.ns, volume.pv, volume.pvc)
}

func createLocalPod(config *localTestConfig, volume *localTestVolume, cmd string) *v1.Pod {
	return framework.MakePod(config.ns, []*v1.PersistentVolumeClaim{volume.pvc}, false, cmd)
}
