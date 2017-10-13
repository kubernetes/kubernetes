/*
Copyright 2015 The Kubernetes Authors.

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
	mathrand "math/rand"
	"strings"
	"time"

	"google.golang.org/api/googleapi"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	gcePDDetachTimeout  = 10 * time.Minute
	gcePDDetachPollTime = 10 * time.Second
	nodeStatusTimeout   = 10 * time.Minute
	nodeStatusPollTime  = 1 * time.Second
	maxReadRetry        = 3
	minNodes            = 2
)

var _ = SIGDescribe("Pod Disks", func() {
	var (
		podClient  v1core.PodInterface
		nodeClient v1core.NodeInterface
		host0Name  types.NodeName
		host1Name  types.NodeName
		nodes      *v1.NodeList
	)
	f := framework.NewDefaultFramework("pod-disks")

	BeforeEach(func() {
		framework.SkipUnlessNodeCountIsAtLeast(minNodes)

		podClient = f.ClientSet.Core().Pods(f.Namespace.Name)
		nodeClient = f.ClientSet.Core().Nodes()
		nodes = framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodes.Items)).To(BeNumerically(">=", minNodes), fmt.Sprintf("Requires at least %d nodes", minNodes))
		host0Name = types.NodeName(nodes.Items[0].ObjectMeta.Name)
		host1Name = types.NodeName(nodes.Items[1].ObjectMeta.Name)

		mathrand.Seed(time.Now().UTC().UnixNano())
	})

	Context("schedule a pod w/ a RW PD, delete pod, schedule it on another host, verify PD contents [Slow]", func() {
		type testT struct {
			descr     string                // It description
			deleteOpt *metav1.DeleteOptions // pod delete option
		}
		tests := []testT{
			{
				descr:     "immediate (0)",
				deleteOpt: metav1.NewDeleteOptions(0),
			},
			{
				descr:     "the default (30s)",
				deleteOpt: &metav1.DeleteOptions{}, // default per provider
			},
		}

		for _, t := range tests {
			It(fmt.Sprintf("when pod delete grace period is %s", t.descr), func() {
				framework.SkipUnlessProviderIs("gce", "gke", "aws")

				By("creating PD")
				diskName, err := framework.CreatePDWithRetry()
				framework.ExpectNoError(err, "Error creating PD")

				By("creating host0Pod on node0")
				host0Pod := testPDPod([]string{diskName}, host0Name, false /* readOnly */, 1 /* numContainers */)
				host1Pod := testPDPod([]string{diskName}, host1Name, false /* readOnly */, 1 /* numContainers */)

				podDelOpt := t.deleteOpt
				defer func() {
					// Teardown should do nothing unless test failed
					By("defer: cleaning up PD-RW test environment")
					framework.Logf("defer cleanup errors can usually be ignored")
					podClient.Delete(host0Pod.Name, podDelOpt)
					podClient.Delete(host1Pod.Name, podDelOpt)
					detachAndDeletePDs(diskName, []types.NodeName{host0Name, host1Name})
				}()

				_, err = podClient.Create(host0Pod)
				framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))
				framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))
				framework.Logf(fmt.Sprintf("host0Pod: %q, node0: %q", host0Pod.Name, host0Name))

				By("writing content to host0Pod on node0")
				containerName := "mycontainer"
				testFile := "/testpd1/tracker"
				testFileContents := fmt.Sprintf("%v", mathrand.Int())
				framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFile, testFileContents))
				framework.Logf(fmt.Sprintf("wrote %q to file %q in pod %q on node %q", testFileContents, testFile, host0Pod.Name, host0Name))

				By("verifying PD is present in node0's VolumeInUse list")
				framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, true /* shouldExist */))

				By("deleting host0Pod")
				framework.ExpectNoError(podClient.Delete(host0Pod.Name, podDelOpt), "Failed to delete host0Pod")
				framework.Logf(fmt.Sprintf("deleted host0Pod %q", host0Pod.Name))

				By("creating host1Pod on node1")
				_, err = podClient.Create(host1Pod)
				framework.ExpectNoError(err, "Failed to create host1Pod")
				framework.ExpectNoError(f.WaitForPodRunningSlow(host1Pod.Name))
				framework.Logf(fmt.Sprintf("host1Pod: %q, node1: %q", host1Pod.Name, host1Name))

				By("verifying PD contents in host1Pod")
				verifyPDContentsViaContainer(f, host1Pod.Name, containerName, map[string]string{testFile: testFileContents})
				framework.Logf(fmt.Sprintf("verified PD contents in pod %q", host1Pod.Name))

				By("verifying PD is removed from node1")
				framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, false /* shouldExist */))
				framework.Logf(fmt.Sprintf("PD %q removed from node %q's VolumeInUse list", diskName, host1Pod.Name))

				By("deleting host1Pod")
				framework.ExpectNoError(podClient.Delete(host1Pod.Name, podDelOpt), "Failed to delete host1Pod")
				framework.Logf(fmt.Sprintf("deleted host1Pod %q", host1Pod.Name))

				By("Test completed successfully, waiting for PD to detach from both nodes")
				waitForPDDetach(diskName, host0Name)
				waitForPDDetach(diskName, host1Name)
			})
		}
	})

	Context("schedule a pod w/ a readonly PD on two hosts, then delete both pods. [Slow]", func() {
		type testT struct {
			descr     string                // It description
			deleteOpt *metav1.DeleteOptions // pod delete option
		}
		tests := []testT{
			{
				descr:     "immediate (0)",
				deleteOpt: metav1.NewDeleteOptions(0),
			},
			{
				descr:     "the default (30s)",
				deleteOpt: &metav1.DeleteOptions{}, // default per provider
			},
		}

		for _, t := range tests {
			It(fmt.Sprintf("when pod delete grace period is %s", t.descr), func() {
				framework.SkipUnlessProviderIs("gce", "gke")

				By("creating PD")
				diskName, err := framework.CreatePDWithRetry()
				framework.ExpectNoError(err, "Error creating PD")

				rwPod := testPDPod([]string{diskName}, host0Name, false /* readOnly */, 1 /* numContainers */)
				host0ROPod := testPDPod([]string{diskName}, host0Name, true /* readOnly */, 1 /* numContainers */)
				host1ROPod := testPDPod([]string{diskName}, host1Name, true /* readOnly */, 1 /* numContainers */)

				podDelOpt := t.deleteOpt
				defer func() {
					// Teardown should do nothing unless test failed.
					By("defer: cleaning up PD-RO test environment")
					framework.Logf("defer cleanup errors can usually be ignored")
					podClient.Delete(rwPod.Name, podDelOpt)
					podClient.Delete(host0ROPod.Name, podDelOpt)
					podClient.Delete(host1ROPod.Name, podDelOpt)
					detachAndDeletePDs(diskName, []types.NodeName{host0Name, host1Name})
				}()

				By("creating rwPod to ensure PD is formatted")
				_, err = podClient.Create(rwPod)
				framework.ExpectNoError(err, "Failed to create rwPod")
				framework.ExpectNoError(f.WaitForPodRunningSlow(rwPod.Name))

				By("deleting the rwPod")
				framework.ExpectNoError(podClient.Delete(rwPod.Name, metav1.NewDeleteOptions(0)), "Failed to delete rwPod")
				framework.Logf(fmt.Sprintf("deleted rwPod %q", rwPod.Name))

				By("waiting for PD to detach")
				framework.ExpectNoError(waitForPDDetach(diskName, host0Name))

				By("creating host0ROPod on node0")
				_, err = podClient.Create(host0ROPod)
				framework.ExpectNoError(err, "Failed to create host0ROPod")
				By("creating host1ROPod on node1")
				_, err = podClient.Create(host1ROPod)
				framework.ExpectNoError(err, "Failed to create host1ROPod")
				framework.ExpectNoError(f.WaitForPodRunningSlow(host0ROPod.Name))
				framework.ExpectNoError(f.WaitForPodRunningSlow(host1ROPod.Name))

				By("deleting host0ROPod")
				framework.ExpectNoError(podClient.Delete(host0ROPod.Name, podDelOpt), "Failed to delete host0ROPod")
				framework.Logf(fmt.Sprintf("deleted host0ROPod %q", host0ROPod.Name))
				By("deleting host1ROPod")
				framework.ExpectNoError(podClient.Delete(host1ROPod.Name, podDelOpt), "Failed to delete host1ROPod")
				framework.Logf(fmt.Sprintf("deleted host1ROPod %q", host1ROPod.Name))

				By("Test completed successfully, waiting for PD to detach from both nodes")
				waitForPDDetach(diskName, host0Name)
				waitForPDDetach(diskName, host1Name)
			})
		}
	})

	Context("schedule a pod w/ RW PD(s) mounted to 1 or more containers, write to PD, verify content, delete pod, and repeat in rapid succession [Slow]", func() {
		var diskNames []string
		type testT struct {
			numContainers int
			numPDs        int
		}
		tests := []testT{
			{
				numContainers: 4,
				numPDs:        1,
			},
			{
				numContainers: 1,
				numPDs:        2,
			},
		}

		for _, t := range tests {
			numPDs := t.numPDs
			numContainers := t.numContainers

			It(fmt.Sprintf("using %d containers and %d PDs", numContainers, numPDs), func() {
				framework.SkipUnlessProviderIs("gce", "gke", "aws")
				var host0Pod *v1.Pod
				var err error
				fileAndContentToVerify := make(map[string]string)

				By(fmt.Sprintf("creating %d PD(s)", numPDs))
				for i := 0; i < numPDs; i++ {
					name, err := framework.CreatePDWithRetry()
					framework.ExpectNoError(err, fmt.Sprintf("Error creating PD %d", i))
					diskNames = append(diskNames, name)
				}

				defer func() {
					// Teardown should do nothing unless test failed.
					By("defer: cleaning up PD-RW test environment")
					framework.Logf("defer cleanup errors can usually be ignored")
					if host0Pod != nil {
						podClient.Delete(host0Pod.Name, metav1.NewDeleteOptions(0))
					}
					for _, diskName := range diskNames {
						detachAndDeletePDs(diskName, []types.NodeName{host0Name})
					}
				}()

				for i := 0; i < 3; i++ { // "rapid" repeat loop
					framework.Logf("PD Read/Writer Iteration #%v", i)
					By(fmt.Sprintf("creating host0Pod with %d containers on node0", numContainers))
					host0Pod = testPDPod(diskNames, host0Name, false /* readOnly */, numContainers)
					_, err = podClient.Create(host0Pod)
					framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))
					framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))

					By(fmt.Sprintf("writing %d file(s) via a container", numPDs))
					containerName := "mycontainer"
					if numContainers > 1 {
						containerName = fmt.Sprintf("mycontainer%v", mathrand.Intn(numContainers)+1)
					}
					for x := 1; x <= numPDs; x++ {
						testFile := fmt.Sprintf("/testpd%d/tracker%d", x, i)
						testFileContents := fmt.Sprintf("%v", mathrand.Int())
						fileAndContentToVerify[testFile] = testFileContents
						framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFile, testFileContents))
						framework.Logf(fmt.Sprintf("wrote %q to file %q in pod %q (container %q) on node %q", testFileContents, testFile, host0Pod.Name, containerName, host0Name))
					}

					By("verifying PD contents via a container")
					if numContainers > 1 {
						containerName = fmt.Sprintf("mycontainer%v", mathrand.Intn(numContainers)+1)
					}
					verifyPDContentsViaContainer(f, host0Pod.Name, containerName, fileAndContentToVerify)

					By("deleting host0Pod")
					framework.ExpectNoError(podClient.Delete(host0Pod.Name, metav1.NewDeleteOptions(0)), "Failed to delete host0Pod")
				}
				By(fmt.Sprintf("Test completed successfully, waiting for %d PD(s) to detach from node0", numPDs))
				for _, diskName := range diskNames {
					waitForPDDetach(diskName, host0Name)
				}
			})
		}
	})

	It("should be able to detach from a node which was deleted [Slow] [Disruptive]", func() {
		framework.SkipUnlessProviderIs("gce")

		initialGroupSize, err := framework.GroupSize(framework.TestContext.CloudConfig.NodeInstanceGroup)
		framework.ExpectNoError(err, "Error getting group size")

		By("creating a pd")
		diskName, err := framework.CreatePDWithRetry()
		framework.ExpectNoError(err, "Error creating a pd")

		host0Pod := testPDPod([]string{diskName}, host0Name, false, 1)
		containerName := "mycontainer"

		defer func() {
			By("defer: cleaning up PD-RW test env")
			framework.Logf("defer cleanup errors can usually be ignored")
			podClient.Delete(host0Pod.Name, metav1.NewDeleteOptions(0))
			detachAndDeletePDs(diskName, []types.NodeName{host0Name})
			framework.WaitForNodeToBeReady(f.ClientSet, string(host0Name), nodeStatusTimeout)
			framework.WaitForAllNodesSchedulable(f.ClientSet, nodeStatusTimeout)
			nodes = framework.GetReadySchedulableNodesOrDie(f.ClientSet)
			Expect(len(nodes.Items)).To(Equal(initialGroupSize), "Requires node count to return to initial group size.")
		}()

		By("creating host0Pod on node0")
		_, err = podClient.Create(host0Pod)
		framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))
		framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))

		By("writing content to host0Pod")
		testFile := "/testpd1/tracker"
		testFileContents := fmt.Sprintf("%v", mathrand.Int())
		framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFile, testFileContents))
		framework.Logf(fmt.Sprintf("wrote %q to file %q in pod %q on node %q", testFileContents, testFile, host0Pod.Name, host0Name))

		By("verifying PD is present in node0's VolumeInUse list")
		framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, true /* should exist*/))

		By("getting gce instances")
		gceCloud, err := framework.GetGCECloud()
		framework.ExpectNoError(err, fmt.Sprintf("Unable to create gcloud client err=%v", err))
		output, err := gceCloud.ListInstanceNames(framework.TestContext.CloudConfig.ProjectID, framework.TestContext.CloudConfig.Zone)
		framework.ExpectNoError(err, fmt.Sprintf("Unable to get list of node instances err=%v output=%s", err, output))
		Expect(true, strings.Contains(string(output), string(host0Name)))

		By("deleting host0")
		resp, err := gceCloud.DeleteInstance(framework.TestContext.CloudConfig.ProjectID, framework.TestContext.CloudConfig.Zone, string(host0Name))
		framework.ExpectNoError(err, fmt.Sprintf("Failed to delete host0Pod: err=%v response=%#v", err, resp))
		output, err = gceCloud.ListInstanceNames(framework.TestContext.CloudConfig.ProjectID, framework.TestContext.CloudConfig.Zone)
		framework.ExpectNoError(err, fmt.Sprintf("Unable to get list of node instances err=%v output=%s", err, output))
		Expect(false, strings.Contains(string(output), string(host0Name)))

		By("waiting for pd to detach from host0")
		waitForPDDetach(diskName, host0Name)
		framework.ExpectNoError(framework.WaitForGroupSize(framework.TestContext.CloudConfig.NodeInstanceGroup, int32(initialGroupSize)), "Unable to get back the cluster to inital size")
	})

	It("should be able to detach from a node whose api object was deleted [Slow] [Disruptive]", func() {
		framework.SkipUnlessProviderIs("gce")

		initialGroupSize, err := framework.GroupSize(framework.TestContext.CloudConfig.NodeInstanceGroup)
		framework.ExpectNoError(err, "Error getting group size")

		By("creating a pd")
		diskName, err := framework.CreatePDWithRetry()
		framework.ExpectNoError(err, "Error creating a pd")

		host0Pod := testPDPod([]string{diskName}, host0Name, false, 1)
		originalCount := len(nodes.Items)
		containerName := "mycontainer"
		nodeToDelete := &nodes.Items[0]

		defer func() {
			By("defer: cleaning up PD-RW test env")
			framework.Logf("defer cleanup errors can usually be ignored")
			detachAndDeletePDs(diskName, []types.NodeName{host0Name})
			nodeToDelete.ObjectMeta.SetResourceVersion("0")
			// need to set the resource version or else the Create() fails
			_, err := nodeClient.Create(nodeToDelete)
			framework.ExpectNoError(err, "Unable to re-create the deleted node")
			framework.ExpectNoError(framework.WaitForGroupSize(framework.TestContext.CloudConfig.NodeInstanceGroup, int32(initialGroupSize)), "Unable to get the node group back to the original size")
			framework.WaitForNodeToBeReady(f.ClientSet, nodeToDelete.Name, nodeStatusTimeout)
			framework.WaitForAllNodesSchedulable(f.ClientSet, nodeStatusTimeout)
			nodes = framework.GetReadySchedulableNodesOrDie(f.ClientSet)
			Expect(len(nodes.Items)).To(Equal(originalCount), "Requires node count to return to original node count.")
		}()

		By("creating host0Pod on node0")
		_, err = podClient.Create(host0Pod)
		framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod %q: %v", host0Pod.Name, err))
		framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))

		By("writing content to host0Pod")
		testFile := "/testpd1/tracker"
		testFileContents := fmt.Sprintf("%v", mathrand.Int())
		framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFile, testFileContents))
		framework.Logf(fmt.Sprintf("wrote %q to file %q in pod %q on node %q", testFileContents, testFile, host0Pod.Name, host0Name))

		By("verifying PD is present in node0's VolumeInUse list")
		framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, true /* should exist*/))

		By("deleting host0 api object")
		framework.ExpectNoError(nodeClient.Delete(string(host0Name), metav1.NewDeleteOptions(0)), "Unable to delete host0")
		By("deleting host0Pod")
		framework.ExpectNoError(podClient.Delete(host0Pod.Name, metav1.NewDeleteOptions(0)), "Unable to delete host0Pod")
		By("Waiting for pd to detach from host0")
		framework.ExpectNoError(waitForPDDetach(diskName, host0Name), "Timed out waiting for detach pd")
	})

	It("should be able to delete a non-existent PD without error", func() {
		framework.SkipUnlessProviderIs("gce")

		By("delete a PD")
		framework.ExpectNoError(framework.DeletePDWithRetry("non-exist"))
	})
})

func verifyPDContentsViaContainer(f *framework.Framework, podName, containerName string, fileAndContentToVerify map[string]string) {
	for filePath, expectedContents := range fileAndContentToVerify {
		var value string
		// Add a retry to avoid temporal failure in reading the content
		for i := 0; i < maxReadRetry; i++ {
			v, err := f.ReadFileViaContainer(podName, containerName, filePath)
			value = v
			if err != nil {
				framework.Logf("Error reading file: %v", err)
			}
			framework.ExpectNoError(err)
			framework.Logf("Read file %q with content: %v (iteration %d)", filePath, v, i)
			if strings.TrimSpace(v) != strings.TrimSpace(expectedContents) {
				framework.Logf("Warning: read content <%q> does not match execpted content <%q>.", v, expectedContents)
				size, err := f.CheckFileSizeViaContainer(podName, containerName, filePath)
				if err != nil {
					framework.Logf("Error checking file size: %v", err)
				}
				framework.Logf("Check file %q size: %q", filePath, size)
			} else {
				break
			}
		}
		Expect(strings.TrimSpace(value)).To(Equal(strings.TrimSpace(expectedContents)))
	}
}

func detachPD(nodeName types.NodeName, pdName string) error {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		gceCloud, err := framework.GetGCECloud()
		if err != nil {
			return err
		}

		err = gceCloud.DetachDisk(pdName, nodeName)
		if err != nil {
			if gerr, ok := err.(*googleapi.Error); ok && strings.Contains(gerr.Message, "Invalid value for field 'disk'") {
				// PD already detached, ignore error.
				return nil
			}

			framework.Logf("Error detaching PD %q: %v", pdName, err)
		}

		return err
	} else if framework.TestContext.Provider == "aws" {
		client := ec2.New(session.New())

		tokens := strings.Split(pdName, "/")
		awsVolumeID := tokens[len(tokens)-1]

		request := ec2.DetachVolumeInput{
			VolumeId: aws.String(awsVolumeID),
		}

		_, err := client.DetachVolume(&request)
		if err != nil {
			return fmt.Errorf("error detaching EBS volume: %v", err)
		}
		return nil
	} else {
		return fmt.Errorf("Provider does not support volume detaching")
	}
}

func testPDPod(diskNames []string, targetNode types.NodeName, readOnly bool, numContainers int) *v1.Pod {
	containers := make([]v1.Container, numContainers)
	for i := range containers {
		containers[i].Name = "mycontainer"
		if numContainers > 1 {
			containers[i].Name = fmt.Sprintf("mycontainer%v", i+1)
		}

		containers[i].Image = imageutils.GetBusyBoxImage()

		containers[i].Command = []string{"sleep", "6000"}

		containers[i].VolumeMounts = make([]v1.VolumeMount, len(diskNames))
		for k := range diskNames {
			containers[i].VolumeMounts[k].Name = fmt.Sprintf("testpd%v", k+1)
			containers[i].VolumeMounts[k].MountPath = fmt.Sprintf("/testpd%v", k+1)
		}

		containers[i].Resources.Limits = v1.ResourceList{}
		containers[i].Resources.Limits[v1.ResourceCPU] = *resource.NewQuantity(int64(0), resource.DecimalSI)

	}

	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: testapi.Groups[v1.GroupName].GroupVersion().String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "pd-test-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			Containers: containers,
			NodeName:   string(targetNode),
		},
	}

	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		pod.Spec.Volumes = make([]v1.Volume, len(diskNames))
		for k, diskName := range diskNames {
			pod.Spec.Volumes[k].Name = fmt.Sprintf("testpd%v", k+1)
			pod.Spec.Volumes[k].VolumeSource = v1.VolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName:   diskName,
					FSType:   "ext4",
					ReadOnly: readOnly,
				},
			}
		}
	} else if framework.TestContext.Provider == "aws" {
		pod.Spec.Volumes = make([]v1.Volume, len(diskNames))
		for k, diskName := range diskNames {
			pod.Spec.Volumes[k].Name = fmt.Sprintf("testpd%v", k+1)
			pod.Spec.Volumes[k].VolumeSource = v1.VolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: diskName,
					FSType:   "ext4",
					ReadOnly: readOnly,
				},
			}
		}
	} else {
		panic("Unknown provider: " + framework.TestContext.Provider)
	}

	return pod
}

// Waits for specified PD to to detach from specified hostName
func waitForPDDetach(diskName string, nodeName types.NodeName) error {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		framework.Logf("Waiting for GCE PD %q to detach from node %q.", diskName, nodeName)
		gceCloud, err := framework.GetGCECloud()
		if err != nil {
			return err
		}

		for start := time.Now(); time.Since(start) < gcePDDetachTimeout; time.Sleep(gcePDDetachPollTime) {
			diskAttached, err := gceCloud.DiskIsAttached(diskName, nodeName)
			if err != nil {
				framework.Logf("Error waiting for PD %q to detach from node %q. 'DiskIsAttached(...)' failed with %v", diskName, nodeName, err)
				return err
			}

			if !diskAttached {
				// Specified disk does not appear to be attached to specified node
				framework.Logf("GCE PD %q appears to have successfully detached from %q.", diskName, nodeName)
				return nil
			}

			framework.Logf("Waiting for GCE PD %q to detach from %q.", diskName, nodeName)
		}

		return fmt.Errorf("Gave up waiting for GCE PD %q to detach from %q after %v", diskName, nodeName, gcePDDetachTimeout)
	}

	return nil
}

func detachAndDeletePDs(diskName string, hosts []types.NodeName) {
	for _, host := range hosts {
		framework.Logf("Detaching GCE PD %q from node %q.", diskName, host)
		detachPD(host, diskName)
		By(fmt.Sprintf("Waiting for PD %q to detach from %q", diskName, host))
		waitForPDDetach(diskName, host)
	}
	By(fmt.Sprintf("Deleting PD %q", diskName))
	framework.ExpectNoError(framework.DeletePDWithRetry(diskName))
}

func waitForPDInVolumesInUse(
	nodeClient v1core.NodeInterface,
	diskName string,
	nodeName types.NodeName,
	timeout time.Duration,
	shouldExist bool) error {
	logStr := "to contain"
	if !shouldExist {
		logStr = "to NOT contain"
	}
	framework.Logf(
		"Waiting for node %s's VolumesInUse Status %s PD %q",
		nodeName, logStr, diskName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(nodeStatusPollTime) {
		nodeObj, err := nodeClient.Get(string(nodeName), metav1.GetOptions{})
		if err != nil || nodeObj == nil {
			framework.Logf(
				"Failed to fetch node object %q from API server. err=%v",
				nodeName, err)
			continue
		}

		exists := false
		for _, volumeInUse := range nodeObj.Status.VolumesInUse {
			volumeInUseStr := string(volumeInUse)
			if strings.Contains(volumeInUseStr, diskName) {
				if shouldExist {
					framework.Logf(
						"Found PD %q in node %q's VolumesInUse Status: %q",
						diskName, nodeName, volumeInUseStr)
					return nil
				}

				exists = true
			}
		}

		if !shouldExist && !exists {
			framework.Logf(
				"Verified PD %q does not exist in node %q's VolumesInUse Status.",
				diskName, nodeName)
			return nil
		}
	}

	return fmt.Errorf(
		"Timed out waiting for node %s VolumesInUse Status %s diskName %q",
		nodeName, logStr, diskName)
}
