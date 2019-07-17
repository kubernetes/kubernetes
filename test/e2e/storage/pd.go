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
	"math/rand"
	"strings"
	"time"

	"google.golang.org/api/googleapi"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	gcePDDetachTimeout  = 10 * time.Minute
	gcePDDetachPollTime = 10 * time.Second
	nodeStatusTimeout   = 10 * time.Minute
	nodeStatusPollTime  = 1 * time.Second
	podEvictTimeout     = 2 * time.Minute
	maxReadRetry        = 3
	minNodes            = 2
)

var _ = utils.SIGDescribe("Pod Disks", func() {
	var (
		ns         string
		cs         clientset.Interface
		podClient  v1core.PodInterface
		nodeClient v1core.NodeInterface
		host0Name  types.NodeName
		host1Name  types.NodeName
		nodes      *v1.NodeList
	)
	f := framework.NewDefaultFramework("pod-disks")

	ginkgo.BeforeEach(func() {
		framework.SkipUnlessNodeCountIsAtLeast(minNodes)
		cs = f.ClientSet
		ns = f.Namespace.Name

		framework.SkipIfMultizone(cs)

		podClient = cs.CoreV1().Pods(ns)
		nodeClient = cs.CoreV1().Nodes()
		nodes = framework.GetReadySchedulableNodesOrDie(cs)
		gomega.Expect(len(nodes.Items)).To(gomega.BeNumerically(">=", minNodes), fmt.Sprintf("Requires at least %d nodes", minNodes))
		host0Name = types.NodeName(nodes.Items[0].ObjectMeta.Name)
		host1Name = types.NodeName(nodes.Items[1].ObjectMeta.Name)
	})

	ginkgo.Context("schedule pods each with a PD, delete pod and verify detach [Slow]", func() {
		const (
			podDefaultGrace   = "default (30s)"
			podImmediateGrace = "immediate (0s)"
		)
		var readOnlyMap = map[bool]string{
			true:  "read-only",
			false: "RW",
		}
		type testT struct {
			descr     string                // It description
			readOnly  bool                  // true means pd is read-only
			deleteOpt *metav1.DeleteOptions // pod delete option
		}
		tests := []testT{
			{
				descr:     podImmediateGrace,
				readOnly:  false,
				deleteOpt: metav1.NewDeleteOptions(0),
			},
			{
				descr:     podDefaultGrace,
				readOnly:  false,
				deleteOpt: &metav1.DeleteOptions{},
			},
			{
				descr:     podImmediateGrace,
				readOnly:  true,
				deleteOpt: metav1.NewDeleteOptions(0),
			},
			{
				descr:     podDefaultGrace,
				readOnly:  true,
				deleteOpt: &metav1.DeleteOptions{},
			},
		}

		for _, t := range tests {
			podDelOpt := t.deleteOpt
			readOnly := t.readOnly
			readOnlyTxt := readOnlyMap[readOnly]

			ginkgo.It(fmt.Sprintf("for %s PD with pod delete grace period of %q", readOnlyTxt, t.descr), func() {
				framework.SkipUnlessProviderIs("gce", "gke", "aws")
				if readOnly {
					framework.SkipIfProviderIs("aws")
				}

				ginkgo.By("creating PD")
				diskName, err := framework.CreatePDWithRetry()
				framework.ExpectNoError(err, "Error creating PD")

				var fmtPod *v1.Pod
				if readOnly {
					// if all test pods are RO then need a RW pod to format pd
					ginkgo.By("creating RW fmt Pod to ensure PD is formatted")
					fmtPod = testPDPod([]string{diskName}, host0Name, false, 1)
					_, err = podClient.Create(fmtPod)
					framework.ExpectNoError(err, "Failed to create fmtPod")
					framework.ExpectNoError(f.WaitForPodRunningSlow(fmtPod.Name))

					ginkgo.By("deleting the fmtPod")
					framework.ExpectNoError(podClient.Delete(fmtPod.Name, metav1.NewDeleteOptions(0)), "Failed to delete fmtPod")
					e2elog.Logf("deleted fmtPod %q", fmtPod.Name)
					ginkgo.By("waiting for PD to detach")
					framework.ExpectNoError(waitForPDDetach(diskName, host0Name))
				}

				// prepare to create two test pods on separate nodes
				host0Pod := testPDPod([]string{diskName}, host0Name, readOnly, 1)
				host1Pod := testPDPod([]string{diskName}, host1Name, readOnly, 1)

				defer func() {
					// Teardown should do nothing unless test failed
					ginkgo.By("defer: cleaning up PD-RW test environment")
					e2elog.Logf("defer cleanup errors can usually be ignored")
					if fmtPod != nil {
						podClient.Delete(fmtPod.Name, podDelOpt)
					}
					podClient.Delete(host0Pod.Name, podDelOpt)
					podClient.Delete(host1Pod.Name, podDelOpt)
					detachAndDeletePDs(diskName, []types.NodeName{host0Name, host1Name})
				}()

				ginkgo.By("creating host0Pod on node0")
				_, err = podClient.Create(host0Pod)
				framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))
				framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))
				e2elog.Logf("host0Pod: %q, node0: %q", host0Pod.Name, host0Name)

				var containerName, testFile, testFileContents string
				if !readOnly {
					ginkgo.By("writing content to host0Pod on node0")
					containerName = "mycontainer"
					testFile = "/testpd1/tracker"
					testFileContents = fmt.Sprintf("%v", rand.Int())
					framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFile, testFileContents))
					e2elog.Logf("wrote %q to file %q in pod %q on node %q", testFileContents, testFile, host0Pod.Name, host0Name)
					ginkgo.By("verifying PD is present in node0's VolumeInUse list")
					framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, true /* shouldExist */))
					ginkgo.By("deleting host0Pod") // delete this pod before creating next pod
					framework.ExpectNoError(podClient.Delete(host0Pod.Name, podDelOpt), "Failed to delete host0Pod")
					e2elog.Logf("deleted host0Pod %q", host0Pod.Name)
				}

				ginkgo.By("creating host1Pod on node1")
				_, err = podClient.Create(host1Pod)
				framework.ExpectNoError(err, "Failed to create host1Pod")
				framework.ExpectNoError(f.WaitForPodRunningSlow(host1Pod.Name))
				e2elog.Logf("host1Pod: %q, node1: %q", host1Pod.Name, host1Name)

				if readOnly {
					ginkgo.By("deleting host0Pod")
					framework.ExpectNoError(podClient.Delete(host0Pod.Name, podDelOpt), "Failed to delete host0Pod")
					e2elog.Logf("deleted host0Pod %q", host0Pod.Name)
				} else {
					ginkgo.By("verifying PD contents in host1Pod")
					verifyPDContentsViaContainer(f, host1Pod.Name, containerName, map[string]string{testFile: testFileContents})
					e2elog.Logf("verified PD contents in pod %q", host1Pod.Name)
					ginkgo.By("verifying PD is removed from node0")
					framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, false /* shouldExist */))
					e2elog.Logf("PD %q removed from node %q's VolumeInUse list", diskName, host1Pod.Name)
				}

				ginkgo.By("deleting host1Pod")
				framework.ExpectNoError(podClient.Delete(host1Pod.Name, podDelOpt), "Failed to delete host1Pod")
				e2elog.Logf("deleted host1Pod %q", host1Pod.Name)

				ginkgo.By("Test completed successfully, waiting for PD to detach from both nodes")
				waitForPDDetach(diskName, host0Name)
				waitForPDDetach(diskName, host1Name)
			})
		}
	})

	ginkgo.Context("schedule a pod w/ RW PD(s) mounted to 1 or more containers, write to PD, verify content, delete pod, and repeat in rapid succession [Slow]", func() {
		type testT struct {
			numContainers int
			numPDs        int
			repeatCnt     int
		}
		tests := []testT{
			{
				numContainers: 4,
				numPDs:        1,
				repeatCnt:     3,
			},
			{
				numContainers: 1,
				numPDs:        2,
				repeatCnt:     3,
			},
		}

		for _, t := range tests {
			numPDs := t.numPDs
			numContainers := t.numContainers

			ginkgo.It(fmt.Sprintf("using %d containers and %d PDs", numContainers, numPDs), func() {
				framework.SkipUnlessProviderIs("gce", "gke", "aws")
				var host0Pod *v1.Pod
				var err error
				fileAndContentToVerify := make(map[string]string)
				diskNames := make([]string, 0, numPDs)

				ginkgo.By(fmt.Sprintf("creating %d PD(s)", numPDs))
				for i := 0; i < numPDs; i++ {
					name, err := framework.CreatePDWithRetry()
					framework.ExpectNoError(err, fmt.Sprintf("Error creating PD %d", i))
					diskNames = append(diskNames, name)
				}

				defer func() {
					// Teardown should do nothing unless test failed.
					ginkgo.By("defer: cleaning up PD-RW test environment")
					e2elog.Logf("defer cleanup errors can usually be ignored")
					if host0Pod != nil {
						podClient.Delete(host0Pod.Name, metav1.NewDeleteOptions(0))
					}
					for _, diskName := range diskNames {
						detachAndDeletePDs(diskName, []types.NodeName{host0Name})
					}
				}()

				for i := 0; i < t.repeatCnt; i++ { // "rapid" repeat loop
					e2elog.Logf("PD Read/Writer Iteration #%v", i)
					ginkgo.By(fmt.Sprintf("creating host0Pod with %d containers on node0", numContainers))
					host0Pod = testPDPod(diskNames, host0Name, false /* readOnly */, numContainers)
					_, err = podClient.Create(host0Pod)
					framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))
					framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))

					ginkgo.By(fmt.Sprintf("writing %d file(s) via a container", numPDs))
					containerName := "mycontainer"
					if numContainers > 1 {
						containerName = fmt.Sprintf("mycontainer%v", rand.Intn(numContainers)+1)
					}
					for x := 1; x <= numPDs; x++ {
						testFile := fmt.Sprintf("/testpd%d/tracker%d", x, i)
						testFileContents := fmt.Sprintf("%v", rand.Int())
						fileAndContentToVerify[testFile] = testFileContents
						framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFile, testFileContents))
						e2elog.Logf("wrote %q to file %q in pod %q (container %q) on node %q", testFileContents, testFile, host0Pod.Name, containerName, host0Name)
					}

					ginkgo.By("verifying PD contents via a container")
					if numContainers > 1 {
						containerName = fmt.Sprintf("mycontainer%v", rand.Intn(numContainers)+1)
					}
					verifyPDContentsViaContainer(f, host0Pod.Name, containerName, fileAndContentToVerify)

					ginkgo.By("deleting host0Pod")
					framework.ExpectNoError(podClient.Delete(host0Pod.Name, metav1.NewDeleteOptions(0)), "Failed to delete host0Pod")
				}
				ginkgo.By(fmt.Sprintf("Test completed successfully, waiting for %d PD(s) to detach from node0", numPDs))
				for _, diskName := range diskNames {
					waitForPDDetach(diskName, host0Name)
				}
			})
		}
	})

	ginkgo.Context("detach in a disrupted environment [Slow] [Disruptive]", func() {
		const (
			deleteNode    = 1 // delete physical node
			deleteNodeObj = 2 // delete node's api object only
			evictPod      = 3 // evict host0Pod on node0
		)
		type testT struct {
			descr     string // It description
			disruptOp int    // disruptive operation performed on target node
		}
		tests := []testT{
			{
				descr:     "node is deleted",
				disruptOp: deleteNode,
			},
			{
				descr:     "node's API object is deleted",
				disruptOp: deleteNodeObj,
			},
			{
				descr:     "pod is evicted",
				disruptOp: evictPod,
			},
		}

		for _, t := range tests {
			disruptOp := t.disruptOp
			ginkgo.It(fmt.Sprintf("when %s", t.descr), func() {
				framework.SkipUnlessProviderIs("gce")
				origNodeCnt := len(nodes.Items) // healhy nodes running kubelet

				ginkgo.By("creating a pd")
				diskName, err := framework.CreatePDWithRetry()
				framework.ExpectNoError(err, "Error creating a pd")

				targetNode := &nodes.Items[0] // for node delete ops
				host0Pod := testPDPod([]string{diskName}, host0Name, false, 1)
				containerName := "mycontainer"

				defer func() {
					ginkgo.By("defer: cleaning up PD-RW test env")
					e2elog.Logf("defer cleanup errors can usually be ignored")
					ginkgo.By("defer: delete host0Pod")
					podClient.Delete(host0Pod.Name, metav1.NewDeleteOptions(0))
					ginkgo.By("defer: detach and delete PDs")
					detachAndDeletePDs(diskName, []types.NodeName{host0Name})
					if disruptOp == deleteNode || disruptOp == deleteNodeObj {
						if disruptOp == deleteNodeObj {
							targetNode.ObjectMeta.SetResourceVersion("0")
							// need to set the resource version or else the Create() fails
							ginkgo.By("defer: re-create host0 node object")
							_, err := nodeClient.Create(targetNode)
							framework.ExpectNoError(err, fmt.Sprintf("defer: Unable to re-create the deleted node object %q", targetNode.Name))
						}
						ginkgo.By("defer: verify the number of ready nodes")
						numNodes := countReadyNodes(cs, host0Name)
						// if this defer is reached due to an Expect then nested
						// Expects are lost, so use Failf here
						if numNodes != origNodeCnt {
							e2elog.Failf("defer: Requires current node count (%d) to return to original node count (%d)", numNodes, origNodeCnt)
						}
					}
				}()

				ginkgo.By("creating host0Pod on node0")
				_, err = podClient.Create(host0Pod)
				framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))
				ginkgo.By("waiting for host0Pod to be running")
				framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))

				ginkgo.By("writing content to host0Pod")
				testFile := "/testpd1/tracker"
				testFileContents := fmt.Sprintf("%v", rand.Int())
				framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFile, testFileContents))
				e2elog.Logf("wrote %q to file %q in pod %q on node %q", testFileContents, testFile, host0Pod.Name, host0Name)

				ginkgo.By("verifying PD is present in node0's VolumeInUse list")
				framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, true /* should exist*/))

				if disruptOp == deleteNode {
					ginkgo.By("getting gce instances")
					gceCloud, err := gce.GetGCECloud()
					framework.ExpectNoError(err, fmt.Sprintf("Unable to create gcloud client err=%v", err))
					output, err := gceCloud.ListInstanceNames(framework.TestContext.CloudConfig.ProjectID, framework.TestContext.CloudConfig.Zone)
					framework.ExpectNoError(err, fmt.Sprintf("Unable to get list of node instances err=%v output=%s", err, output))
					gomega.Expect(true, strings.Contains(string(output), string(host0Name)))

					ginkgo.By("deleting host0")
					err = gceCloud.DeleteInstance(framework.TestContext.CloudConfig.ProjectID, framework.TestContext.CloudConfig.Zone, string(host0Name))
					framework.ExpectNoError(err, fmt.Sprintf("Failed to delete host0Pod: err=%v", err))
					ginkgo.By("expecting host0 node to be re-created")
					numNodes := countReadyNodes(cs, host0Name)
					framework.ExpectEqual(numNodes, origNodeCnt, fmt.Sprintf("Requires current node count (%d) to return to original node count (%d)", numNodes, origNodeCnt))
					output, err = gceCloud.ListInstanceNames(framework.TestContext.CloudConfig.ProjectID, framework.TestContext.CloudConfig.Zone)
					framework.ExpectNoError(err, fmt.Sprintf("Unable to get list of node instances err=%v output=%s", err, output))
					gomega.Expect(false, strings.Contains(string(output), string(host0Name)))

				} else if disruptOp == deleteNodeObj {
					ginkgo.By("deleting host0's node api object")
					framework.ExpectNoError(nodeClient.Delete(string(host0Name), metav1.NewDeleteOptions(0)), "Unable to delete host0's node object")
					ginkgo.By("deleting host0Pod")
					framework.ExpectNoError(podClient.Delete(host0Pod.Name, metav1.NewDeleteOptions(0)), "Unable to delete host0Pod")

				} else if disruptOp == evictPod {
					evictTarget := &policyv1beta1.Eviction{
						ObjectMeta: metav1.ObjectMeta{
							Name:      host0Pod.Name,
							Namespace: ns,
						},
					}
					ginkgo.By("evicting host0Pod")
					err = wait.PollImmediate(framework.Poll, podEvictTimeout, func() (bool, error) {
						err = cs.CoreV1().Pods(ns).Evict(evictTarget)
						if err != nil {
							return false, nil
						}
						return true, nil
					})
					framework.ExpectNoError(err, fmt.Sprintf("failed to evict host0Pod after %v", podEvictTimeout))
				}

				ginkgo.By("waiting for pd to detach from host0")
				waitForPDDetach(diskName, host0Name)
			})
		}
	})

	ginkgo.It("should be able to delete a non-existent PD without error", func() {
		framework.SkipUnlessProviderIs("gce")

		ginkgo.By("delete a PD")
		framework.ExpectNoError(framework.DeletePDWithRetry("non-exist"))
	})
})

func countReadyNodes(c clientset.Interface, hostName types.NodeName) int {
	e2enode.WaitForNodeToBeReady(c, string(hostName), nodeStatusTimeout)
	framework.WaitForAllNodesSchedulable(c, nodeStatusTimeout)
	nodes := framework.GetReadySchedulableNodesOrDie(c)
	return len(nodes.Items)
}

func verifyPDContentsViaContainer(f *framework.Framework, podName, containerName string, fileAndContentToVerify map[string]string) {
	for filePath, expectedContents := range fileAndContentToVerify {
		var value string
		// Add a retry to avoid temporal failure in reading the content
		for i := 0; i < maxReadRetry; i++ {
			v, err := f.ReadFileViaContainer(podName, containerName, filePath)
			value = v
			if err != nil {
				e2elog.Logf("Error reading file: %v", err)
			}
			framework.ExpectNoError(err)
			e2elog.Logf("Read file %q with content: %v (iteration %d)", filePath, v, i)
			if strings.TrimSpace(v) != strings.TrimSpace(expectedContents) {
				e2elog.Logf("Warning: read content <%q> does not match execpted content <%q>.", v, expectedContents)
				size, err := f.CheckFileSizeViaContainer(podName, containerName, filePath)
				if err != nil {
					e2elog.Logf("Error checking file size: %v", err)
				}
				e2elog.Logf("Check file %q size: %q", filePath, size)
			} else {
				break
			}
		}
		framework.ExpectEqual(strings.TrimSpace(value), strings.TrimSpace(expectedContents))
	}
}

func detachPD(nodeName types.NodeName, pdName string) error {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		gceCloud, err := gce.GetGCECloud()
		if err != nil {
			return err
		}
		err = gceCloud.DetachDisk(pdName, nodeName)
		if err != nil {
			if gerr, ok := err.(*googleapi.Error); ok && strings.Contains(gerr.Message, "Invalid value for field 'disk'") {
				// PD already detached, ignore error.
				return nil
			}
			e2elog.Logf("Error detaching PD %q: %v", pdName, err)
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

// Returns pod spec suitable for api Create call. Handles gce, gke and aws providers only and
// escapes if a different provider is supplied.
// The first container name is hard-coded to "mycontainer". Subsequent containers are named:
// "mycontainer<number> where <number> is 1..numContainers. Note if there is only one container it's
// name has no number.
// Container's volumeMounts are hard-coded to "/testpd<number>" where <number> is 1..len(diskNames).
func testPDPod(diskNames []string, targetNode types.NodeName, readOnly bool, numContainers int) *v1.Pod {
	// escape if not a supported provider
	if !(framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" ||
		framework.TestContext.Provider == "aws") {
		e2elog.Failf(fmt.Sprintf("func `testPDPod` only supports gce, gke, and aws providers, not %v", framework.TestContext.Provider))
	}

	containers := make([]v1.Container, numContainers)
	for i := range containers {
		containers[i].Name = "mycontainer"
		if numContainers > 1 {
			containers[i].Name = fmt.Sprintf("mycontainer%v", i+1)
		}
		containers[i].Image = imageutils.GetE2EImage(imageutils.BusyBox)
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
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "pd-test-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			Containers: containers,
			NodeName:   string(targetNode),
		},
	}

	pod.Spec.Volumes = make([]v1.Volume, len(diskNames))
	for k, diskName := range diskNames {
		pod.Spec.Volumes[k].Name = fmt.Sprintf("testpd%v", k+1)
		if framework.TestContext.Provider == "aws" {
			pod.Spec.Volumes[k].VolumeSource = v1.VolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: diskName,
					FSType:   "ext4",
					ReadOnly: readOnly,
				},
			}
		} else { // "gce" or "gke"
			pod.Spec.Volumes[k].VolumeSource = v1.VolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName:   diskName,
					FSType:   "ext4",
					ReadOnly: readOnly,
				},
			}
		}
	}
	return pod
}

// Waits for specified PD to detach from specified hostName
func waitForPDDetach(diskName string, nodeName types.NodeName) error {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		e2elog.Logf("Waiting for GCE PD %q to detach from node %q.", diskName, nodeName)
		gceCloud, err := gce.GetGCECloud()
		if err != nil {
			return err
		}
		for start := time.Now(); time.Since(start) < gcePDDetachTimeout; time.Sleep(gcePDDetachPollTime) {
			diskAttached, err := gceCloud.DiskIsAttached(diskName, nodeName)
			if err != nil {
				e2elog.Logf("Error waiting for PD %q to detach from node %q. 'DiskIsAttached(...)' failed with %v", diskName, nodeName, err)
				return err
			}
			if !diskAttached {
				// Specified disk does not appear to be attached to specified node
				e2elog.Logf("GCE PD %q appears to have successfully detached from %q.", diskName, nodeName)
				return nil
			}
			e2elog.Logf("Waiting for GCE PD %q to detach from %q.", diskName, nodeName)
		}
		return fmt.Errorf("Gave up waiting for GCE PD %q to detach from %q after %v", diskName, nodeName, gcePDDetachTimeout)
	}
	return nil
}

func detachAndDeletePDs(diskName string, hosts []types.NodeName) {
	for _, host := range hosts {
		e2elog.Logf("Detaching GCE PD %q from node %q.", diskName, host)
		detachPD(host, diskName)
		ginkgo.By(fmt.Sprintf("Waiting for PD %q to detach from %q", diskName, host))
		waitForPDDetach(diskName, host)
	}
	ginkgo.By(fmt.Sprintf("Deleting PD %q", diskName))
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
	e2elog.Logf("Waiting for node %s's VolumesInUse Status %s PD %q", nodeName, logStr, diskName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(nodeStatusPollTime) {
		nodeObj, err := nodeClient.Get(string(nodeName), metav1.GetOptions{})
		if err != nil || nodeObj == nil {
			e2elog.Logf("Failed to fetch node object %q from API server. err=%v", nodeName, err)
			continue
		}
		exists := false
		for _, volumeInUse := range nodeObj.Status.VolumesInUse {
			volumeInUseStr := string(volumeInUse)
			if strings.Contains(volumeInUseStr, diskName) {
				if shouldExist {
					e2elog.Logf("Found PD %q in node %q's VolumesInUse Status: %q", diskName, nodeName, volumeInUseStr)
					return nil
				}
				exists = true
			}
		}
		if !shouldExist && !exists {
			e2elog.Logf("Verified PD %q does not exist in node %q's VolumesInUse Status.", diskName, nodeName)
			return nil
		}
	}
	return fmt.Errorf("Timed out waiting for node %s VolumesInUse Status %s diskName %q", nodeName, logStr, diskName)
}
