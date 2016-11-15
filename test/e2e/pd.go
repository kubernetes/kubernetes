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

package e2e

import (
	"fmt"
	mathrand "math/rand"
	"strings"
	"time"

	"google.golang.org/api/googleapi"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	awscloud "k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	gcePDDetachTimeout  = 10 * time.Minute
	gcePDDetachPollTime = 10 * time.Second
	nodeStatusTimeout   = 1 * time.Minute
	nodeStatusPollTime  = 1 * time.Second
	gcePDRetryTimeout   = 5 * time.Minute
	gcePDRetryPollTime  = 5 * time.Second
	maxReadRetry        = 3
)

var _ = framework.KubeDescribe("Pod Disks", func() {
	var (
		podClient  unversionedcore.PodInterface
		nodeClient unversionedcore.NodeInterface
		host0Name  types.NodeName
		host1Name  types.NodeName
	)
	f := framework.NewDefaultFramework("pod-disks")

	BeforeEach(func() {
		framework.SkipUnlessNodeCountIsAtLeast(2)

		podClient = f.ClientSet.Core().Pods(f.Namespace.Name)
		nodeClient = f.ClientSet.Core().Nodes()
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)

		Expect(len(nodes.Items)).To(BeNumerically(">=", 2), "Requires at least 2 nodes")

		host0Name = types.NodeName(nodes.Items[0].ObjectMeta.Name)
		host1Name = types.NodeName(nodes.Items[1].ObjectMeta.Name)

		mathrand.Seed(time.Now().UTC().UnixNano())
	})

	It("should schedule a pod w/ a RW PD, ungracefully remove it, then schedule it on another host [Slow]", func() {
		framework.SkipUnlessProviderIs("gce", "gke", "aws")

		By("creating PD")
		diskName, err := createPDWithRetry()
		framework.ExpectNoError(err, "Error creating PD")

		host0Pod := testPDPod([]string{diskName}, host0Name, false /* readOnly */, 1 /* numContainers */)
		host1Pod := testPDPod([]string{diskName}, host1Name, false /* readOnly */, 1 /* numContainers */)
		containerName := "mycontainer"

		defer func() {
			// Teardown pods, PD. Ignore errors.
			// Teardown should do nothing unless test failed.
			By("cleaning up PD-RW test environment")
			podClient.Delete(host0Pod.Name, api.NewDeleteOptions(0))
			podClient.Delete(host1Pod.Name, api.NewDeleteOptions(0))
			detachAndDeletePDs(diskName, []types.NodeName{host0Name, host1Name})
		}()

		By("submitting host0Pod to kubernetes")
		_, err = podClient.Create(host0Pod)
		framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))

		framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))

		testFile := "/testpd1/tracker"
		testFileContents := fmt.Sprintf("%v", mathrand.Int())

		framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFile, testFileContents))
		framework.Logf("Wrote value: %v", testFileContents)

		// Verify that disk shows up for in node 1's VolumeInUse list
		framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, true /* shouldExist */))

		By("deleting host0Pod")
		// Delete pod with 0 grace period
		framework.ExpectNoError(podClient.Delete(host0Pod.Name, api.NewDeleteOptions(0)), "Failed to delete host0Pod")

		By("submitting host1Pod to kubernetes")
		_, err = podClient.Create(host1Pod)
		framework.ExpectNoError(err, "Failed to create host1Pod")

		framework.ExpectNoError(f.WaitForPodRunningSlow(host1Pod.Name))

		v, err := f.ReadFileViaContainer(host1Pod.Name, containerName, testFile)
		framework.ExpectNoError(err)
		framework.Logf("Read value: %v", v)

		Expect(strings.TrimSpace(v)).To(Equal(strings.TrimSpace(testFileContents)))

		// Verify that disk is removed from node 1's VolumeInUse list
		framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, false /* shouldExist */))

		By("deleting host1Pod")
		framework.ExpectNoError(podClient.Delete(host1Pod.Name, api.NewDeleteOptions(0)), "Failed to delete host1Pod")

		By("Test completed successfully, waiting for PD to safely detach")
		waitForPDDetach(diskName, host0Name)
		waitForPDDetach(diskName, host1Name)

		return
	})

	It("Should schedule a pod w/ a RW PD, gracefully remove it, then schedule it on another host [Slow]", func() {
		framework.SkipUnlessProviderIs("gce", "gke", "aws")

		By("creating PD")
		diskName, err := createPDWithRetry()
		framework.ExpectNoError(err, "Error creating PD")

		host0Pod := testPDPod([]string{diskName}, host0Name, false /* readOnly */, 1 /* numContainers */)
		host1Pod := testPDPod([]string{diskName}, host1Name, false /* readOnly */, 1 /* numContainers */)
		containerName := "mycontainer"

		defer func() {
			// Teardown pods, PD. Ignore errors.
			// Teardown should do nothing unless test failed.
			By("cleaning up PD-RW test environment")
			podClient.Delete(host0Pod.Name, &api.DeleteOptions{})
			podClient.Delete(host1Pod.Name, &api.DeleteOptions{})
			detachAndDeletePDs(diskName, []types.NodeName{host0Name, host1Name})
		}()

		By("submitting host0Pod to kubernetes")
		_, err = podClient.Create(host0Pod)
		framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))

		framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))

		testFile := "/testpd1/tracker"
		testFileContents := fmt.Sprintf("%v", mathrand.Int())

		framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFile, testFileContents))
		framework.Logf("Wrote value: %v", testFileContents)

		// Verify that disk shows up for in node 1's VolumeInUse list
		framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, true /* shouldExist */))

		By("deleting host0Pod")
		// Delete pod with default grace period 30s
		framework.ExpectNoError(podClient.Delete(host0Pod.Name, &api.DeleteOptions{}), "Failed to delete host0Pod")

		By("submitting host1Pod to kubernetes")
		_, err = podClient.Create(host1Pod)
		framework.ExpectNoError(err, "Failed to create host1Pod")

		framework.ExpectNoError(f.WaitForPodRunningSlow(host1Pod.Name))

		v, err := f.ReadFileViaContainer(host1Pod.Name, containerName, testFile)
		framework.ExpectNoError(err)
		framework.Logf("Read value: %v", v)

		Expect(strings.TrimSpace(v)).To(Equal(strings.TrimSpace(testFileContents)))

		// Verify that disk is removed from node 1's VolumeInUse list
		framework.ExpectNoError(waitForPDInVolumesInUse(nodeClient, diskName, host0Name, nodeStatusTimeout, false /* shouldExist */))

		By("deleting host1Pod")
		framework.ExpectNoError(podClient.Delete(host1Pod.Name, &api.DeleteOptions{}), "Failed to delete host1Pod")

		By("Test completed successfully, waiting for PD to safely detach")
		waitForPDDetach(diskName, host0Name)
		waitForPDDetach(diskName, host1Name)

		return
	})

	It("should schedule a pod w/ a readonly PD on two hosts, then remove both ungracefully. [Slow]", func() {
		framework.SkipUnlessProviderIs("gce", "gke")

		By("creating PD")
		diskName, err := createPDWithRetry()
		framework.ExpectNoError(err, "Error creating PD")

		rwPod := testPDPod([]string{diskName}, host0Name, false /* readOnly */, 1 /* numContainers */)
		host0ROPod := testPDPod([]string{diskName}, host0Name, true /* readOnly */, 1 /* numContainers */)
		host1ROPod := testPDPod([]string{diskName}, host1Name, true /* readOnly */, 1 /* numContainers */)

		defer func() {
			By("cleaning up PD-RO test environment")
			// Teardown pods, PD. Ignore errors.
			// Teardown should do nothing unless test failed.
			podClient.Delete(rwPod.Name, api.NewDeleteOptions(0))
			podClient.Delete(host0ROPod.Name, api.NewDeleteOptions(0))
			podClient.Delete(host1ROPod.Name, api.NewDeleteOptions(0))
			detachAndDeletePDs(diskName, []types.NodeName{host0Name, host1Name})
		}()

		By("submitting rwPod to ensure PD is formatted")
		_, err = podClient.Create(rwPod)
		framework.ExpectNoError(err, "Failed to create rwPod")
		framework.ExpectNoError(f.WaitForPodRunningSlow(rwPod.Name))
		// Delete pod with 0 grace period
		framework.ExpectNoError(podClient.Delete(rwPod.Name, api.NewDeleteOptions(0)), "Failed to delete host0Pod")
		framework.ExpectNoError(waitForPDDetach(diskName, host0Name))

		By("submitting host0ROPod to kubernetes")
		_, err = podClient.Create(host0ROPod)
		framework.ExpectNoError(err, "Failed to create host0ROPod")

		By("submitting host1ROPod to kubernetes")
		_, err = podClient.Create(host1ROPod)
		framework.ExpectNoError(err, "Failed to create host1ROPod")

		framework.ExpectNoError(f.WaitForPodRunningSlow(host0ROPod.Name))

		framework.ExpectNoError(f.WaitForPodRunningSlow(host1ROPod.Name))

		By("deleting host0ROPod")
		framework.ExpectNoError(podClient.Delete(host0ROPod.Name, api.NewDeleteOptions(0)), "Failed to delete host0ROPod")

		By("deleting host1ROPod")
		framework.ExpectNoError(podClient.Delete(host1ROPod.Name, api.NewDeleteOptions(0)), "Failed to delete host1ROPod")

		By("Test completed successfully, waiting for PD to safely detach")
		waitForPDDetach(diskName, host0Name)
		waitForPDDetach(diskName, host1Name)
	})

	It("Should schedule a pod w/ a readonly PD on two hosts, then remove both gracefully. [Slow]", func() {
		framework.SkipUnlessProviderIs("gce", "gke")

		By("creating PD")
		diskName, err := createPDWithRetry()
		framework.ExpectNoError(err, "Error creating PD")

		rwPod := testPDPod([]string{diskName}, host0Name, false /* readOnly */, 1 /* numContainers */)
		host0ROPod := testPDPod([]string{diskName}, host0Name, true /* readOnly */, 1 /* numContainers */)
		host1ROPod := testPDPod([]string{diskName}, host1Name, true /* readOnly */, 1 /* numContainers */)

		defer func() {
			By("cleaning up PD-RO test environment")
			// Teardown pods, PD. Ignore errors.
			// Teardown should do nothing unless test failed.
			podClient.Delete(rwPod.Name, &api.DeleteOptions{})
			podClient.Delete(host0ROPod.Name, &api.DeleteOptions{})
			podClient.Delete(host1ROPod.Name, &api.DeleteOptions{})
			detachAndDeletePDs(diskName, []types.NodeName{host0Name, host1Name})
		}()

		By("submitting rwPod to ensure PD is formatted")
		_, err = podClient.Create(rwPod)
		framework.ExpectNoError(err, "Failed to create rwPod")
		framework.ExpectNoError(f.WaitForPodRunningSlow(rwPod.Name))
		// Delete pod with default grace period 30s
		framework.ExpectNoError(podClient.Delete(rwPod.Name, &api.DeleteOptions{}), "Failed to delete host0Pod")
		framework.ExpectNoError(waitForPDDetach(diskName, host0Name))

		By("submitting host0ROPod to kubernetes")
		_, err = podClient.Create(host0ROPod)
		framework.ExpectNoError(err, "Failed to create host0ROPod")

		By("submitting host1ROPod to kubernetes")
		_, err = podClient.Create(host1ROPod)
		framework.ExpectNoError(err, "Failed to create host1ROPod")

		framework.ExpectNoError(f.WaitForPodRunningSlow(host0ROPod.Name))

		framework.ExpectNoError(f.WaitForPodRunningSlow(host1ROPod.Name))

		By("deleting host0ROPod")
		framework.ExpectNoError(podClient.Delete(host0ROPod.Name, &api.DeleteOptions{}), "Failed to delete host0ROPod")

		By("deleting host1ROPod")
		framework.ExpectNoError(podClient.Delete(host1ROPod.Name, &api.DeleteOptions{}), "Failed to delete host1ROPod")

		By("Test completed successfully, waiting for PD to safely detach")
		waitForPDDetach(diskName, host0Name)
		waitForPDDetach(diskName, host1Name)
	})

	It("should schedule a pod w/ a RW PD shared between multiple containers, write to PD, delete pod, verify contents, and repeat in rapid succession [Slow]", func() {
		framework.SkipUnlessProviderIs("gce", "gke", "aws")

		By("creating PD")
		diskName, err := createPDWithRetry()
		framework.ExpectNoError(err, "Error creating PD")
		numContainers := 4
		var host0Pod *api.Pod

		defer func() {
			By("cleaning up PD-RW test environment")
			// Teardown pods, PD. Ignore errors.
			// Teardown should do nothing unless test failed.
			if host0Pod != nil {
				podClient.Delete(host0Pod.Name, api.NewDeleteOptions(0))
			}
			detachAndDeletePDs(diskName, []types.NodeName{host0Name})
		}()

		fileAndContentToVerify := make(map[string]string)
		for i := 0; i < 3; i++ {
			framework.Logf("PD Read/Writer Iteration #%v", i)
			By("submitting host0Pod to kubernetes")
			host0Pod = testPDPod([]string{diskName}, host0Name, false /* readOnly */, numContainers)
			_, err = podClient.Create(host0Pod)
			framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))

			framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))

			// randomly select a container and read/verify pd contents from it
			containerName := fmt.Sprintf("mycontainer%v", mathrand.Intn(numContainers)+1)
			verifyPDContentsViaContainer(f, host0Pod.Name, containerName, fileAndContentToVerify)

			// Randomly select a container to write a file to PD from
			containerName = fmt.Sprintf("mycontainer%v", mathrand.Intn(numContainers)+1)
			testFile := fmt.Sprintf("/testpd1/tracker%v", i)
			testFileContents := fmt.Sprintf("%v", mathrand.Int())
			fileAndContentToVerify[testFile] = testFileContents
			framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFile, testFileContents))
			framework.Logf("Wrote value: \"%v\" to PD %q from pod %q container %q", testFileContents, diskName, host0Pod.Name, containerName)

			// Randomly select a container and read/verify pd contents from it
			containerName = fmt.Sprintf("mycontainer%v", mathrand.Intn(numContainers)+1)
			verifyPDContentsViaContainer(f, host0Pod.Name, containerName, fileAndContentToVerify)

			By("deleting host0Pod")
			framework.ExpectNoError(podClient.Delete(host0Pod.Name, api.NewDeleteOptions(0)), "Failed to delete host0Pod")
		}

		By("Test completed successfully, waiting for PD to safely detach")
		waitForPDDetach(diskName, host0Name)
	})

	It("should schedule a pod w/two RW PDs both mounted to one container, write to PD, verify contents, delete pod, recreate pod, verify contents, and repeat in rapid succession [Slow]", func() {
		framework.SkipUnlessProviderIs("gce", "gke", "aws")

		By("creating PD1")
		disk1Name, err := createPDWithRetry()
		framework.ExpectNoError(err, "Error creating PD1")
		By("creating PD2")
		disk2Name, err := createPDWithRetry()
		framework.ExpectNoError(err, "Error creating PD2")
		var host0Pod *api.Pod

		defer func() {
			By("cleaning up PD-RW test environment")
			// Teardown pods, PD. Ignore errors.
			// Teardown should do nothing unless test failed.
			if host0Pod != nil {
				podClient.Delete(host0Pod.Name, api.NewDeleteOptions(0))
			}
			detachAndDeletePDs(disk1Name, []types.NodeName{host0Name})
			detachAndDeletePDs(disk2Name, []types.NodeName{host0Name})
		}()

		containerName := "mycontainer"
		fileAndContentToVerify := make(map[string]string)
		for i := 0; i < 3; i++ {
			framework.Logf("PD Read/Writer Iteration #%v", i)
			By("submitting host0Pod to kubernetes")
			host0Pod = testPDPod([]string{disk1Name, disk2Name}, host0Name, false /* readOnly */, 1 /* numContainers */)
			_, err = podClient.Create(host0Pod)
			framework.ExpectNoError(err, fmt.Sprintf("Failed to create host0Pod: %v", err))

			framework.ExpectNoError(f.WaitForPodRunningSlow(host0Pod.Name))

			// Read/verify pd contents for both disks from container
			verifyPDContentsViaContainer(f, host0Pod.Name, containerName, fileAndContentToVerify)

			// Write a file to both PDs from container
			testFilePD1 := fmt.Sprintf("/testpd1/tracker%v", i)
			testFilePD2 := fmt.Sprintf("/testpd2/tracker%v", i)
			testFilePD1Contents := fmt.Sprintf("%v", mathrand.Int())
			testFilePD2Contents := fmt.Sprintf("%v", mathrand.Int())
			fileAndContentToVerify[testFilePD1] = testFilePD1Contents
			fileAndContentToVerify[testFilePD2] = testFilePD2Contents
			framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFilePD1, testFilePD1Contents))
			framework.Logf("Wrote value: \"%v\" to PD1 (%q) from pod %q container %q", testFilePD1Contents, disk1Name, host0Pod.Name, containerName)
			framework.ExpectNoError(f.WriteFileViaContainer(host0Pod.Name, containerName, testFilePD2, testFilePD2Contents))
			framework.Logf("Wrote value: \"%v\" to PD2 (%q) from pod %q container %q", testFilePD2Contents, disk2Name, host0Pod.Name, containerName)

			// Read/verify pd contents for both disks from container
			verifyPDContentsViaContainer(f, host0Pod.Name, containerName, fileAndContentToVerify)

			By("deleting host0Pod")
			framework.ExpectNoError(podClient.Delete(host0Pod.Name, api.NewDeleteOptions(0)), "Failed to delete host0Pod")
		}

		By("Test completed successfully, waiting for PD to safely detach")
		waitForPDDetach(disk1Name, host0Name)
		waitForPDDetach(disk2Name, host0Name)
	})
})

func createPDWithRetry() (string, error) {
	newDiskName := ""
	var err error
	for start := time.Now(); time.Since(start) < gcePDRetryTimeout; time.Sleep(gcePDRetryPollTime) {
		if newDiskName, err = createPD(); err != nil {
			framework.Logf("Couldn't create a new PD. Sleeping 5 seconds (%v)", err)
			continue
		}
		framework.Logf("Successfully created a new PD: %q.", newDiskName)
		break
	}
	return newDiskName, err
}

func deletePDWithRetry(diskName string) {
	var err error
	for start := time.Now(); time.Since(start) < gcePDRetryTimeout; time.Sleep(gcePDRetryPollTime) {
		if err = deletePD(diskName); err != nil {
			framework.Logf("Couldn't delete PD %q. Sleeping 5 seconds (%v)", diskName, err)
			continue
		}
		framework.Logf("Successfully deleted PD %q.", diskName)
		break
	}
	framework.ExpectNoError(err, "Error deleting PD")
}

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

func createPD() (string, error) {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		pdName := fmt.Sprintf("%s-%s", framework.TestContext.Prefix, string(uuid.NewUUID()))

		gceCloud, err := getGCECloud()
		if err != nil {
			return "", err
		}

		tags := map[string]string{}
		err = gceCloud.CreateDisk(pdName, gcecloud.DiskTypeSSD, framework.TestContext.CloudConfig.Zone, 10 /* sizeGb */, tags)
		if err != nil {
			return "", err
		}
		return pdName, nil
	} else if framework.TestContext.Provider == "aws" {
		client := ec2.New(session.New())

		request := &ec2.CreateVolumeInput{}
		request.AvailabilityZone = aws.String(cloudConfig.Zone)
		request.Size = aws.Int64(10)
		request.VolumeType = aws.String(awscloud.DefaultVolumeType)
		response, err := client.CreateVolume(request)
		if err != nil {
			return "", err
		}

		az := aws.StringValue(response.AvailabilityZone)
		awsID := aws.StringValue(response.VolumeId)

		volumeName := "aws://" + az + "/" + awsID
		return volumeName, nil
	} else {
		return "", fmt.Errorf("Provider does not support volume creation")
	}
}

func deletePD(pdName string) error {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		gceCloud, err := getGCECloud()
		if err != nil {
			return err
		}

		err = gceCloud.DeleteDisk(pdName)

		if err != nil {
			if gerr, ok := err.(*googleapi.Error); ok && len(gerr.Errors) > 0 && gerr.Errors[0].Reason == "notFound" {
				// PD already exists, ignore error.
				return nil
			}

			framework.Logf("Error deleting PD %q: %v", pdName, err)
		}
		return err
	} else if framework.TestContext.Provider == "aws" {
		client := ec2.New(session.New())

		tokens := strings.Split(pdName, "/")
		awsVolumeID := tokens[len(tokens)-1]

		request := &ec2.DeleteVolumeInput{VolumeId: aws.String(awsVolumeID)}
		_, err := client.DeleteVolume(request)
		if err != nil {
			if awsError, ok := err.(awserr.Error); ok && awsError.Code() == "InvalidVolume.NotFound" {
				framework.Logf("Volume deletion implicitly succeeded because volume %q does not exist.", pdName)
			} else {
				return fmt.Errorf("error deleting EBS volumes: %v", err)
			}
		}
		return nil
	} else {
		return fmt.Errorf("Provider does not support volume deletion")
	}
}

func detachPD(nodeName types.NodeName, pdName string) error {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		gceCloud, err := getGCECloud()
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

func testPDPod(diskNames []string, targetNode types.NodeName, readOnly bool, numContainers int) *api.Pod {
	containers := make([]api.Container, numContainers)
	for i := range containers {
		containers[i].Name = "mycontainer"
		if numContainers > 1 {
			containers[i].Name = fmt.Sprintf("mycontainer%v", i+1)
		}

		containers[i].Image = "gcr.io/google_containers/busybox:1.24"

		containers[i].Command = []string{"sleep", "6000"}

		containers[i].VolumeMounts = make([]api.VolumeMount, len(diskNames))
		for k := range diskNames {
			containers[i].VolumeMounts[k].Name = fmt.Sprintf("testpd%v", k+1)
			containers[i].VolumeMounts[k].MountPath = fmt.Sprintf("/testpd%v", k+1)
		}

		containers[i].Resources.Limits = api.ResourceList{}
		containers[i].Resources.Limits[api.ResourceCPU] = *resource.NewQuantity(int64(0), resource.DecimalSI)

	}

	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
		},
		ObjectMeta: api.ObjectMeta{
			Name: "pd-test-" + string(uuid.NewUUID()),
		},
		Spec: api.PodSpec{
			Containers: containers,
			NodeName:   string(targetNode),
		},
	}

	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		pod.Spec.Volumes = make([]api.Volume, len(diskNames))
		for k, diskName := range diskNames {
			pod.Spec.Volumes[k].Name = fmt.Sprintf("testpd%v", k+1)
			pod.Spec.Volumes[k].VolumeSource = api.VolumeSource{
				GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
					PDName:   diskName,
					FSType:   "ext4",
					ReadOnly: readOnly,
				},
			}
		}
	} else if framework.TestContext.Provider == "aws" {
		pod.Spec.Volumes = make([]api.Volume, len(diskNames))
		for k, diskName := range diskNames {
			pod.Spec.Volumes[k].Name = fmt.Sprintf("testpd%v", k+1)
			pod.Spec.Volumes[k].VolumeSource = api.VolumeSource{
				AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
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
		gceCloud, err := getGCECloud()
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

func getGCECloud() (*gcecloud.GCECloud, error) {
	gceCloud, ok := framework.TestContext.CloudConfig.Provider.(*gcecloud.GCECloud)

	if !ok {
		return nil, fmt.Errorf("failed to convert CloudConfig.Provider to GCECloud: %#v", framework.TestContext.CloudConfig.Provider)
	}

	return gceCloud, nil
}

func detachAndDeletePDs(diskName string, hosts []types.NodeName) {
	for _, host := range hosts {
		framework.Logf("Detaching GCE PD %q from node %q.", diskName, host)
		detachPD(host, diskName)
		By(fmt.Sprintf("Waiting for PD %q to detach from %q", diskName, host))
		waitForPDDetach(diskName, host)
	}
	By(fmt.Sprintf("Deleting PD %q", diskName))
	deletePDWithRetry(diskName)
}

func waitForPDInVolumesInUse(
	nodeClient unversionedcore.NodeInterface,
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
		nodeObj, err := nodeClient.Get(string(nodeName))
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
