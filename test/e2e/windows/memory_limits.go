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

package windows

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"regexp"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/kubernetes/scheme"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("[Feature:Windows] Memory Limits [Serial] [Slow]", func() {

	f := framework.NewDefaultFramework("memory-limit-test-windows")

	ginkgo.BeforeEach(func() {
		// NOTE(vyta): these tests are Windows specific
		framework.SkipUnlessNodeOSDistroIs("windows")
	})

	ginkgo.Context("Allocatable node memory", func() {
		ginkgo.It("should be equal to a calculated allocatable memory value", func() {
			checkNodeAllocatableTest(f)
		})
	})

	ginkgo.Context("attempt to deploy past allocatable memory limits", func() {
		ginkgo.It("should fail deployments of pods once there isn't enough memory", func() {
			overrideAllocatableMemoryTest(f, 4)
		})
	})

})

type nodeMemory struct {
	// capacity
	capacity resource.Quantity
	// allocatable memory
	allocatable resource.Quantity
	// memory reserved for OS level processes
	systemReserve resource.Quantity
	// memory reserved for kubelet (not implemented)
	kubeReserve resource.Quantity
	// grace period memory limit (not implemented)
	softEviction resource.Quantity
	// no grace period memory limit
	hardEviction resource.Quantity
}

// runDensityBatchTest runs the density batch pod creation test
// checks that a calculated value for NodeAllocatable is equal to the reported value
func checkNodeAllocatableTest(f *framework.Framework) {

	nodeMem := getNodeMemory(f)
	framework.Logf("nodeMem says: %+v", nodeMem)

	// calculate the allocatable mem based on capacity - reserved amounts
	calculatedNodeAlloc := nodeMem.capacity.Copy()
	calculatedNodeAlloc.Sub(nodeMem.systemReserve)
	calculatedNodeAlloc.Sub(nodeMem.kubeReserve)
	calculatedNodeAlloc.Sub(nodeMem.softEviction)
	calculatedNodeAlloc.Sub(nodeMem.hardEviction)

	ginkgo.By(fmt.Sprintf("Checking stated allocatable memory %v against calculated allocatable memory %v", &nodeMem.allocatable, calculatedNodeAlloc))

	// sanity check against stated allocatable
	gomega.Expect(calculatedNodeAlloc.Cmp(nodeMem.allocatable)).To(gomega.Equal(0))
}

// Deploys `allocatablePods + 1` pods, each with a memory limit of `1/allocatablePods` of the total allocatable
// memory, then confirms that the last pod failed because of failedScheduling
func overrideAllocatableMemoryTest(f *framework.Framework, allocatablePods int) {
	const (
		podType = "memory_limit_test_pod"
	)

	totalAllocatable := getTotalAllocatableMemory(f)

	memValue := totalAllocatable.Value()
	memPerPod := memValue / int64(allocatablePods)
	ginkgo.By(fmt.Sprintf("Deploying %d pods with mem limit %v, then one additional pod", allocatablePods, memPerPod))

	// these should all work
	pods := newMemLimitTestPods(allocatablePods, imageutils.GetPauseImageName(), podType, strconv.FormatInt(memPerPod, 10))
	f.PodClient().CreateBatch(pods)

	failurePods := newMemLimitTestPods(1, imageutils.GetPauseImageName(), podType, strconv.FormatInt(memPerPod, 10))
	f.PodClient().Create(failurePods[0])

	gomega.Eventually(func() bool {
		eventList, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(metav1.ListOptions{})
		framework.ExpectNoError(err)
		for _, e := range eventList.Items {
			// Look for an event that shows FailedScheduling
			if e.Type == "Warning" && e.Reason == "FailedScheduling" && e.InvolvedObject.Name == failurePods[0].ObjectMeta.Name {
				framework.Logf("Found %+v event with message %+v", e.Reason, e.Message)
				return true
			}
		}
		return false
	}, 3*time.Minute, 10*time.Second).Should(gomega.Equal(true))
}

// newMemLimitTestPods creates a list of pods (specification) for test.
func newMemLimitTestPods(numPods int, imageName, podType string, memoryLimit string) []*v1.Pod {
	var pods []*v1.Pod

	memLimitQuantity, err := resource.ParseQuantity(memoryLimit)
	framework.ExpectNoError(err)

	for i := 0; i < numPods; i++ {

		podName := "test-" + string(uuid.NewUUID())
		pod := v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
				Labels: map[string]string{
					"type": podType,
					"name": podName,
				},
			},
			Spec: v1.PodSpec{
				// Restart policy is always (default).
				Containers: []v1.Container{
					{
						Image: imageName,
						Name:  podName,
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceMemory: memLimitQuantity,
							},
						},
					},
				},
				NodeSelector: map[string]string{
					"beta.kubernetes.io/os": "windows",
				},
			},
		}

		pods = append(pods, &pod)
	}

	return pods
}

// getNodeMemory populates a nodeMemory struct with information from the first
func getNodeMemory(f *framework.Framework) nodeMemory {
	selector := labels.Set{"beta.kubernetes.io/os": "windows"}.AsSelector()
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	framework.ExpectNoError(err)

	// Assuming that agent nodes have the same config
	// Make sure there is >0 agent nodes, then use the first one for info
	gomega.Expect(nodeList.Size()).NotTo(gomega.Equal(0))

	ginkgo.By("Getting memory details from node status and kubelet config")

	status := nodeList.Items[0].Status

	nodeName := nodeList.Items[0].ObjectMeta.Name

	kubeletConfig, err := getCurrentKubeletConfig(nodeName)
	framework.ExpectNoError(err)

	systemReserve, err := resource.ParseQuantity(kubeletConfig.SystemReserved["memory"])
	if err != nil {
		systemReserve = *resource.NewQuantity(0, resource.BinarySI)
	}
	kubeReserve, err := resource.ParseQuantity(kubeletConfig.KubeReserved["memory"])
	if err != nil {
		kubeReserve = *resource.NewQuantity(0, resource.BinarySI)
	}
	hardEviction, err := resource.ParseQuantity(kubeletConfig.EvictionHard["memory.available"])
	if err != nil {
		hardEviction = *resource.NewQuantity(0, resource.BinarySI)
	}
	softEviction, err := resource.ParseQuantity(kubeletConfig.EvictionSoft["memory.available"])
	if err != nil {
		softEviction = *resource.NewQuantity(0, resource.BinarySI)
	}

	nodeMem := nodeMemory{
		capacity:      status.Capacity[v1.ResourceMemory],
		allocatable:   status.Allocatable[v1.ResourceMemory],
		systemReserve: systemReserve,
		hardEviction:  hardEviction,
		// these are not implemented and are here for future use - will always be 0 at the moment
		kubeReserve:  kubeReserve,
		softEviction: softEviction,
	}

	return nodeMem
}

// getTotalAllocatableMemory gets the sum of all agent node's allocatable memory
func getTotalAllocatableMemory(f *framework.Framework) *resource.Quantity {
	selector := labels.Set{"beta.kubernetes.io/os": "windows"}.AsSelector()
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	framework.ExpectNoError(err)

	ginkgo.By("Summing allocatable memory across all agent nodes")

	totalAllocatable := resource.NewQuantity(0, resource.BinarySI)

	for _, node := range nodeList.Items {
		status := node.Status

		totalAllocatable.Add(status.Allocatable[v1.ResourceMemory])
	}

	return totalAllocatable
}

// getCurrentKubeletConfig modified from test/e2e_node/util.go
func getCurrentKubeletConfig(nodeName string) (*kubeletconfig.KubeletConfiguration, error) {

	resp := pollConfigz(5*time.Minute, 5*time.Second, nodeName)
	kubeCfg, err := decodeConfigz(resp)
	if err != nil {
		return nil, err
	}
	return kubeCfg, nil
}

// Causes the test to fail, or returns a status 200 response from the /configz endpoint
func pollConfigz(timeout time.Duration, pollInterval time.Duration, nodeName string) *http.Response {
	// start local proxy, so we can send graceful deletion over query string, rather than body parameter
	ginkgo.By("Opening proxy to cluster")
	cmd := framework.KubectlCmd("proxy", "-p", "0")
	stdout, stderr, err := framework.StartCmdAndStreamOutput(cmd)
	framework.ExpectNoError(err)
	defer stdout.Close()
	defer stderr.Close()
	defer framework.TryKill(cmd)
	buf := make([]byte, 128)
	var n int
	n, err = stdout.Read(buf)
	framework.ExpectNoError(err)
	output := string(buf[:n])
	proxyRegexp := regexp.MustCompile("Starting to serve on 127.0.0.1:([0-9]+)")
	match := proxyRegexp.FindStringSubmatch(output)
	gomega.Expect(len(match)).To(gomega.Equal(2))
	port, err := strconv.Atoi(match[1])
	framework.ExpectNoError(err)
	ginkgo.By("http requesting node kubelet /configz")
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/api/v1/nodes/%s/proxy/configz", port, nodeName)
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	client := &http.Client{Transport: tr}
	req, err := http.NewRequest("GET", endpoint, nil)
	framework.ExpectNoError(err)
	req.Header.Add("Accept", "application/json")

	var resp *http.Response
	gomega.Eventually(func() bool {
		resp, err = client.Do(req)
		if err != nil {
			framework.Logf("Failed to get /configz, retrying. Error: %v", err)
			return false
		}
		if resp.StatusCode != 200 {
			framework.Logf("/configz response status not 200, retrying. Response was: %+v", resp)
			return false
		}

		return true
	}, timeout, pollInterval).Should(gomega.Equal(true))
	return resp
}

// Decodes the http response from /configz and returns a kubeletconfig.KubeletConfiguration (internal type).
func decodeConfigz(resp *http.Response) (*kubeletconfig.KubeletConfiguration, error) {
	// This hack because /configz reports the following structure:
	// {"kubeletconfig": {the JSON representation of kubeletconfigv1beta1.KubeletConfiguration}}
	type configzWrapper struct {
		ComponentConfig kubeletconfigv1beta1.KubeletConfiguration `json:"kubeletconfig"`
	}

	configz := configzWrapper{}
	kubeCfg := kubeletconfig.KubeletConfiguration{}

	contentsBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(contentsBytes, &configz)
	if err != nil {
		return nil, err
	}

	err = scheme.Scheme.Convert(&configz.ComponentConfig, &kubeCfg, nil)
	if err != nil {
		return nil, err
	}

	return &kubeCfg, nil
}
