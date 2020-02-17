/*
Copyright 2020 The Kubernetes Authors.

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

package upgrades

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"regexp"
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/kubernetes/scheme"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const testCmd string = `
NCPU=$( getconf _NPROCESSORS_ONLN );
MAXCPU=$( expr $NCPU - 1 );
echo ALLOWED=$(grep Cpus_allowed_list /proc/self/status | cut -f2);
for N in $( seq 0 ${MAXCPU} ); do echo "SIBLING$N=$( cat /sys/devices/system/cpu/cpu${N}/topology/core_siblings_list )"; done;
sleep 12h
`

// TopologyManagerUpgradeTest tests that the topology manager provides numa-aligned resources
// before and after a cluster upgrade.
type TopologyManagerUpgradeTest struct {
}

// Name returns the tracking name of the test.
func (TopologyManagerUpgradeTest) Name() string {
	return "topology-manager-upgrade [sig-node]"
}

// Skip returns true when this test can be skipped.
func (TopologyManagerUpgradeTest) Skip(upgCtx UpgradeContext) bool {
	return false
}

// Setup creates a pod requesting aligned resources.
func (t *TopologyManagerUpgradeTest) Setup(f *framework.Framework) {
	ginkgo.By("Creating a Pod with aligned resources")
	node := getTopologyManagerSingleNumaNodePolicyEnabledNode(f)
	if node == nil {
		framework.Logf("No suitable node configured with Topology Manager before upgrade")
		return
	}

	name := "pod-before-" + string(uuid.NewUUID())
	err := runAndValidate(f, name, node)
	framework.ExpectNoError(err)
}

// Test waits for the upgrade to complete, and then verifies that it is possible to run
// a pod requesting aligned resources after the upgrade
func (t *TopologyManagerUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	<-done
	if !isRelevantUpgrade(upgrade) {
		return
	}

	ginkgo.By("Verifying it is still possible to create a Pod with aligned resources")
	node := getTopologyManagerSingleNumaNodePolicyEnabledNode(f)
	if node == nil {
		framework.Logf("No suitable node configured with Topology Manager before upgrade")
		return
	}

	name := "pod-after-" + string(uuid.NewUUID())
	err := runAndValidate(f, name, node)
	framework.ExpectNoError(err)
}

// Teardown cleans up any remaining resources.
func (t *TopologyManagerUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

func runAndValidate(f *framework.Framework, name string, node *v1.Node) error {
	pod := makePod(node.ObjectMeta.Name, f.Namespace.Name, name, testCmd)
	pod = f.PodClient().CreateSync(pod)

	output, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
	framework.ExpectNoError(err, fmt.Sprintf("Failed to get pod %q output", name))

	return validatePodOutput(output)
}

func validatePodOutput(output string) error {
	cpuMap := make(map[string][]int)
	re := regexp.MustCompile(`(\w*)=(\w*)`)
	for _, match := range re.FindAllStringSubmatch(output, -1) {
		if len(match) != 3 {
			framework.Logf("unexpected match %v", match)
			continue
		}

		cset, err := cpuset.Parse(match[2])
		framework.ExpectNoError(err, "parsing %v", match)
		// ToSlice() returns a sorted slice
		cpuMap[match[1]] = cset.ToSlice()
	}

	allowed, ok := cpuMap["ALLOWED"]
	if !ok {
		return fmt.Errorf("no allowed CPUs found in output")
	}

	if len(allowed) < 2 {
		return fmt.Errorf("the test requires at least two CPUs requested for the pod")
	}

	for i := 1; i < len(allowed); i++ {
		cpuNumPrev := allowed[i-1]
		siblingsPrev, ok := cpuMap[fmt.Sprintf("SIBLING%d", cpuNumPrev)]
		if !ok {
			return fmt.Errorf("Unknown siblings for cpu %d", cpuNumPrev)
		}

		cpuNum := allowed[i]
		siblings, ok := cpuMap[fmt.Sprintf("SIBLING%d", cpuNum)]
		if !ok {
			return fmt.Errorf("Unknown siblings for cpu %d", cpuNum)
		}

		// per https://www.kernel.org/doc/Documentation/cputopology.txt , cpu_siblings_list is
		// "(the) human-readable list of cpuX's hardware threads within the same physical_package_id."
		// hence, if two entries have the same ordered list of siblings, they are on the same
		// physical package, thus in the same NUMA node.
		if !reflect.DeepEqual(siblingsPrev, siblings) {
			return fmt.Errorf("disaligned cpus %d and %d (%v and %v)", cpuNumPrev, cpuNum, siblingsPrev, siblings)
		}
	}
	return nil
}

func isRelevantUpgrade(upgrade UpgradeType) bool {
	return upgrade == NodeUpgrade || upgrade == ClusterUpgrade
}

func getTopologyManagerSingleNumaNodePolicyEnabledNode(f *framework.Framework) *v1.Node {
	selector := labels.Set{"node-role.kubernetes.io/worker=": ""}.AsSelector()
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	framework.ExpectNoError(err)

	ginkgo.By("Finding a worker node with Topology Manager configured")

	for _, node := range nodeList.Items {
		kubeletConfig, err := getCurrentKubeletConfig(node.ObjectMeta.Name, f.Namespace.Name)
		framework.ExpectNoError(err)

		framework.Logf("node %q TopologyManagerPolicy %q", node.ObjectMeta.Name, kubeletConfig.TopologyManagerPolicy)
		if kubeletConfig.TopologyManagerPolicy == topologymanager.PolicySingleNumaNode {
			return &node
		}
	}

	return nil
}

// TODO: this dupes test/e2e/windows/memory_limits.go. How can we factor this code out?
// getCurrentKubeletConfig modified from test/e2e_node/util.go
func getCurrentKubeletConfig(nodeName, namespace string) (*kubeletconfig.KubeletConfiguration, error) {

	resp := pollConfigz(5*time.Minute, 5*time.Second, nodeName, namespace)
	kubeCfg, err := decodeConfigz(resp)
	if err != nil {
		return nil, err
	}
	return kubeCfg, nil
}

// Causes the test to fail, or returns a status 200 response from the /configz endpoint
func pollConfigz(timeout time.Duration, pollInterval time.Duration, nodeName, namespace string) *http.Response {
	// start local proxy, so we can send graceful deletion over query string, rather than body parameter
	ginkgo.By("Opening proxy to cluster")
	tk := e2ekubectl.NewTestKubeconfig(framework.TestContext.CertDir, framework.TestContext.Host, framework.TestContext.KubeConfig, framework.TestContext.KubeContext, framework.TestContext.KubectlPath, namespace)
	cmd := tk.KubectlCmd("proxy", "-p", "0")
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
	framework.ExpectEqual(len(match), 2)
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

// cpusetCmd := fmt.Sprintf("grep Cpus_allowed_list /proc/self/status | cut -f2 && sleep 1d")
func makePod(nodeName, namespace, podName, cmd string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  podName,
					Image: imageutils.GetE2EImage(imageutils.BusyBox),
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("2000m"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("100Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("2000m"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("100Mi"),
						},
					},
					Command: []string{"sh", "-c", cmd},
				},
			},
			NodeName: nodeName,
		},
	}
}
