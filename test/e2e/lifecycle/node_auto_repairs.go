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

package lifecycle

import (
	"fmt"
	"os/exec"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	defaultTimeout = 3 * time.Minute
	repairTimeout  = 20 * time.Minute
)

var _ = SIGDescribe("Node Auto Repairs [Slow] [Disruptive]", func() {
	f := framework.NewDefaultFramework("lifecycle")
	var c clientset.Interface
	var originalNodes map[string]string
	var nodeCount int

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gke")
		c = f.ClientSet
		nodeCount = 0
		originalNodes = make(map[string]string)
		for _, groupName := range strings.Split(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") {
			glog.Infof("Processing group %s", groupName)
			nodes, err := framework.GetGroupNodes(groupName)
			framework.ExpectNoError(err)
			for _, node := range nodes {
				nodeReady, err := isNodeReady(c, node)
				framework.ExpectNoError(err)
				Expect(nodeReady).To(Equal(true))
				originalNodes[groupName] = node
				nodeCount++
			}
		}
		glog.Infof("Number of nodes %d", nodeCount)
	})

	AfterEach(func() {
		framework.SkipUnlessProviderIs("gke")
		By(fmt.Sprintf("Restoring initial size of the cluster"))
		for groupName, nodeName := range originalNodes {
			nodeReady, err := isNodeReady(c, nodeName)
			framework.ExpectNoError(err)
			if !nodeReady {
				framework.ExpectNoError(recreateNode(nodeName, groupName))
			}
		}
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, defaultTimeout))
	})

	It("should repair node [Feature:NodeAutoRepairs]", func() {
		framework.SkipUnlessProviderIs("gke")
		framework.ExpectNoError(enableAutoRepair("default-pool"))
		defer disableAutoRepair("default-pool")
		readyNodes := getReadyNodes(c)
		Expect(len(readyNodes)).NotTo(Equal(0))
		nodeName := readyNodes[0].Name
		framework.ExpectNoError(stopKubeletOnNode(nodeName))
		By("Wait till node is unready.")
		Expect(framework.WaitForNodeToBeNotReady(c, nodeName, defaultTimeout)).To(Equal(true))
		By("Wait till node is repaired.")
		Expect(framework.WaitForNodeToBeReady(c, nodeName, repairTimeout)).To(Equal(true))
	})
})

func execCmd(args ...string) *exec.Cmd {
	glog.Infof("Executing: %s", strings.Join(args, " "))
	return exec.Command(args[0], args[1:]...)
}

func getReadyNodes(c clientset.Interface) []v1.Node {
	nodeList := framework.GetReadySchedulableNodesOrDie(c)
	return nodeList.Items
}

func enableAutoRepair(nodePool string) error {
	glog.Infof("Using gcloud to enable auto repair for pool %s", nodePool)
	output, err := execCmd("gcloud", "beta", "container", "node-pools", "update", nodePool,
		"--enable-autorepair",
		"--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zone="+framework.TestContext.CloudConfig.Zone,
		"--cluster="+framework.TestContext.CloudConfig.Cluster).CombinedOutput()
	if err != nil {
		glog.Errorf("Failed to enable auto repair: %s", string(output))
		return fmt.Errorf("failed to enable auto repair: %v", err)
	}
	return nil
}

func disableAutoRepair(nodePool string) error {
	glog.Infof("Using gcloud to disable auto repair for pool %s", nodePool)
	output, err := execCmd("gcloud", "beta", "container", "node-pools", "update", nodePool,
		"--no-enable-autorepair",
		"--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zone="+framework.TestContext.CloudConfig.Zone,
		"--cluster="+framework.TestContext.CloudConfig.Cluster).CombinedOutput()
	if err != nil {
		glog.Errorf("Failed to disable auto repair: %s", string(output))
		return fmt.Errorf("failed to disable auto repair: %v", err)
	}
	return nil
}

func stopKubeletOnNode(node string) error {
	glog.Infof("Using gcloud to stop Kublet on node %s", node)
	output, err := execCmd("gcloud", "compute", "ssh", node,
		"--command=sudo systemctl stop kubelet-monitor.service && sudo systemctl stop kubelet.service",
		"--zone="+framework.TestContext.CloudConfig.Zone).CombinedOutput()
	if err != nil {
		glog.Errorf("Failed to stop Kubelet: %v", string(output))
		return fmt.Errorf("failed to stop Kubelet: %v", err)
	}
	return nil
}

func recreateNode(nodeName string, groupName string) error {
	glog.Infof("Using gcloud to recreate node %s", nodeName)
	//gcloud compute instance-groups managed recreate-instances gke-oneoff-default-pool-e4383993-grp --instances=gke-oneoff-default-pool-e4383993-81jm --zone=us-central1-c
	output, err := execCmd("gcloud", "compute", "instance-groups", "managed", "recreate-instances", groupName,
		"--instances="+nodeName,
		"--zone="+framework.TestContext.CloudConfig.Zone).CombinedOutput()
	if err != nil {
		glog.Errorf("Failed to recreate node: %s", string(output))
		return fmt.Errorf("failed to recreate node: %v", err)
	}
	return nil
}

func isNodeReady(c clientset.Interface, nodeName string) (bool, error) {
	glog.Infof("Check if node %s is ready ", nodeName)
	node, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	if err != nil {
		return false, err
	}
	result := framework.IsNodeConditionSetAsExpected(node, v1.NodeReady, true)
	glog.Infof("Node %s is ready: %t", nodeName, result)
	return result, nil
}
