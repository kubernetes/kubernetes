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

package utils

import (
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"regexp"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	"k8s.io/kubernetes/test/e2e/storage/podlogs"
)

// StartPodLogs begins capturing log output and events from current
// and future pods running in the namespace of the framework. That
// ends when the returned cleanup function is called.
//
// The output goes to log files (when using --report-dir, as in the
// CI) or the output stream (otherwise).
func StartPodLogs(ctx context.Context, f *framework.Framework, driverNamespace *v1.Namespace) func() {
	ctx, cancel := context.WithCancel(ctx)
	cs := f.ClientSet

	ns := driverNamespace.Name

	var podEventLog io.Writer = ginkgo.GinkgoWriter
	var podEventLogCloser io.Closer
	to := podlogs.LogOutput{
		StatusWriter: ginkgo.GinkgoWriter,
	}
	if framework.TestContext.ReportDir == "" {
		to.LogWriter = ginkgo.GinkgoWriter
	} else {
		test := ginkgo.CurrentSpecReport()
		// Clean up each individual component text such that
		// it contains only characters that are valid as file
		// name.
		reg := regexp.MustCompile("[^a-zA-Z0-9_-]+")
		var testName []string
		for _, text := range test.ContainerHierarchyTexts {
			testName = append(testName, reg.ReplaceAllString(text, "_"))
			if len(test.LeafNodeText) > 0 {
				testName = append(testName, reg.ReplaceAllString(test.LeafNodeText, "_"))
			}
		}
		// We end the prefix with a slash to ensure that all logs
		// end up in a directory named after the current test.
		//
		// Each component name maps to a directory. This
		// avoids cluttering the root artifact directory and
		// keeps each directory name smaller (the full test
		// name at one point exceeded 256 characters, which was
		// too much for some filesystems).
		logDir := framework.TestContext.ReportDir + "/" + strings.Join(testName, "/")
		to.LogPathPrefix = logDir + "/"

		err := os.MkdirAll(logDir, 0755)
		framework.ExpectNoError(err, "create pod log directory")
		f, err := os.Create(path.Join(logDir, "pod-event.log"))
		framework.ExpectNoError(err, "create pod events log file")
		podEventLog = f
		podEventLogCloser = f
	}
	podlogs.CopyAllLogs(ctx, cs, ns, to)

	// The framework doesn't know about the driver pods because of
	// the separate namespace.  Therefore we always capture the
	// events ourselves.
	podlogs.WatchPods(ctx, cs, ns, podEventLog, podEventLogCloser)

	return cancel
}

// KubeletCommand performs `start`, `restart`, or `stop` on the kubelet running on the node of the target pod and waits
// for the desired statues..
// Allowed kubeletOps are `KStart`, `KStop`, and `KRestart`
func KubeletCommand(ctx context.Context, kOp KubeletOpt, c clientset.Interface, pod *v1.Pod) {
	nodeIP, err := getHostAddress(ctx, c, pod)
	framework.ExpectNoError(err)
	nodeIP = nodeIP + ":22"

	commandTemplate := "systemctl %s kubelet"
	sudoPresent := isSudoPresent(ctx, nodeIP, framework.TestContext.Provider)
	if sudoPresent {
		commandTemplate = "sudo " + commandTemplate
	}

	runCmd := func(cmd string) {
		command := fmt.Sprintf(commandTemplate, cmd)
		framework.Logf("Attempting `%s`", command)
		sshResult, err := e2essh.SSH(ctx, command, nodeIP, framework.TestContext.Provider)
		framework.ExpectNoError(err, fmt.Sprintf("SSH to Node %q errored.", pod.Spec.NodeName))
		e2essh.LogResult(sshResult)
		gomega.Expect(sshResult.Code).To(gomega.BeZero(), "Failed to [%s] kubelet:\n%#v", cmd, sshResult)
	}

	if kOp == KStop || kOp == KRestart {
		runCmd("stop")
	}
	if kOp == KStop {
		return
	}

	if kOp == KStart && getKubeletRunning(ctx, nodeIP) {
		framework.Logf("Kubelet is already running on node %q", pod.Spec.NodeName)
		// Just skip. Or we cannot get a new heartbeat in time.
		return
	}

	node, err := c.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	heartbeatTime := e2enode.GetNodeHeartbeatTime(node)

	runCmd("start")
	// Wait for next heartbeat, which must be sent by the new kubelet process.
	e2enode.WaitForNodeHeartbeatAfter(ctx, c, pod.Spec.NodeName, heartbeatTime, NodeStateTimeout)
	// Then wait until Node with new process becomes Ready.
	if ok := e2enode.WaitForNodeToBeReady(ctx, c, pod.Spec.NodeName, NodeStateTimeout); !ok {
		framework.Failf("Node %s failed to enter Ready state", pod.Spec.NodeName)
	}
}

// getHostAddress gets the node for a pod and returns the first
// address. Returns an error if the node the pod is on doesn't have an
// address.
func getHostAddress(ctx context.Context, client clientset.Interface, p *v1.Pod) (string, error) {
	node, err := client.CoreV1().Nodes().Get(ctx, p.Spec.NodeName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	// Try externalAddress first
	for _, address := range node.Status.Addresses {
		if address.Type == v1.NodeExternalIP {
			if address.Address != "" {
				return address.Address, nil
			}
		}
	}
	// If no externalAddress found, try internalAddress
	for _, address := range node.Status.Addresses {
		if address.Type == v1.NodeInternalIP {
			if address.Address != "" {
				return address.Address, nil
			}
		}
	}

	// If not found, return error
	return "", fmt.Errorf("No address for pod %v on node %v",
		p.Name, p.Spec.NodeName)
}
