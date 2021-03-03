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
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	"k8s.io/kubernetes/test/e2e/storage/podlogs"
)

// StartPodLogs begins capturing log output and events from current
// and future pods running in the namespace of the framework. That
// ends when the returned cleanup function is called.
//
// The output goes to log files (when using --report-dir, as in the
// CI) or the output stream (otherwise).
func StartPodLogs(f *framework.Framework, driverNamespace *v1.Namespace) func() {
	ctx, cancel := context.WithCancel(context.Background())
	cs := f.ClientSet

	ns := driverNamespace.Name

	to := podlogs.LogOutput{
		StatusWriter: ginkgo.GinkgoWriter,
	}
	if framework.TestContext.ReportDir == "" {
		to.LogWriter = ginkgo.GinkgoWriter
	} else {
		test := ginkgo.CurrentGinkgoTestDescription()
		// Clean up each individual component text such that
		// it contains only characters that are valid as file
		// name.
		reg := regexp.MustCompile("[^a-zA-Z0-9_-]+")
		var components []string
		for _, component := range test.ComponentTexts {
			components = append(components, reg.ReplaceAllString(component, "_"))
		}
		// We end the prefix with a slash to ensure that all logs
		// end up in a directory named after the current test.
		//
		// Each component name maps to a directory. This
		// avoids cluttering the root artifact directory and
		// keeps each directory name smaller (the full test
		// name at one point exceeded 256 characters, which was
		// too much for some filesystems).
		to.LogPathPrefix = framework.TestContext.ReportDir + "/" +
			strings.Join(components, "/") + "/"
	}
	podlogs.CopyAllLogs(ctx, cs, ns, to)

	// pod events are something that the framework already collects itself
	// after a failed test. Logging them live is only useful for interactive
	// debugging, not when we collect reports.
	if framework.TestContext.ReportDir == "" {
		podlogs.WatchPods(ctx, cs, ns, ginkgo.GinkgoWriter)
	}

	return cancel
}

// KubeletCommand performs `start`, `restart`, or `stop` on the kubelet running on the node of the target pod and waits
// for the desired statues..
// - First issues the command via `systemctl`
// - If `systemctl` returns stderr "command not found, issues the command via `service`
// - If `service` also returns stderr "command not found", the test is aborted.
// Allowed kubeletOps are `KStart`, `KStop`, and `KRestart`
func KubeletCommand(kOp KubeletOpt, c clientset.Interface, pod *v1.Pod) {
	command := ""
	systemctlPresent := false
	kubeletPid := ""

	nodeIP, err := getHostAddress(c, pod)
	framework.ExpectNoError(err)
	nodeIP = nodeIP + ":22"

	framework.Logf("Checking if systemctl command is present")
	sshResult, err := e2essh.SSH("systemctl --version", nodeIP, framework.TestContext.Provider)
	framework.ExpectNoError(err, fmt.Sprintf("SSH to Node %q errored.", pod.Spec.NodeName))
	if !strings.Contains(sshResult.Stderr, "command not found") {
		command = fmt.Sprintf("systemctl %s kubelet", string(kOp))
		systemctlPresent = true
	} else {
		command = fmt.Sprintf("service kubelet %s", string(kOp))
	}

	sudoPresent := isSudoPresent(nodeIP, framework.TestContext.Provider)
	if sudoPresent {
		command = fmt.Sprintf("sudo %s", command)
	}

	if kOp == KRestart {
		kubeletPid = getKubeletMainPid(nodeIP, sudoPresent, systemctlPresent)
	}

	framework.Logf("Attempting `%s`", command)
	sshResult, err = e2essh.SSH(command, nodeIP, framework.TestContext.Provider)
	framework.ExpectNoError(err, fmt.Sprintf("SSH to Node %q errored.", pod.Spec.NodeName))
	e2essh.LogResult(sshResult)
	gomega.Expect(sshResult.Code).To(gomega.BeZero(), "Failed to [%s] kubelet:\n%#v", string(kOp), sshResult)

	if kOp == KStop {
		if ok := e2enode.WaitForNodeToBeNotReady(c, pod.Spec.NodeName, NodeStateTimeout); !ok {
			framework.Failf("Node %s failed to enter NotReady state", pod.Spec.NodeName)
		}
	}
	if kOp == KRestart {
		// Wait for a minute to check if kubelet Pid is getting changed
		isPidChanged := false
		for start := time.Now(); time.Since(start) < 1*time.Minute; time.Sleep(2 * time.Second) {
			kubeletPidAfterRestart := getKubeletMainPid(nodeIP, sudoPresent, systemctlPresent)
			if kubeletPid != kubeletPidAfterRestart {
				isPidChanged = true
				break
			}
		}
		framework.ExpectEqual(isPidChanged, true, "Kubelet PID remained unchanged after restarting Kubelet")
		framework.Logf("Noticed that kubelet PID is changed. Waiting for 30 Seconds for Kubelet to come back")
		time.Sleep(30 * time.Second)
	}
	if kOp == KStart || kOp == KRestart {
		// For kubelet start and restart operations, Wait until Node becomes Ready
		if ok := e2enode.WaitForNodeToBeReady(c, pod.Spec.NodeName, NodeStateTimeout); !ok {
			framework.Failf("Node %s failed to enter Ready state", pod.Spec.NodeName)
		}
	}
}

// getHostAddress gets the node for a pod and returns the first
// address. Returns an error if the node the pod is on doesn't have an
// address.
func getHostAddress(client clientset.Interface, p *v1.Pod) (string, error) {
	node, err := client.CoreV1().Nodes().Get(context.TODO(), p.Spec.NodeName, metav1.GetOptions{})
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

// StopPodAndDependents first tries to log the output of the pod's container,
// then deletes the pod and waits for that to succeed. Also waits for all owned
// resources to be deleted.
func StopPodAndDependents(c clientset.Interface, timeouts *framework.TimeoutContext, pod *v1.Pod) {
	if pod == nil {
		return
	}
	body, err := c.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &v1.PodLogOptions{}).Do(context.TODO()).Raw()
	if err != nil {
		framework.Logf("Error getting logs for pod %s: %v", pod.Name, err)
	} else {
		framework.Logf("Pod %s has the following logs: %s", pod.Name, body)
	}

	// We must wait explicitly for removal of the generic ephemeral volume PVs.
	// For that we must find them first...
	pvs, err := c.CoreV1().PersistentVolumes().List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err, "list PVs")
	var podPVs []v1.PersistentVolume
	for _, pv := range pvs.Items {
		if pv.Spec.ClaimRef == nil ||
			pv.Spec.ClaimRef.Namespace != pod.Namespace {
			continue
		}
		pvc, err := c.CoreV1().PersistentVolumeClaims(pod.Namespace).Get(context.TODO(), pv.Spec.ClaimRef.Name, metav1.GetOptions{})
		if err != nil && apierrors.IsNotFound(err) {
			// Must have been some unrelated PV, otherwise the PVC should exist.
			continue
		}
		framework.ExpectNoError(err, "get PVC")
		if pv.Spec.ClaimRef.UID == pvc.UID && metav1.IsControlledBy(pvc, pod) {
			podPVs = append(podPVs, pv)
		}
	}

	framework.Logf("Deleting pod %q in namespace %q", pod.Name, pod.Namespace)
	deletionPolicy := metav1.DeletePropagationForeground
	err = c.CoreV1().Pods(pod.Namespace).Delete(context.TODO(), pod.Name,
		metav1.DeleteOptions{
			// If the pod is the owner of some resources (like ephemeral inline volumes),
			// then we want to be sure that those are also gone before we return.
			// Blocking pod deletion via metav1.DeletePropagationForeground achieves that.
			PropagationPolicy: &deletionPolicy,
		})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return // assume pod was already deleted
		}
		framework.Logf("pod Delete API error: %v", err)
	}
	framework.Logf("Wait up to %v for pod %q to be fully deleted", timeouts.PodDelete, pod.Name)
	e2epod.WaitForPodNotFoundInNamespace(c, pod.Name, pod.Namespace, timeouts.PodDelete)
	if len(podPVs) > 0 {
		for _, pv := range podPVs {
			// As with CSI inline volumes, we use the pod delete timeout here because conceptually
			// the volume deletion needs to be that fast (whatever "that" is).
			framework.Logf("Wait up to %v for pod PV %s to be fully deleted", timeouts.PodDelete, pv.Name)
			e2epv.WaitForPersistentVolumeDeleted(c, pv.Name, 5*time.Second, timeouts.PodDelete)
		}
	}
}
