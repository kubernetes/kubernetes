/*
Copyright 2014 The Kubernetes Authors.

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

package output

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	gomegatypes "github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubectl/pkg/util/podutils"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

// DEPRECATED constants. Use the timeouts in framework.Framework instead.
const (
	// Poll is how often to Poll pods, nodes and claims.
	Poll = 2 * time.Second
)

// LookForStringInPodExec looks for the given string in the output of a command
// executed in the first container of specified pod.
func LookForStringInPodExec(ns, podName string, command []string, expectedString string, timeout time.Duration) (result string, err error) {
	return LookForStringInPodExecToContainer(ns, podName, "", command, expectedString, timeout)
}

// LookForStringInPodExecToContainer looks for the given string in the output of a
// command executed in specified pod container, or first container if not specified.
func LookForStringInPodExecToContainer(ns, podName, containerName string, command []string, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		args := []string{"exec", podName, fmt.Sprintf("--namespace=%v", ns)}
		if len(containerName) > 0 {
			args = append(args, fmt.Sprintf("--container=%s", containerName))
		}
		args = append(args, "--")
		args = append(args, command...)
		return e2ekubectl.RunKubectlOrDie(ns, args...)
	})
}

// lookForString looks for the given string in the output of fn, repeatedly calling fn until
// the timeout is reached or the string is found. Returns last log and possibly
// error if the string was not found.
func lookForString(expectedString string, timeout time.Duration, fn func() string) (result string, err error) {
	for t := time.Now(); time.Since(t) < timeout; time.Sleep(Poll) {
		result = fn()
		if strings.Contains(result, expectedString) {
			return
		}
	}
	err = fmt.Errorf("Failed to find \"%s\", last result: \"%s\"", expectedString, result)
	return
}

// RunHostCmd runs the given cmd in the context of the given pod using `kubectl exec`
// inside of a shell.
func RunHostCmd(ns, name, cmd string) (string, error) {
	return e2ekubectl.RunKubectl(ns, "exec", name, "--", "/bin/sh", "-x", "-c", cmd)
}

// RunHostCmdWithFullOutput runs the given cmd in the context of the given pod using `kubectl exec`
// inside of a shell. It will also return the command's stderr.
func RunHostCmdWithFullOutput(ns, name, cmd string) (string, string, error) {
	return e2ekubectl.RunKubectlWithFullOutput(ns, "exec", name, "--", "/bin/sh", "-x", "-c", cmd)
}

// RunHostCmdOrDie calls RunHostCmd and dies on error.
func RunHostCmdOrDie(ns, name, cmd string) string {
	stdout, err := RunHostCmd(ns, name, cmd)
	framework.Logf("stdout: %v", stdout)
	framework.ExpectNoError(err)
	return stdout
}

// RunHostCmdWithRetries calls RunHostCmd and retries all errors
// until it succeeds or the specified timeout expires.
// This can be used with idempotent commands to deflake transient Node issues.
func RunHostCmdWithRetries(ns, name, cmd string, interval, timeout time.Duration) (string, error) {
	start := time.Now()
	for {
		out, err := RunHostCmd(ns, name, cmd)
		if err == nil {
			return out, nil
		}
		if elapsed := time.Since(start); elapsed > timeout {
			return out, fmt.Errorf("RunHostCmd still failed after %v: %w", elapsed, err)
		}
		framework.Logf("Waiting %v to retry failed RunHostCmd: %v", interval, err)
		time.Sleep(interval)
	}
}

// LookForStringInLog looks for the given string in the log of a specific pod container
func LookForStringInLog(ns, podName, container, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		return e2ekubectl.RunKubectlOrDie(ns, "logs", podName, container)
	})
}

// LookForStringInLogWithoutKubectl looks for the given string in the log of a specific pod container
func LookForStringInLogWithoutKubectl(ctx context.Context, client clientset.Interface, ns string, podName string, container string, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		podLogs, err := e2epod.GetPodLogs(ctx, client, ns, podName, container)
		framework.ExpectNoError(err)
		return podLogs
	})
}

// CreateEmptyFileOnPod creates empty file at given path on the pod.
func CreateEmptyFileOnPod(namespace string, podName string, filePath string) error {
	_, err := e2ekubectl.RunKubectl(namespace, "exec", podName, "--", "/bin/sh", "-c", fmt.Sprintf("touch %s", filePath))
	return err
}

// DumpDebugInfo dumps debug info of tests.
func DumpDebugInfo(ctx context.Context, c clientset.Interface, ns string) {
	sl, _ := c.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{LabelSelector: labels.Everything().String()})
	for _, s := range sl.Items {
		desc, _ := e2ekubectl.RunKubectl(ns, "describe", "po", s.Name)
		framework.Logf("\nOutput of kubectl describe %v:\n%v", s.Name, desc)

		l, _ := e2ekubectl.RunKubectl(ns, "logs", s.Name, "--tail=100")
		framework.Logf("\nLast 100 log lines of %v:\n%v", s.Name, l)
	}
}

// MatchContainerOutput creates a pod and waits for all it's containers to exit with success.
// It then tests that the matcher with each expectedOutput matches the output of the specified container.
func MatchContainerOutput(
	ctx context.Context,
	f *framework.Framework,
	pod *v1.Pod,
	containerName string,
	expectedOutput []string,
	matcher func(string, ...interface{}) gomegatypes.GomegaMatcher) error {
	ns := pod.ObjectMeta.Namespace
	if ns == "" {
		ns = f.Namespace.Name
	}
	podClient := e2epod.PodClientNS(f, ns)

	createdPod := podClient.Create(ctx, pod)
	defer func() {
		ginkgo.By("delete the pod")
		podClient.DeleteSync(ctx, createdPod.Name, metav1.DeleteOptions{}, e2epod.DefaultPodDeletionTimeout)
	}()

	// Wait for client pod to complete.
	podErr := e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, f.ClientSet, createdPod.Name, ns, f.Timeouts.PodStart)

	// Grab its logs.  Get host first.
	podStatus, err := podClient.Get(ctx, createdPod.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get pod status: %w", err)
	}

	if podErr != nil {
		// Pod failed. Dump all logs from all containers to see what's wrong
		_ = podutils.VisitContainers(&podStatus.Spec, podutils.AllContainers, func(c *v1.Container, containerType podutils.ContainerType) bool {
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, ns, podStatus.Name, c.Name)
			if err != nil {
				framework.Logf("Failed to get logs from node %q pod %q container %q: %v",
					podStatus.Spec.NodeName, podStatus.Name, c.Name, err)
			} else {
				framework.Logf("Output of node %q pod %q container %q: %s", podStatus.Spec.NodeName, podStatus.Name, c.Name, logs)
			}
			return true
		})
		return fmt.Errorf("expected pod %q success: %v", createdPod.Name, podErr)
	}

	framework.Logf("Trying to get logs from node %s pod %s container %s: %v",
		podStatus.Spec.NodeName, podStatus.Name, containerName, err)

	// Sometimes the actual containers take a second to get started, try to get logs for 60s
	logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, ns, podStatus.Name, containerName)
	if err != nil {
		framework.Logf("Failed to get logs from node %q pod %q container %q. %v",
			podStatus.Spec.NodeName, podStatus.Name, containerName, err)
		return fmt.Errorf("failed to get logs from %s for %s: %w", podStatus.Name, containerName, err)
	}

	for _, expected := range expectedOutput {
		m := matcher(expected)
		matches, err := m.Match(logs)
		if err != nil {
			return fmt.Errorf("expected %q in container output: %w", expected, err)
		} else if !matches {
			return fmt.Errorf("expected %q in container output: %s", expected, m.FailureMessage(logs))
		}
	}

	return nil
}

// TestContainerOutput runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status, and tests
// the specified container log against the given expected output using a substring matcher.
func TestContainerOutput(ctx context.Context, f *framework.Framework, scenarioName string, pod *v1.Pod, containerIndex int, expectedOutput []string) {
	TestContainerOutputMatcher(ctx, f, scenarioName, pod, containerIndex, expectedOutput, gomega.ContainSubstring)
}

// TestContainerOutputRegexp runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status, and tests
// the specified container log against the given expected output using a regexp matcher.
func TestContainerOutputRegexp(ctx context.Context, f *framework.Framework, scenarioName string, pod *v1.Pod, containerIndex int, expectedOutput []string) {
	TestContainerOutputMatcher(ctx, f, scenarioName, pod, containerIndex, expectedOutput, gomega.MatchRegexp)
}

// TestContainerOutputMatcher runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status, and tests
// the specified container log against the given expected output using the given matcher.
func TestContainerOutputMatcher(ctx context.Context, f *framework.Framework,
	scenarioName string,
	pod *v1.Pod,
	containerIndex int,
	expectedOutput []string,
	matcher func(string, ...interface{}) gomegatypes.GomegaMatcher) {
	ginkgo.By(fmt.Sprintf("Creating a pod to test %v", scenarioName))
	if containerIndex < 0 || containerIndex >= len(pod.Spec.Containers) {
		framework.Failf("Invalid container index: %d", containerIndex)
	}
	framework.ExpectNoError(MatchContainerOutput(ctx, f, pod, pod.Spec.Containers[containerIndex].Name, expectedOutput, matcher))
}
