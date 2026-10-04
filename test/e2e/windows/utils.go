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
	"context"
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"

	semver "github.com/blang/semver/v4"
)

// waits for a deployment to be created and the desired replicas
// are updated and available, and no old pods are running.
func waitForDeployment(getDeploymentFunc func() (*appsv1.Deployment, error), interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		deployment, err := getDeploymentFunc()
		if err != nil {
			if apierrors.IsNotFound(err) {
				framework.Logf("deployment not found, continue waiting: %s", err)
				return false, nil
			}

			framework.Logf("error while deploying, error %s", err)
			return false, err
		}
		framework.Logf("deployment status %s", &deployment.Status)
		return util.DeploymentComplete(deployment, &deployment.Status), nil
	})
}

// gets the container runtime and version for a node
func getNodeContainerRuntimeAndVersion(n v1.Node) (string, semver.Version, error) {
	containerRuntimeVersionString := n.Status.NodeInfo.DeepCopy().ContainerRuntimeVersion
	parts := strings.Split(containerRuntimeVersionString, "://")

	if len(parts) != 2 {
		return "", semver.Version{}, fmt.Errorf("could not get container runtime and version from '%s'", containerRuntimeVersionString)
	}

	v, err := semver.ParseTolerant(parts[1])
	if err != nil {
		return "", semver.Version{}, err
	}

	return parts[0], v, nil
}

func getRandomUserGrounName() string {
	var letters = []rune("abcdefghijklmnopqrstuvwxya")

	s := make([]rune, 8)
	for i := range s {
		s[i] = letters[rand.Intn(len(letters))]
	}

	return "hpc-" + string(s)
}

func skipUnlessContainerdOneSevenOrGreater(ctx context.Context, f *framework.Framework) {
	ginkgo.By("Ensuring Windows nodes are running containerd v1.7+")
	windowsNode, err := findWindowsNode(ctx, f)
	framework.ExpectNoError(err, "error finding Windows node")
	r, v, err := getNodeContainerRuntimeAndVersion(windowsNode)
	framework.ExpectNoError(err, "error getting node container runtime and version")
	framework.Logf("Got runtime: %s, version %v for node %s", r, v, windowsNode.Name)

	if !strings.EqualFold(r, "containerd") {
		e2eskipper.Skipf("container runtime is not containerd")
	}

	v1dot7 := semver.MustParse("1.7.0-alpha.1")
	if v.LT(v1dot7) {
		e2eskipper.Skipf("container runtime is < 1.7.0")
	}
}

// windowsTestMemorySafetyBuffer is fixed headroom (in bytes) reserved on a
// node above the sum of admitted pod requests + pod.Spec.Overhead. It absorbs
// kubelet/cgroup memory-accounting drift that would otherwise produce flaky
// OutOfmemory admission errors at the capacity boundary.
//
// 256 MiB is empirical, not derived from any kubelet constant: a comfortable
// upper bound on the accounting jitter we observed in practice (single- to
// low-tens of MiB) while remaining a small fraction (~5-10%) of typical
// Windows test-node Allocatable. It can be tightened with measurements if it
// ever becomes a constraint, or replaced with a fraction of Allocatable if
// we ever need to run on nodes too small for a fixed buffer.
const windowsTestMemorySafetyBuffer int64 = 256 * 1024 * 1024

var (
	podOverheadMemoryOnce  sync.Once
	podOverheadMemoryCache int64
	podOverheadMemoryErr   error
)

// detectPodOverheadMemory returns the per-pod memory overhead the cluster's
// admission chain injects (Pod Overhead, KEP-688), in bytes, or 0 if no
// overhead is applied. The error is non-nil only when the probe itself
// could not be completed (e.g., the API server rejected the DryRun create);
// a successful probe with no overhead returns (0, nil).
//
// Result and error are cached for the lifetime of the test process — the
// admission chain is traversed at most once per run. See
// computePodOverheadMemory for probe details.
func detectPodOverheadMemory(ctx context.Context, c clientset.Interface, namespace string) (int64, error) {
	podOverheadMemoryOnce.Do(func() {
		podOverheadMemoryCache, podOverheadMemoryErr = computePodOverheadMemory(ctx, c, namespace)
	})
	return podOverheadMemoryCache, podOverheadMemoryErr
}

// computePodOverheadMemory performs a single DryRun pod create and inspects
// the mutated result for pod.Spec.Overhead. DryRun (rather than reading
// RuntimeClass directly) is needed because admission webhooks may inject
// overhead conditionally on namespace, labels, or other request-scoped data
// that is not visible from a static API read.
//
// Assumes overhead is RuntimeClass-driven (or otherwise static per pod) and
// does not vary with container count, image, or per-container requests; a
// future webhook that scaled overhead by pod shape would invalidate the
// capacity calculations in this package. Callers must use a namespace that
// admits a pause-image pod (current callers set PodSecurity LevelPrivileged).
func computePodOverheadMemory(ctx context.Context, c clientset.Interface, namespace string) (int64, error) {
	probePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "overhead-probe-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "probe",
				Image: imageutils.GetPauseImageName(),
			}},
			NodeSelector: map[string]string{
				"kubernetes.io/os": "windows",
			},
		},
	}
	result, err := c.CoreV1().Pods(namespace).Create(
		ctx, probePod, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}},
	)
	if err != nil {
		return 0, fmt.Errorf("DryRun pod create for overhead detection failed: %w", err)
	}
	if result.Spec.Overhead == nil {
		return 0, nil
	}
	if mem, ok := result.Spec.Overhead[v1.ResourceMemory]; ok {
		framework.Logf("Detected pod overhead memory: %d bytes (per Pod Overhead admission, KEP-688)", mem.Value())
		return mem.Value(), nil
	}
	return 0, nil
}

// waitForNodeMemoryToSettle polls until the node has at least neededBytes
// of free admittable memory (allocatable - sum of admitted-pod requests),
// or the 90s timeout elapses. Useful in [Serial] tests that follow one
// which leaves large pods Terminating — their requests stay counted until
// fully removed.
//
// On timeout the function does not fail the test; it logs a tagged
// "did NOT settle" message so a downstream OutOfmemory failure points
// back at the cause (leftover pods) rather than appearing as an
// unexplained scheduling error.
func waitForNodeMemoryToSettle(ctx context.Context, c clientset.Interface, nodeName string, neededBytes int64) {
	node, err := c.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		framework.Logf("waitForNodeMemoryToSettle: cannot get node %s: %v (skipping wait)", nodeName, err)
		return
	}
	allocatable := node.Status.Allocatable.Memory().Value()
	var lastFree int64
	pollErr := wait.PollUntilContextTimeout(ctx, 5*time.Second, 90*time.Second, true, func(ctx context.Context) (bool, error) {
		existing := sumExistingPodMemoryReservation(ctx, c, nodeName)
		lastFree = allocatable - existing
		framework.Logf("Waiting for node %s memory to settle: existing-reservation=%d free=%d (need >= %d)",
			nodeName, existing, lastFree, neededBytes)
		return lastFree >= neededBytes, nil
	})
	if pollErr == nil {
		framework.Logf("Node %s memory settled: free=%d (need >= %d)", nodeName, lastFree, neededBytes)
		return
	}
	framework.Logf("Node %s memory did NOT settle within 90s: free=%d (need >= %d): %v "+
		"(test will likely fail with OutOfmemory; check for leftover Terminating pods)",
		nodeName, lastFree, neededBytes, pollErr)
}

// sumExistingPodMemoryReservation returns the total memory reservation
// (sum of container memory requests + pod overhead) for all non-terminal pods
// scheduled on the given node. Tests that compute "remaining schedulable
// memory" (e.g., the Memory Limits test) must subtract this from
// allocatable to leave room for DaemonSets and other system pods.
func sumExistingPodMemoryReservation(ctx context.Context, c clientset.Interface, nodeName string) int64 {
	podList, err := c.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		FieldSelector: "spec.nodeName=" + nodeName,
	})
	if err != nil {
		framework.Logf("Could not list pods on node %s: %v (assuming 0 reservation)", nodeName, err)
		return 0
	}
	var total int64
	for _, p := range podList.Items {
		if p.Status.Phase == v1.PodSucceeded || p.Status.Phase == v1.PodFailed {
			continue
		}
		for _, c := range p.Spec.Containers {
			if mem, ok := c.Resources.Requests[v1.ResourceMemory]; ok {
				total += mem.Value()
			}
		}
		if p.Spec.Overhead != nil {
			if oh, ok := p.Spec.Overhead[v1.ResourceMemory]; ok {
				total += oh.Value()
			}
		}
	}
	return total
}
