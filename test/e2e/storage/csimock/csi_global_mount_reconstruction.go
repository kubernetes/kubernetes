/*
Copyright 2026 The Kubernetes Authors.

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

package csimock

import (
	"context"
	"fmt"
	"path"
	"strings"
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

// These tests stop and start kubelet over SSH, so the provider check matches
// the one in csi_kubelet_restart.go.
var _ = utils.SIGDescribe("CSI Mock global mount reconstruction", framework.WithFeatureGate(features.CSIGlobalMountReconstruction), framework.WithSerial(), framework.WithDisruptive(), framework.WithProvider(framework.ProvidersWithSSH...), func() {
	f := framework.NewDefaultFramework("csi-mock-global-mount-reconstruction")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessSSHKeyPresent()
	})

	ginkgo.DescribeTable("should call NodeUnstage after kubelet restart when the pod is deleted", func(ctx context.Context, podDirGone bool) {
		trackedCalls := []string{
			"NodeStageVolume",
			"NodeUnstageVolume",
		}
		// Fail NodeUnstage until kubelet is stopped, so the volume is still
		// staged when the new kubelet starts with no pod to find it through.
		var failUnstage atomic.Bool
		failUnstage.Store(true)
		hooks := createPreHook("NodeUnstageVolume", func(counter int64) error {
			if failUnstage.Load() {
				return status.Error(codes.DeadlineExceeded, "fake transient error")
			}
			return nil
		})
		m.init(ctx, testParameters{
			disableAttach:  true,
			registerDriver: true,
			hooks:          hooks,
		})
		ginkgo.DeferCleanup(m.cleanup)

		_, claim, pod := m.createPod(ctx, pvcReference)
		if pod == nil {
			return
		}
		// Wait for PVC to get bound to make sure the CSI driver is fully started.
		err := e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, f.ClientSet, f.Namespace.Name, claim.Name, time.Second, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err, "while waiting for PVC to get provisioned")
		err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
		framework.ExpectNoError(err, "while waiting for the pod to start")
		// KubeletCommand needs the node the pod was scheduled to.
		pod, err = m.cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "while getting the pod")

		ginkgo.By("Deleting the pod and waiting for the failed NodeUnstage")
		err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
		framework.ExpectNoError(err, "while deleting the pod")
		waitForCSICalls(ctx, trackedCalls, []csiCall{
			{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
			{expectedMethod: "NodeUnstageVolume", expectedError: codes.DeadlineExceeded},
		}, csiUnstageWaitTimeout, m.driver.GetCalls)

		node, err := m.cs.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err, "while getting the node")
		podDir := path.Join(framework.TestContext.KubeletRootDir, "pods", string(pod.UID))
		if podDirGone {
			// The old kubelet removes the directory of a deleted pod once its
			// volumes are unpublished, and the global mount stays staged.
			ginkgo.By("Waiting for kubelet to remove the pod directory")
			err = wait.PollUntilContextTimeout(ctx, time.Second, csiUnstageWaitTimeout, true, func(ctx context.Context) (bool, error) {
				return !podDirExists(ctx, node, podDir), nil
			})
			framework.ExpectNoError(err, "while waiting for kubelet to remove %s", podDir)
		}

		ginkgo.By("Stopping kubelet while the volume is still staged")
		utils.KubeletCommand(ctx, utils.KStop, f.ClientSet, pod)
		ginkgo.DeferCleanup(utils.KubeletCommand, utils.KStart, f.ClientSet, pod)
		failUnstage.Store(false)
		if !podDirGone {
			// A kubelet that stops before it removes the directory of a deleted
			// pod leaves it with no volume in it. Make sure the new kubelet
			// starts with that directory, whatever the old one did.
			ginkgo.By("Leaving an empty pod directory on the node")
			framework.Logf("Pod directory %s left by the old kubelet: %v", podDir, podDirExists(ctx, node, podDir))
			createCmd := fmt.Sprintf("sudo -n true 2>/dev/null && S=sudo; $S mkdir -p %q", podDir)
			_, err = e2essh.IssueSSHCommandWithResult(ctx, createCmd, framework.TestContext.Provider, node)
			framework.ExpectNoError(err, "while creating %s", podDir)
		}
		gomega.Expect(podDirExists(ctx, node, podDir)).To(gomega.Equal(!podDirGone), "pod directory %s on the node", podDir)

		// CSINode keeps the driver across a kubelet restart, so it cannot tell
		// when the new kubelet registered the driver. Kubelet calls NodeGetInfo
		// when it registers the driver, so wait for a new call instead.
		nodeGetInfoCalls := countCSICalls(ctx, "NodeGetInfo", m.driver.GetCalls)

		ginkgo.By("Starting kubelet and waiting for the driver to register again")
		utils.KubeletCommand(ctx, utils.KStart, f.ClientSet, pod)
		err = wait.PollUntilContextTimeout(ctx, time.Second, csiPodRunningTimeout, true, func(ctx context.Context) (bool, error) {
			return countCSICalls(ctx, "NodeGetInfo", m.driver.GetCalls) > nodeGetInfoCalls, nil
		})
		framework.ExpectNoError(err, "while waiting for kubelet to register the CSI driver again")

		ginkgo.By("Waiting for NodeUnstage of the reconstructed global mount")
		// The new kubelet retries the unstage with exponential backoff, up to
		// about two minutes, from before the driver registers again, so the
		// next try can come well over a minute after the registration.
		waitForCSICalls(ctx, trackedCalls, []csiCall{
			{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
			{expectedMethod: "NodeUnstageVolume", expectedError: codes.DeadlineExceeded},
			{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
		}, csiPodRunningTimeout, m.driver.GetCalls)
	},
		ginkgo.Entry("and its directory is gone", true),
		ginkgo.Entry("and an empty pod directory is left", false),
	)
})

// podDirExists reports whether dir exists on node.
func podDirExists(ctx context.Context, node *v1.Node, dir string) bool {
	cmd := fmt.Sprintf("sudo -n true 2>/dev/null && S=sudo; if $S test -e %q; then echo present; else echo absent; fi", dir)
	result, err := e2essh.IssueSSHCommandWithResult(ctx, cmd, framework.TestContext.Provider, node)
	framework.ExpectNoError(err, "while checking %s", dir)
	return strings.TrimSpace(result.Stdout) == "present"
}

// waitForCSICalls waits until the mock driver received all expectedCalls, in order.
func waitForCSICalls(ctx context.Context, trackedCalls []string, expectedCalls []csiCall, timeout time.Duration, getCalls func(ctx context.Context) ([]drivers.MockCSICall, error)) {
	err := wait.PollUntilContextTimeout(ctx, time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		_, index, err := compareCSICalls(ctx, trackedCalls, expectedCalls, getCalls)
		if err != nil {
			return true, err
		}
		return index == len(expectedCalls), nil
	})
	framework.ExpectNoError(err, "while waiting for CSI calls %v", expectedCalls)
}

// countCSICalls returns how many calls of method the mock driver received.
func countCSICalls(ctx context.Context, method string, getCalls func(ctx context.Context) ([]drivers.MockCSICall, error)) int {
	calls, err := getCalls(ctx)
	if err != nil {
		framework.Logf("intermittent (?) log retrieval error, proceeding without output: %v", err)
		return 0
	}
	count := 0
	for _, c := range calls {
		if c.Method == method {
			count++
		}
	}
	return count
}
