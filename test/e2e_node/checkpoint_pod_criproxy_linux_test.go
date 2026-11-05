/*
Copyright The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletconfigpaths "k8s.io/kubernetes/pkg/kubelet/kubeletconfig"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e_node/criproxy"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Checkpoint Pod with CRI proxy", feature.PodLevelCheckpointRestore, feature.CriProxy, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("checkpoint-pod-cri-proxy")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.BeforeEach(func(ctx context.Context) {
		if e2eCriProxy == nil {
			ginkgo.Skip("Requires --cri-proxy-enabled=true")
		}
		framework.ExpectNoError(resetCRIProxyInjector(e2eCriProxy))
		ginkgo.DeferCleanup(func() error { return resetCRIProxyInjector(e2eCriProxy) })
		runtimeStatus, err := e2eCriProxy.Status(ctx, &runtimeapi.StatusRequest{})
		framework.ExpectNoError(err)
		if !runtimeStatus.GetFeatures().GetPodCheckpointRestoreCancellation() {
			ginkgo.Skip("Requires runtime support for confirmed Pod checkpoint cancellation")
		}
	})
	tempSetCurrentKubeletConfig(f, func(_ context.Context, config *kubeletconfig.KubeletConfiguration) {
		config.PodCheckpointTimeout = metav1.Duration{Duration: 30 * time.Second}
	})

	ginkgo.It("should fail the checkpoint when the runtime returns Unimplemented", func(ctx context.Context) {
		pod := e2epod.NewPodClient(f).CreateSync(ctx, podCheckpointSource(f.Namespace.Name, "checkpoint-unimplemented"))
		const message = "injected CheckpointPod is unavailable"
		framework.ExpectNoError(addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
			if apiName == criproxy.CheckpointPod {
				return status.Error(codes.Unimplemented, message)
			}
			return nil
		}))

		checkpoint := createPodCheckpoint(ctx, f, newPodCheckpoint(pod, "checkpoint-unimplemented-cp"))
		checkpoint = waitForPodCheckpoint(ctx, f, checkpoint.Name)
		condition := expectPodCheckpointFailed(ctx, f, pod, checkpoint)
		gomega.Expect(condition.Message).To(gomega.ContainSubstring("code = Unimplemented"))
		gomega.Expect(condition.Message).To(gomega.ContainSubstring(message))
	})

	ginkgo.It("should fail the checkpoint when its CRI deadline expires", func(ctx context.Context) {
		pod := e2epod.NewPodClient(f).CreateSync(ctx, podCheckpointSource(f.Namespace.Name, "checkpoint-timeout"))
		entered, _ := holdPodOperationRPC(ctx, criproxy.CheckpointPod, fmt.Errorf("injected checkpoint failure"))
		checkpoint := newPodCheckpoint(pod, "checkpoint-timeout-cp")
		checkpoint.Spec.TimeoutSeconds = new(int32(2))
		checkpoint = createPodCheckpoint(ctx, f, checkpoint)
		gomega.Eventually(ctx, entered, 30*time.Second).Should(gomega.BeClosed(), "checkpoint must reach the runtime")

		checkpoint = waitForPodCheckpoint(ctx, f, checkpoint.Name)
		condition := expectPodCheckpointFailed(ctx, f, pod, checkpoint)
		gomega.Expect(condition.Message).To(gomega.Or(gomega.ContainSubstring("DeadlineExceeded"), gomega.ContainSubstring("deadline exceeded")))
	})

	ginkgo.It("should suspend all Pod probes until failed checkpoint cleanup is acknowledged", func(ctx context.Context) {
		pod := e2epod.NewPodClient(f).CreateSync(ctx, podCheckpointSource(f.Namespace.Name, "checkpoint-probes"))
		for _, container := range pod.Spec.Containers {
			gomega.Eventually(ctx, func() error {
				counts, err := podCheckpointProbeCounts(f, pod, container.Name)
				if err != nil {
					return err
				}
				if counts[0] == 0 || counts[1] == 0 {
					return fmt.Errorf("both probe types must run in container %q before checkpoint", container.Name)
				}
				return nil
			}, time.Minute, time.Second).Should(gomega.Succeed())
		}

		entered, release := holdPodOperationRPC(ctx, criproxy.CheckpointPod, fmt.Errorf("injected checkpoint failure"))
		cleanupEntered, releaseCleanup := holdPodOperationRPC(ctx, criproxy.CancelPodOperation, nil)
		checkpoint := createPodCheckpoint(ctx, f, newPodCheckpoint(pod, "checkpoint-probes-cp"))
		gomega.Eventually(ctx, entered, 30*time.Second).Should(gomega.BeClosed(), "checkpoint must reach the runtime after draining probes")
		framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "checkpoint condition set", 10*time.Second, func(current *v1.Pod) (bool, error) {
			for _, condition := range current.Status.Conditions {
				if condition.Type == v1.PodCheckpointing && condition.Status == v1.ConditionTrue {
					return true, nil
				}
			}
			return false, nil
		}))
		before := map[string][2]int{}
		for _, container := range pod.Spec.Containers {
			counts, err := podCheckpointProbeCounts(f, pod, container.Name)
			framework.ExpectNoError(err)
			before[container.Name] = counts
		}

		ginkgo.By("checking multiple probe periods while the checkpoint RPC is held")
		gomega.Consistently(ctx, func() error {
			for _, container := range pod.Spec.Containers {
				counts, err := podCheckpointProbeCounts(f, pod, container.Name)
				if err != nil {
					return err
				}
				if counts != before[container.Name] {
					return fmt.Errorf("probes ran in container %q during checkpoint: got %v, want %v", container.Name, counts, before[container.Name])
				}
			}
			return nil
		}, 3*time.Second, 500*time.Millisecond).Should(gomega.Succeed())

		ginkgo.By("returning the capture failure while withholding cleanup acknowledgment")
		release()
		gomega.Eventually(ctx, cleanupEntered, 10*time.Second).Should(gomega.BeClosed())
		checkpointPath := filepath.Join(framework.TestContext.KubeletRootDir, kubeletconfigpaths.DefaultKubeletPodCheckpointsDirName, "checkpoint-"+string(checkpoint.UID))
		gomega.Consistently(ctx, func() error {
			if _, err := os.Stat(checkpointPath); err != nil {
				return fmt.Errorf("checkpoint output was removed before cleanup acknowledgment: %w", err)
			}
			current, err := f.ClientSet.NodeV1alpha1().PodCheckpoints(checkpoint.Namespace).Get(ctx, checkpoint.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			condition := apimeta.FindStatusCondition(current.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
			if condition == nil || condition.Status != metav1.ConditionFalse || condition.Reason != nodev1alpha1.PodCheckpointReasonInProgress {
				return fmt.Errorf("checkpoint stopped reporting in progress before cleanup acknowledgment: %v", condition)
			}
			source, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			checkpointing := false
			for _, condition := range source.Status.Conditions {
				if condition.Type == v1.PodCheckpointing && condition.Status == v1.ConditionTrue {
					checkpointing = true
				}
			}
			if !checkpointing {
				return fmt.Errorf("source Pod stopped reporting Checkpointing before cleanup acknowledgment")
			}
			for _, container := range pod.Spec.Containers {
				counts, err := podCheckpointProbeCounts(f, pod, container.Name)
				if err != nil {
					return err
				}
				if counts != before[container.Name] {
					return fmt.Errorf("probes resumed in container %q before cleanup acknowledgment: got %v, want %v", container.Name, counts, before[container.Name])
				}
			}
			return nil
		}, 3*time.Second, 500*time.Millisecond).Should(gomega.Succeed())

		ginkgo.By("acknowledging runtime cleanup and checking that all probes resume")
		releaseCleanup()
		checkpoint = waitForPodCheckpoint(ctx, f, checkpoint.Name)
		condition := expectPodCheckpointFailed(ctx, f, pod, checkpoint)
		gomega.Expect(condition.Message).To(gomega.ContainSubstring("injected checkpoint failure"))
		for _, container := range pod.Spec.Containers {
			gomega.Eventually(ctx, func() error {
				counts, err := podCheckpointProbeCounts(f, pod, container.Name)
				if err != nil {
					return err
				}
				initial := before[container.Name]
				if counts[0] <= initial[0] || counts[1] <= initial[1] {
					return fmt.Errorf("both probes have not resumed in container %q: got %v, previous %v", container.Name, counts, initial)
				}
				return nil
			}, time.Minute, time.Second).Should(gomega.Succeed())
		}
	})
})

// The proxy injector has no RPC context. Release it explicitly on every exit so
// a client deadline cannot leave a blocked server goroutine behind.
func holdPodOperationRPC(ctx context.Context, rpcName string, result error) (<-chan struct{}, func()) {
	ginkgo.GinkgoHelper()
	entered := make(chan struct{})
	released := make(chan struct{})
	finished := make(chan struct{})
	var calls atomic.Int32
	var releaseOnce, finishOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(released) }) }
	framework.ExpectNoError(addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
		if apiName != rpcName {
			return nil
		}
		if calls.Add(1) == 1 {
			close(entered)
		} else if rpcName != criproxy.CancelPodOperation {
			return fmt.Errorf("unexpected duplicate %s call", rpcName)
		}
		defer finishOnce.Do(func() { close(finished) })
		select {
		case <-released:
		case <-ctx.Done():
		}
		return result
	}))
	ginkgo.DeferCleanup(func() error {
		release()
		select {
		case <-entered:
			select {
			case <-finished:
			case <-time.After(10 * time.Second):
				return fmt.Errorf("checkpoint proxy injector did not return after release")
			}
		default:
		}
		return nil
	})
	return entered, release
}

func expectPodCheckpointFailed(ctx context.Context, f *framework.Framework, source *v1.Pod, checkpoint *nodev1alpha1.PodCheckpoint) *metav1.Condition {
	ginkgo.GinkgoHelper()
	condition := apimeta.FindStatusCondition(checkpoint.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	gomega.Expect(condition).NotTo(gomega.BeNil())
	gomega.Expect(condition.Status).To(gomega.Equal(metav1.ConditionFalse))
	gomega.Expect(condition.Reason).To(gomega.Equal(nodev1alpha1.PodCheckpointReasonFailed))
	gomega.Expect(checkpoint.Status.CheckpointLocation).To(gomega.BeNil())
	gomega.Expect(checkpoint.Status.CompletionTime).To(gomega.BeNil())

	checkpointPath := filepath.Join(framework.TestContext.KubeletRootDir, kubeletconfigpaths.DefaultKubeletPodCheckpointsDirName, "checkpoint-"+string(checkpoint.UID))
	_, err := os.Stat(checkpointPath)
	gomega.Expect(os.IsNotExist(err)).To(gomega.BeTrueBecause("failed checkpoint output must be removed: %v", err))
	framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, source.Namespace, source.Name, "checkpoint condition cleared and containers still running", time.Minute, func(pod *v1.Pod) (bool, error) {
		for _, condition := range pod.Status.Conditions {
			if condition.Type == v1.PodCheckpointing && condition.Status == v1.ConditionTrue {
				return false, nil
			}
		}
		if len(pod.Status.ContainerStatuses) != len(source.Status.ContainerStatuses) {
			return false, nil
		}
		for _, container := range pod.Status.ContainerStatuses {
			if container.RestartCount != 0 || container.State.Running == nil {
				return false, fmt.Errorf("source container %q restarted or stopped after checkpoint failure", container.Name)
			}
		}
		return true, nil
	}))
	return condition
}
