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

	ginkgo.BeforeEach(func() {
		if e2eCriProxy == nil {
			ginkgo.Skip("Requires --cri-proxy-enabled=true")
		}
		framework.ExpectNoError(resetCRIProxyInjector(e2eCriProxy))
		ginkgo.DeferCleanup(func() error { return resetCRIProxyInjector(e2eCriProxy) })
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
		entered, _ := holdPodCheckpointRPC(ctx)
		checkpoint := newPodCheckpoint(pod, "checkpoint-timeout-cp")
		checkpoint.Spec.TimeoutSeconds = new(int32(2))
		checkpoint = createPodCheckpoint(ctx, f, checkpoint)
		gomega.Eventually(ctx, entered, 30*time.Second).Should(gomega.BeClosed(), "checkpoint must reach the runtime")

		checkpoint = waitForPodCheckpoint(ctx, f, checkpoint.Name)
		condition := expectPodCheckpointFailed(ctx, f, pod, checkpoint)
		gomega.Expect(condition.Message).To(gomega.Or(gomega.ContainSubstring("DeadlineExceeded"), gomega.ContainSubstring("deadline exceeded")))
	})

})

// The proxy injector has no RPC context. Release it explicitly on every exit so
// a client deadline cannot leave a blocked server goroutine behind.
func holdPodCheckpointRPC(ctx context.Context) (<-chan struct{}, func()) {
	ginkgo.GinkgoHelper()
	entered := make(chan struct{})
	released := make(chan struct{})
	finished := make(chan struct{})
	var calls atomic.Int32
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(released) }) }
	framework.ExpectNoError(addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
		if apiName != criproxy.CheckpointPod {
			return nil
		}
		if calls.Add(1) != 1 {
			return fmt.Errorf("unexpected duplicate CheckpointPod call")
		}
		close(entered)
		defer close(finished)
		select {
		case <-released:
		case <-ctx.Done():
		}
		return fmt.Errorf("injected checkpoint failure")
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
