/*
Copyright 2025 The Kubernetes Authors.

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
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/kubeletconfig"
	"k8s.io/kubernetes/pkg/util/parsers"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

func podCheckpointRuntimeUnsupported(rpcName, message string) bool {
	for _, fragment := range []string{
		"code = Unimplemented desc = unknown method " + rpcName + " for service ",
		"code = Unimplemented desc = method " + rpcName + " not implemented",
		"checkpoint/restore support not available",
		"criu support is disabled by configuration",
		"CRIU binary not found or too old",
		"criu binary not found in shim path or system PATH",
		"checkpoint/restore requires at least CRIU ",
	} {
		if strings.Contains(message, fragment) {
			return true
		}
	}
	return false
}

// The token and counter live in the shell, so starting the command again cannot
// satisfy the continuity check even if a runtime preserves the writable layer.
func podCheckpointSource(namespace, name string) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: v1.PodSpec{
			TerminationGracePeriodSeconds: new(int64(1)),
		},
	}
	for i := 1; i <= 2; i++ {
		pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
			Name:  fmt.Sprintf("test-container-%d", i),
			Image: imageutils.GetE2EImage(imageutils.BusyBox),
			Command: []string{"/bin/sh", "-ec", `token=$(cat /proc/sys/kernel/random/uuid)
count=0
while true; do
  count=$((count + 1))
  printf '%s %s\n' "$token" "$count" > /tmp/checkpoint-state.next
  mv /tmp/checkpoint-state.next /tmp/checkpoint-state
  sleep 1
done`},
			LivenessProbe: &v1.Probe{
				ProbeHandler:   v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"/bin/sh", "-ec", "printf . >> /tmp/liveness-probes; test -s /tmp/checkpoint-state"}}},
				PeriodSeconds:  1,
				TimeoutSeconds: 1,
			},
			ReadinessProbe: &v1.Probe{
				ProbeHandler:   v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"/bin/sh", "-ec", "printf . >> /tmp/readiness-probes; test -s /tmp/checkpoint-state"}}},
				PeriodSeconds:  1,
				TimeoutSeconds: 1,
			},
		})
	}
	return pod
}

func newPodCheckpoint(pod *v1.Pod, name string) *nodev1alpha1.PodCheckpoint {
	return &nodev1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: pod.Namespace},
		Spec: nodev1alpha1.PodCheckpointSpec{
			SourcePod: &nodev1alpha1.PodReference{Name: pod.Name, UID: new(pod.UID)},
		},
	}
}

func createPodCheckpoint(ctx context.Context, f *framework.Framework, checkpoint *nodev1alpha1.PodCheckpoint) *nodev1alpha1.PodCheckpoint {
	ginkgo.GinkgoHelper()
	checkpoints := f.ClientSet.NodeV1alpha1().PodCheckpoints(checkpoint.Namespace)
	created, err := checkpoints.Create(ctx, checkpoint, metav1.CreateOptions{})
	if apierrors.IsNotFound(err) {
		ginkgo.Skip("PodLevelCheckpointRestore API is not enabled")
	}
	framework.ExpectNoError(err)
	ginkgo.DeferCleanup(func(ctx context.Context) error {
		err := checkpoints.Delete(ctx, created.Name, metav1.DeleteOptions{})
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	})
	return created
}

func waitForPodCheckpoint(ctx context.Context, f *framework.Framework, name string) *nodev1alpha1.PodCheckpoint {
	ginkgo.GinkgoHelper()
	var checkpoint *nodev1alpha1.PodCheckpoint
	err := wait.PollUntilContextTimeout(ctx, time.Second, 3*time.Minute, true, func(ctx context.Context) (bool, error) {
		var err error
		checkpoint, err = f.ClientSet.NodeV1alpha1().PodCheckpoints(f.Namespace.Name).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		condition := apimeta.FindStatusCondition(checkpoint.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
		if condition == nil {
			return false, nil
		}
		switch condition.Reason {
		case nodev1alpha1.PodCheckpointReasonCompleted, nodev1alpha1.PodCheckpointReasonFailed, nodev1alpha1.PodCheckpointReasonSourcePodReplaced:
			return true, nil
		}
		return false, nil
	})
	framework.ExpectNoError(err, "PodCheckpoint %q did not reach a terminal Ready condition", name)
	return checkpoint
}

type podCheckpointState struct {
	token string
	count int
}

func readPodCheckpointState(f *framework.Framework, pod *v1.Pod, container string) (podCheckpointState, error) {
	stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, container, "cat", "/tmp/checkpoint-state")
	if err != nil {
		return podCheckpointState{}, fmt.Errorf("read state in %s/%s: %w: %s", pod.Name, container, err, stderr)
	}
	parts := strings.Fields(stdout)
	if len(parts) != 2 {
		return podCheckpointState{}, fmt.Errorf("state in %s/%s has %d fields, want token and counter", pod.Name, container, len(parts))
	}
	count, err := strconv.Atoi(parts[1])
	if err != nil {
		return podCheckpointState{}, fmt.Errorf("parse counter in %s/%s: %w", pod.Name, container, err)
	}
	return podCheckpointState{token: parts[0], count: count}, nil
}

func podCheckpointProbeCounts(f *framework.Framework, pod *v1.Pod, container string) ([2]int, error) {
	stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, container, "/bin/sh", "-ec", "wc -c < /tmp/liveness-probes; wc -c < /tmp/readiness-probes")
	if err != nil {
		return [2]int{}, fmt.Errorf("read probes in %s/%s: %w: %s", pod.Name, container, err, stderr)
	}
	parts := strings.Fields(stdout)
	if len(parts) != 2 {
		return [2]int{}, fmt.Errorf("probe counts in %s/%s have %d fields, want liveness and readiness counts", pod.Name, container, len(parts))
	}
	var counts [2]int
	for i, part := range parts {
		counts[i], err = strconv.Atoi(part)
		if err != nil {
			return [2]int{}, fmt.Errorf("parse probe count in %s/%s: %w", pod.Name, container, err)
		}
	}
	return counts, nil
}

var _ = SIGDescribe("Checkpoint Pod", feature.PodLevelCheckpointRestore, func() {
	f := framework.NewDefaultFramework("checkpoint-pod-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("should checkpoint the data of both containers on their node", func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)
		pod := podClient.CreateSync(ctx, podCheckpointSource(f.Namespace.Name, "checkpoint-source"))
		sourceContainerIDs := map[string]string{}
		for _, container := range pod.Status.ContainerStatuses {
			gomega.Expect(container.ContainerID).NotTo(gomega.BeEmpty())
			sourceContainerIDs[container.Name] = container.ContainerID
		}
		gomega.Expect(sourceContainerIDs).To(gomega.HaveLen(len(pod.Spec.Containers)))

		ginkgo.By("waiting for both containers to advance their in-memory counters")
		before := map[string]podCheckpointState{}
		for _, container := range pod.Spec.Containers {
			gomega.Eventually(ctx, func() error {
				state, err := readPodCheckpointState(f, pod, container.Name)
				if err != nil {
					return err
				}
				if state.count < 3 {
					return fmt.Errorf("counter in %q is %d, want at least 3", container.Name, state.count)
				}
				before[container.Name] = state
				return nil
			}, time.Minute, time.Second).Should(gomega.Succeed())
		}

		ginkgo.By("checkpointing the Pod through its assigned kubelet")
		checkpoint := createPodCheckpoint(ctx, f, newPodCheckpoint(pod, "checkpoint-source-cp"))
		checkpoint = waitForPodCheckpoint(ctx, f, checkpoint.Name)
		condition := apimeta.FindStatusCondition(checkpoint.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
		if condition.Status == metav1.ConditionFalse && condition.Reason == nodev1alpha1.PodCheckpointReasonFailed && podCheckpointRuntimeUnsupported("CheckpointPod", condition.Message) {
			ginkgo.Skip("Container runtime does not support Pod checkpoint: " + condition.Message)
		}
		gomega.Expect(condition.Status).To(gomega.Equal(metav1.ConditionTrue), "checkpoint failed: %s", condition.Message)
		gomega.Expect(condition.Reason).To(gomega.Equal(nodev1alpha1.PodCheckpointReasonCompleted))
		gomega.Expect(checkpoint.Status.NodeName).To(gomega.HaveValue(gomega.Equal(pod.Spec.NodeName)))
		gomega.Expect(checkpoint.Status.SourcePodUID).To(gomega.HaveValue(gomega.Equal(pod.UID)))
		gomega.Expect(checkpoint.Status.CompletionTime).NotTo(gomega.BeNil())
		gomega.Expect(checkpoint.Status.CheckpointedPodTemplate).NotTo(gomega.BeNil())
		gomega.Expect(checkpoint.Status.CheckpointedPodTemplate.Spec.Containers).To(gomega.HaveLen(len(pod.Spec.Containers)))

		ginkgo.By("verifying that the source processes and probes continue after capture")
		for _, container := range pod.Spec.Containers {
			state, err := readPodCheckpointState(f, pod, container.Name)
			framework.ExpectNoError(err)
			gomega.Expect(state.token).To(gomega.Equal(before[container.Name].token))
			gomega.Expect(state.count).To(gomega.BeNumerically(">=", before[container.Name].count))
			counts, err := podCheckpointProbeCounts(f, pod, container.Name)
			framework.ExpectNoError(err)
			gomega.Eventually(ctx, func() error {
				current, err := podCheckpointProbeCounts(f, pod, container.Name)
				if err != nil {
					return err
				}
				if current[0] <= counts[0] || current[1] <= counts[1] {
					return fmt.Errorf("both probes have not advanced after capture in container %q", container.Name)
				}
				return nil
			}, time.Minute, time.Second).Should(gomega.Succeed())
		}
		sourceAfter, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		ready, err := testutils.PodRunningReady(sourceAfter)
		framework.ExpectNoError(err)
		gomega.Expect(ready).To(gomega.BeTrueBecause("source Pod must remain running and ready after capture"))
		gomega.Expect(sourceAfter.Status.ContainerStatuses).To(gomega.HaveLen(len(pod.Status.ContainerStatuses)))
		for _, status := range sourceAfter.Status.ContainerStatuses {
			gomega.Expect(status.RestartCount).To(gomega.BeZero())
			gomega.Expect(sourceContainerIDs).To(gomega.HaveKey(status.Name))
			gomega.Expect(status.ContainerID).To(gomega.Equal(sourceContainerIDs[status.Name]))
		}

		captured := map[string]bool{}
		for _, container := range checkpoint.Status.CheckpointedContainers {
			gomega.Expect(captured).NotTo(gomega.HaveKey(container.Name), "duplicate checkpointed container")
			captured[container.Name] = true
		}
		gomega.Expect(captured).To(gomega.HaveLen(len(pod.Spec.Containers)))
		for _, container := range checkpoint.Status.CheckpointedPodTemplate.Spec.Containers {
			gomega.Expect(captured).To(gomega.HaveKey(container.Name))
			_, _, digest, err := parsers.ParseImageName(container.Image)
			framework.ExpectNoError(err)
			gomega.Expect(digest).NotTo(gomega.BeEmpty(), "container %q must use its resolved image", container.Name)
		}

		ginkgo.By("verifying that the node-local checkpoint contains data")
		location := checkpoint.Status.CheckpointLocation
		gomega.Expect(location).NotTo(gomega.BeNil())
		gomega.Expect(location.Type).To(gomega.Equal(nodev1alpha1.CheckpointSourceTypeNodeLocal))
		gomega.Expect(location.NodeLocal).NotTo(gomega.BeNil())
		gomega.Expect(filepath.IsLocal(location.NodeLocal.Path)).To(gomega.BeTrueBecause("checkpoint location %q must stay below the checkpoint root", location.NodeLocal.Path))
		gomega.Expect(location.NodeLocal.Path).To(gomega.Equal("checkpoint-" + string(checkpoint.UID)))
		checkpointPath := filepath.Join(framework.TestContext.KubeletRootDir, kubeletconfig.DefaultKubeletPodCheckpointsDirName, location.NodeLocal.Path)
		info, err := os.Lstat(checkpointPath)
		framework.ExpectNoError(err, "checkpoint directory must exist")
		gomega.Expect(info.IsDir()).To(gomega.BeTrueBecause("checkpoint path %q must be a directory", checkpointPath))
		ginkgo.DeferCleanup(func() error { return os.RemoveAll(checkpointPath) })
		dataFiles := 0
		framework.ExpectNoError(filepath.WalkDir(checkpointPath, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.Type().IsRegular() {
				info, err := entry.Info()
				if err != nil {
					return err
				}
				if info.Size() > 0 {
					dataFiles++
				}
			}
			return nil
		}))
		gomega.Expect(dataFiles).To(gomega.BeNumerically(">", 0), "checkpoint must contain nonempty files")
	})
})
