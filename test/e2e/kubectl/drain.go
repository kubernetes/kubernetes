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

// OWNER = sig/cli

package kubectl

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"syscall"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const drainNodeConditionsEnv = "KUBECTL_DRAIN_NODE_CONDITIONS=true"

var _ = SIGDescribe("Kubectl drain with lifecycle conditions", func() {
	f := framework.NewDefaultFramework("kubectl-drain-conditions")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var nodeName string
	ginkgo.BeforeEach(func(ctx context.Context) {
		nodeName = "kubectl-drain-conditions-" + rand.String(8)
		_, err := f.ClientSet.CoreV1().Nodes().Create(ctx, &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: nodeName},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(framework.IgnoreNotFound(f.ClientSet.CoreV1().Nodes().Delete), nodeName, metav1.DeleteOptions{})
	})

	ginkgo.It("reports successful drain and uncordon observations", func(ctx context.Context) {
		e2ekubectl.NewKubectlCommand(f.Namespace.Name, "drain", nodeName, "--force", "--ignore-daemonsets").
			AppendEnv([]string{drainNodeConditionsEnv}).
			ExecOrDie(f.Namespace.Name)

		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		expectNodeCondition(node, v1.NodeDrainInProgress, v1.ConditionFalse, "KubectlDrainCompleted")
		expectNodeCondition(node, v1.NodeDrained, v1.ConditionTrue, "KubectlDrainCompleted")

		e2ekubectl.NewKubectlCommand(f.Namespace.Name, "uncordon", nodeName).
			AppendEnv([]string{drainNodeConditionsEnv}).
			ExecOrDie(f.Namespace.Name)

		node, err = f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		expectNodeCondition(node, v1.NodeDrainInProgress, v1.ConditionFalse, "KubectlUncordoned")
		expectNodeCondition(node, v1.NodeDrained, v1.ConditionFalse, "KubectlUncordoned")
	})

	ginkgo.It("reports a failed drain", func(ctx context.Context) {
		podName := "unmanaged-pod-" + rand.String(8)
		_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: podName},
			Spec: v1.PodSpec{
				NodeName: nodeName,
				Containers: []v1.Container{{
					Name:  "pause",
					Image: imageutils.GetE2EImage(imageutils.Pause),
				}},
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(framework.IgnoreNotFound(f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete), podName, metav1.DeleteOptions{})

		_, err = e2ekubectl.NewKubectlCommand(f.Namespace.Name, "drain", nodeName, "--ignore-daemonsets").
			AppendEnv([]string{drainNodeConditionsEnv}).
			Exec()
		gomega.Expect(err).To(gomega.HaveOccurred())

		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		expectNodeCondition(node, v1.NodeDrainInProgress, v1.ConditionFalse, "KubectlDrainFailed")
		expectNodeCondition(node, v1.NodeDrained, v1.ConditionFalse, "KubectlDrainFailed")
	})

	ginkgo.It("reports an interrupted drain before exiting", func(ctx context.Context) {
		podName := "drain-blocker-" + rand.String(8)
		_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:       podName,
				Finalizers: []string{"e2e.kubernetes.io/drain-blocker"},
			},
			Spec: v1.PodSpec{
				NodeName: nodeName,
				Containers: []v1.Container{{
					Name:  "pause",
					Image: imageutils.GetE2EImage(imageutils.Pause),
				}},
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(func(ctx context.Context) {
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podName, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return
			}
			framework.ExpectNoError(err)
			pod.Finalizers = nil
			_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Update(ctx, pod, metav1.UpdateOptions{})
			if apierrors.IsNotFound(err) {
				return
			}
			framework.ExpectNoError(err)
			err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(ctx, podName, metav1.DeleteOptions{})
			if !apierrors.IsNotFound(err) {
				framework.ExpectNoError(err)
			}
		})

		command := drainCommand(f.Namespace.Name, nodeName)
		var stdout, stderr bytes.Buffer
		command.Stdout = &stdout
		command.Stderr = &stderr
		framework.ExpectNoError(command.Start())

		waitResult := make(chan error, 1)
		go func() {
			waitResult <- command.Wait()
		}()
		waited := false
		ginkgo.DeferCleanup(func() {
			if waited {
				return
			}
			framework.TryKill(command)
			select {
			case waitErr := <-waitResult:
				framework.Logf("kubectl drain cleanup result: %v", waitErr)
			case <-time.After(wait.ForeverTestTimeout):
				framework.Logf("timed out waiting for kubectl drain cleanup")
			}
		})

		err = wait.PollUntilContextTimeout(ctx, 200*time.Millisecond, framework.PodStartTimeout, true, func(ctx context.Context) (bool, error) {
			node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			condition := findNodeCondition(node, v1.NodeDrainInProgress)
			return condition != nil && condition.Status == v1.ConditionTrue && condition.Reason == "KubectlDrainStarted", nil
		})
		framework.ExpectNoError(err, "waiting for drain to start; stdout: %s, stderr: %s", stdout.String(), stderr.String())
		framework.ExpectNoError(command.Process.Signal(syscall.SIGINT))

		expired := time.NewTimer(wait.ForeverTestTimeout)
		defer expired.Stop()
		select {
		case err = <-waitResult:
			waited = true
		case <-expired.C:
			framework.Failf("timed out waiting for interrupted drain; stdout: %s, stderr: %s", stdout.String(), stderr.String())
		}
		gomega.Expect(err).To(gomega.HaveOccurred())
		var exitError *exec.ExitError
		if !errors.As(err, &exitError) {
			framework.Failf("expected kubectl drain to return *exec.ExitError, got %T: %v", err, err)
		}
		gomega.Expect(exitError.ExitCode()).To(gomega.Equal(1))

		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		expectNodeCondition(node, v1.NodeDrainInProgress, v1.ConditionFalse, "KubectlDrainInterrupted")
		expectNodeCondition(node, v1.NodeDrained, v1.ConditionFalse, "KubectlDrainInterrupted")
	})
})

func drainCommand(namespace, nodeName string) *exec.Cmd {
	testKubeconfig := e2ekubectl.NewTestKubeconfig(
		framework.TestContext.CertDir,
		framework.TestContext.Host,
		framework.TestContext.KubeConfig,
		framework.TestContext.KubeContext,
		framework.TestContext.KubectlPath,
		namespace,
	)
	command := testKubeconfig.KubectlCmd("drain", nodeName, "--force", "--ignore-daemonsets", "--disable-eviction", "--timeout=10m")
	command.Env = append(os.Environ(), drainNodeConditionsEnv)
	return command
}

func findNodeCondition(node *v1.Node, conditionType v1.NodeConditionType) *v1.NodeCondition {
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == conditionType {
			return &node.Status.Conditions[i]
		}
	}
	return nil
}

func expectNodeCondition(node *v1.Node, conditionType v1.NodeConditionType, status v1.ConditionStatus, reason string) {
	ginkgo.GinkgoHelper()
	condition := findNodeCondition(node, conditionType)
	gomega.Expect(condition).NotTo(gomega.BeNil(), fmt.Sprintf("condition %q", conditionType))
	gomega.Expect(condition.Status).To(gomega.Equal(status))
	gomega.Expect(condition.Reason).To(gomega.Equal(reason))
}
