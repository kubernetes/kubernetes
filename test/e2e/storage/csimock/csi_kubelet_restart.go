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

package csimock

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock when kubelet restart", feature.Kind, framework.WithSerial(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("csi-mock-when-kubelet-restart")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	ginkgo.It("should not umount volume when the pvc is terminating but still used by a running pod", func(ctx context.Context) {
		m.init(ctx, testParameters{
			registerDriver: true,
		})
		ginkgo.DeferCleanup(m.cleanup)

		ginkgo.By("Creating a Pod with a PVC backed by a CSI volume")
		_, pvc, pod := m.createPod(ctx, pvcReference)

		ginkgo.By("Waiting for the Pod to be running")
		err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
		framework.ExpectNoError(err, "failed to wait for pod %s to be running", pod.Name)

		ginkgo.By("Deleting the PVC")
		err = f.ClientSet.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(ctx, pvc.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete PVC %s", pvc.Name)

		ginkgo.By("Restarting kubelet")
		err = stopKindKubelet(ctx)
		framework.ExpectNoError(err, "failed to stop kubelet")
		err = startKindKubelet(ctx)
		framework.ExpectNoError(err, "failed to start kubelet")

		ginkgo.By("Verifying the PVC is terminating during kubelet restart")
		pvc, err = f.ClientSet.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(ctx, pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get PVC %s", pvc.Name)
		gomega.Expect(pvc.DeletionTimestamp).NotTo(gomega.BeNil(), "PVC %s should have deletion timestamp", pvc.Name)

		ginkgo.By(fmt.Sprintf("Verifying that the driver didn't receive NodeUnpublishVolume call for PVC %s", pvc.Name))
		gomega.Consistently(ctx,
			func(ctx context.Context) interface{} {
				calls, err := m.driver.GetCalls(ctx)
				if err != nil {
					if apierrors.IsUnexpectedServerError(err) {
						// kubelet might not be ready yet when getting the calls
						gomega.TryAgainAfter(framework.Poll).Wrap(err).Now()
						return nil
					}
					return nil
				}
				return calls
			}).
			WithPolling(framework.Poll).
			WithTimeout(framework.ClaimProvisionShortTimeout).
			ShouldNot(gomega.ContainElement(gomega.HaveField("Method", gomega.Equal("NodeUnpublishVolume"))))

		ginkgo.By("Verifying the Pod is still running")
		err = e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
		framework.ExpectNoError(err, "failed to wait for pod %s to be running", pod.Name)
	})
})

func stopKindKubelet(ctx context.Context) error {
	return kubeletExec("systemctl", "stop", "kubelet")
}

func startKindKubelet(ctx context.Context) error {
	return kubeletExec("systemctl", "start", "kubelet")
}

// Run a command in container with kubelet (and the whole control plane as containers)
func kubeletExec(command ...string) error {
	containerName := getKindContainerName()
	args := []string{"exec", containerName}
	args = append(args, command...)
	cmd := exec.Command("docker", args...)

	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command %q failed: %v\noutput:%s", prettyCmd(cmd), err, string(out))
	}

	framework.Logf("command %q succeeded:\n%s", prettyCmd(cmd), string(out))
	return nil
}

func getKindContainerName() string {
	clusterName := os.Getenv("KIND_CLUSTER_NAME")
	if clusterName == "" {
		clusterName = "kind"
	}
	return clusterName + "-control-plane"
}

func prettyCmd(cmd *exec.Cmd) string {
	return fmt.Sprintf("%s %s", cmd.Path, strings.Join(cmd.Args, " "))
}
