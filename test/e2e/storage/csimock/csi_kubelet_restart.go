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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock when kubelet restart", framework.WithSerial(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("csi-mock-when-kubelet-restart")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	ginkgo.BeforeEach(func() {
		// These tests requires SSH to nodes, so the provider check should be identical to there
		// (the limiting factor is the implementation of util.go's e2essh.GetSigner(...)).

		// Cluster must support node reboot
		e2eskipper.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		e2eskipper.SkipUnlessSSHKeyPresent()
	})

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
		pod, err = f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod %s", pod.Name)

		ginkgo.By("Deleting the PVC")
		err = f.ClientSet.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(ctx, pvc.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete PVC %s", pvc.Name)

		ginkgo.By("Restarting kubelet")
		utils.KubeletCommand(ctx, utils.KRestart, f.ClientSet, pod)
		ginkgo.DeferCleanup(utils.KubeletCommand, utils.KStart, f.ClientSet, pod)

		ginkgo.By("Verifying the PVC is terminating during kubelet restart")
		pvc, err = f.ClientSet.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(ctx, pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get PVC %s", pvc.Name)
		gomega.Expect(pvc.DeletionTimestamp).NotTo(gomega.BeNil(), "PVC %s should have deletion timestamp", pvc.Name)

		ginkgo.By(fmt.Sprintf("Verifying that the driver didn't receive NodeUnpublishVolume call for PVC %s", pvc.Name))
		gomega.Consistently(ctx,
			func(ctx context.Context) []drivers.MockCSICall {
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
