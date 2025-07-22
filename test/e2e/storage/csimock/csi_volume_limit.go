/*
Copyright 2022 The Kubernetes Authors.

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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock volume limit", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-limit")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	ginkgo.Context("CSI volume limit information using mock driver", func() {
		f.It("should report attach limit when limit is bigger than 0", f.WithSlow(), func(ctx context.Context) {
			// define volume limit to be 2 for this test
			var err error
			m.init(ctx, testParameters{attachLimit: 2})
			ginkgo.DeferCleanup(m.cleanup)

			nodeName := m.config.ClientNodeSelection.Name
			driverName := m.config.GetUniqueDriverName()

			csiNodeAttachLimit, err := checkCSINodeForLimits(nodeName, driverName, m.cs)
			framework.ExpectNoError(err, "while checking limits in CSINode: %v", err)

			gomega.Expect(csiNodeAttachLimit).To(gomega.BeNumerically("==", 2))

			_, _, pod1 := m.createPod(ctx, pvcReference)
			gomega.Expect(pod1).NotTo(gomega.BeNil(), "while creating first pod")

			err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod1.Name, pod1.Namespace)
			framework.ExpectNoError(err, "Failed to start pod1: %v", err)

			_, _, pod2 := m.createPod(ctx, pvcReference)
			gomega.Expect(pod2).NotTo(gomega.BeNil(), "while creating second pod")

			err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod2.Name, pod2.Namespace)
			framework.ExpectNoError(err, "Failed to start pod2: %v", err)

			_, _, pod3 := m.createPod(ctx, pvcReference)
			gomega.Expect(pod3).NotTo(gomega.BeNil(), "while creating third pod")
			err = waitForMaxVolumeCondition(pod3, m.cs)
			framework.ExpectNoError(err, "while waiting for max volume condition on pod : %+v", pod3)
		})

		f.It("should report attach limit for generic ephemeral volume when persistent volume is attached", f.WithSlow(), func(ctx context.Context) {
			// define volume limit to be 2 for this test
			var err error
			m.init(ctx, testParameters{attachLimit: 1})
			ginkgo.DeferCleanup(m.cleanup)

			nodeName := m.config.ClientNodeSelection.Name
			driverName := m.config.GetUniqueDriverName()

			csiNodeAttachLimit, err := checkCSINodeForLimits(nodeName, driverName, m.cs)
			framework.ExpectNoError(err, "while checking limits in CSINode: %v", err)

			gomega.Expect(csiNodeAttachLimit).To(gomega.BeNumerically("==", 1))

			_, _, pod1 := m.createPod(ctx, pvcReference)
			gomega.Expect(pod1).NotTo(gomega.BeNil(), "while creating pod with persistent volume")

			err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod1.Name, pod1.Namespace)
			framework.ExpectNoError(err, "Failed to start pod1: %v", err)

			_, _, pod2 := m.createPod(ctx, genericEphemeral)
			gomega.Expect(pod2).NotTo(gomega.BeNil(), "while creating pod with ephemeral volume")
			err = waitForMaxVolumeCondition(pod2, m.cs)
			framework.ExpectNoError(err, "while waiting for max volume condition on pod : %+v", pod2)
		})

		f.It("should report attach limit for persistent volume when generic ephemeral volume is attached", f.WithSlow(), func(ctx context.Context) {
			// define volume limit to be 2 for this test
			var err error
			m.init(ctx, testParameters{attachLimit: 1})
			ginkgo.DeferCleanup(m.cleanup)

			nodeName := m.config.ClientNodeSelection.Name
			driverName := m.config.GetUniqueDriverName()

			csiNodeAttachLimit, err := checkCSINodeForLimits(nodeName, driverName, m.cs)
			framework.ExpectNoError(err, "while checking limits in CSINode: %v", err)

			gomega.Expect(csiNodeAttachLimit).To(gomega.BeNumerically("==", 1))

			_, _, pod1 := m.createPod(ctx, genericEphemeral)
			gomega.Expect(pod1).NotTo(gomega.BeNil(), "while creating pod with persistent volume")

			err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod1.Name, pod1.Namespace)
			framework.ExpectNoError(err, "Failed to start pod1: %v", err)

			_, _, pod2 := m.createPod(ctx, pvcReference)
			gomega.Expect(pod2).NotTo(gomega.BeNil(), "while creating pod with ephemeral volume")
			err = waitForMaxVolumeCondition(pod2, m.cs)
			framework.ExpectNoError(err, "while waiting for max volume condition on pod : %+v", pod2)
		})
	})
})

func checkCSINodeForLimits(nodeName string, driverName string, cs clientset.Interface) (int32, error) {
	var attachLimit int32

	waitErr := wait.PollImmediate(10*time.Second, csiNodeLimitUpdateTimeout, func() (bool, error) {
		csiNode, err := cs.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return false, err
		}
		attachLimit = getVolumeLimitFromCSINode(csiNode, driverName)
		if attachLimit > 0 {
			return true, nil
		}
		return false, nil
	})
	if waitErr != nil {
		return 0, fmt.Errorf("error waiting for non-zero volume limit of driver %s on node %s: %v", driverName, nodeName, waitErr)
	}
	return attachLimit, nil
}

func getVolumeLimitFromCSINode(csiNode *storagev1.CSINode, driverName string) int32 {
	for _, d := range csiNode.Spec.Drivers {
		if d.Name != driverName {
			continue
		}
		if d.Allocatable != nil && d.Allocatable.Count != nil {
			return *d.Allocatable.Count
		}
	}
	return 0
}
