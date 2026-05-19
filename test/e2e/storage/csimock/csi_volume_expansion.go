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

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

type expansionStatus int

const (
	expansionSuccess = iota
	expansionFailedOnControllerWithInfeasibleError
	expansionFailedOnControllerWithFinalError
	expansionFailedOnNodeWithInfeasibleError
	expansionFailedOnNodeWithFinalError
	expansionFailedMissingStagingPath
)

const (
	resizePollInterval = 2 * time.Second
	pvcCountQuotaKey   = "persistentvolumeclaims"
	pvcSizeQuotaKey    = "requests.storage"
)

var (
	maxControllerSizeLimit = resource.MustParse("10Gi")

	maxNodeExpansionLimit = resource.MustParse("8Gi")
)

type recoveryTest struct {
	name                       string
	pvcRequestSize             string
	allocatedResource          string
	simulatedCSIDriverError    expansionStatus
	disableControllerExpansion bool
	expectedResizeStatus       v1.ClaimResourceStatus
	recoverySize               resource.Quantity
	fullResourceQuota          *v1.ResourceQuota
	expectedQuotaUsage         *v1.ResourceQuota
}

var _ = utils.SIGDescribe("CSI Mock volume expansion", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-expansion")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	ginkgo.Context("CSI Volume expansion", func() {
		tests := []struct {
			name                    string
			nodeExpansionRequired   bool
			disableAttach           bool
			disableResizingOnDriver bool
			simulatedCSIDriverError expansionStatus
			expectFailure           bool
		}{
			{
				name:                    "should expand volume without restarting pod if nodeExpansion=off",
				nodeExpansionRequired:   false,
				simulatedCSIDriverError: expansionSuccess,
			},
			{
				name:                    "should expand volume by restarting pod if attach=on, nodeExpansion=on",
				nodeExpansionRequired:   true,
				simulatedCSIDriverError: expansionSuccess,
			},
			{
				name:                    "should not have staging_path missing in node expand volume pod if attach=on, nodeExpansion=on",
				nodeExpansionRequired:   true,
				simulatedCSIDriverError: expansionFailedMissingStagingPath,
			},
			{
				name:                    "should expand volume by restarting pod if attach=off, nodeExpansion=on",
				disableAttach:           true,
				nodeExpansionRequired:   true,
				simulatedCSIDriverError: expansionSuccess,
			},
			{
				name:                    "should not expand volume if resizingOnDriver=off, resizingOnSC=on",
				disableResizingOnDriver: true,
				expectFailure:           true,
				simulatedCSIDriverError: expansionSuccess,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(t.name, func(ctx context.Context) {
				var err error
				tp := testParameters{
					enableResizing:          true,
					enableNodeExpansion:     test.nodeExpansionRequired,
					disableResizingOnDriver: test.disableResizingOnDriver,
				}
				// disabling attach requires drive registration feature
				if test.disableAttach {
					tp.disableAttach = true
					tp.registerDriver = true
				}
				tp.hooks = createExpansionHook(test.simulatedCSIDriverError)

				m.init(ctx, tp)
				ginkgo.DeferCleanup(m.cleanup)

				sc, pvc, pod := m.createPod(ctx, pvcReference)
				gomega.Expect(pod).NotTo(gomega.BeNil(), "while creating pod for resizing")

				if !*sc.AllowVolumeExpansion {
					framework.Fail("failed creating sc with allowed expansion")
				}

				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod1: %v", err)

				ginkgo.By("Expanding current pvc")
				newSize := resource.MustParse("6Gi")
				newPVC, err := testsuites.ExpandPVCSize(ctx, pvc, newSize, m.cs)
				framework.ExpectNoError(err, "While updating pvc for more size")
				pvc = newPVC
				gomega.Expect(pvc).NotTo(gomega.BeNil())

				pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
				if pvcSize.Cmp(newSize) != 0 {
					framework.Failf("error updating pvc size %q", pvc.Name)
				}
				if test.expectFailure {
					gomega.Consistently(ctx, framework.GetObject(m.cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get, pvc.Name, metav1.GetOptions{})).WithTimeout(csiResizingConditionWait).
						ShouldNot(gomega.HaveField("Status.Conditions", gomega.ContainElement(gomega.HaveField("Type", gomega.Equal("PersistentVolumeClaimResizing")))), "unexpected resizing condition on PVC")
					return
				}
				ginkgo.By("Waiting for persistent volume resize to finish")
				err = testsuites.WaitForControllerVolumeResize(ctx, pvc, m.cs, csiResizeWaitPeriod)
				framework.ExpectNoError(err, "While waiting for CSI PV resize to finish")

				checkPVCSize := func() {
					ginkgo.By("Waiting for PVC resize to finish")
					pvc, err = testsuites.WaitForFSResize(ctx, pvc, m.cs)
					framework.ExpectNoError(err, "while waiting for PVC resize to finish")

					pvcConditions := pvc.Status.Conditions
					gomega.Expect(pvcConditions).To(gomega.BeEmpty(), "pvc should not have conditions")
				}

				// if node expansion is not required PVC should be resized as well
				if !test.nodeExpansionRequired {
					checkPVCSize()
				} else {
					ginkgo.By("Checking for conditions on pvc")
					npvc, err := testsuites.WaitForPendingFSResizeCondition(ctx, pvc, m.cs)
					framework.ExpectNoError(err, "While waiting for pvc to have fs resizing condition")
					pvc = npvc

					ginkgo.By("Deleting the previously created pod")
					if test.simulatedCSIDriverError == expansionFailedMissingStagingPath {
						e2epod.DeletePodOrFail(ctx, m.cs, pod.Namespace, pod.Name)
					} else {
						err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
						framework.ExpectNoError(err, "while deleting pod for resizing")
					}

					ginkgo.By("Creating a new pod with same volume")
					pod2, err := m.createPodWithPVC(pvc)
					gomega.Expect(pod2).NotTo(gomega.BeNil(), "while creating pod for csi resizing")
					framework.ExpectNoError(err, "while recreating pod for resizing")

					checkPVCSize()
				}
			})
		}
	})
	ginkgo.Context("CSI online volume expansion with secret", func() {
		var stringSecret = map[string]string{
			"username": "admin",
			"password": "t0p-Secret",
		}
		trackedCalls := []string{
			"NodeExpandVolume",
		}
		tests := []struct {
			name          string
			disableAttach bool
			expectedCalls []csiCall

			// Called for each NodeExpandVolume calls, with counter incremented atomically before
			// the invocation (i.e first value will be 1).
			nodeExpandHook func(counter int64) error
		}{
			{
				name: "should expand volume without restarting pod if attach=on, nodeExpansion=on, csiNodeExpandSecret=on",
				expectedCalls: []csiCall{
					{expectedMethod: "NodeExpandVolume", expectedError: codes.OK, expectedSecret: stringSecret},
				},
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func(ctx context.Context) {
				var (
					err        error
					hooks      *drivers.Hooks
					secretName = "test-secret"
					secret     = &v1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Namespace: f.Namespace.Name,
							Name:      secretName,
						},
						StringData: stringSecret,
					}
				)
				if test.nodeExpandHook != nil {
					hooks = createPreHook("NodeExpandVolume", test.nodeExpandHook)
				}
				params := testParameters{enableResizing: true, enableNodeExpansion: true, enableCSINodeExpandSecret: true, hooks: hooks}
				if test.disableAttach {
					params.disableAttach = true
					params.registerDriver = true
				}

				m.init(ctx, params)
				ginkgo.DeferCleanup(m.cleanup)

				if secret, err := m.cs.CoreV1().Secrets(f.Namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{}); err != nil {
					framework.Failf("unable to create test secret %s: %v", secret.Name, err)
				}

				sc, pvc, pod := m.createPod(ctx, pvcReference)
				gomega.Expect(pod).NotTo(gomega.BeNil(), "while creating pod for resizing")

				if !*sc.AllowVolumeExpansion {
					framework.Fail("failed creating sc with allowed expansion")
				}
				if sc.Parameters == nil {
					framework.Fail("failed creating sc with secret")
				}
				if _, ok := sc.Parameters[csiNodeExpandSecretKey]; !ok {
					framework.Failf("creating sc without %s", csiNodeExpandSecretKey)
				}
				if _, ok := sc.Parameters[csiNodeExpandSecretNamespaceKey]; !ok {
					framework.Failf("creating sc without %s", csiNodeExpandSecretNamespaceKey)
				}
				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod1: %v", err)

				pvc, err = m.cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(ctx, pvc.Name, metav1.GetOptions{})
				if err != nil {
					framework.Failf("failed to get pvc %s, %v", pvc.Name, err)
				}
				gomega.Expect(pvc.Spec.VolumeName).ShouldNot(gomega.BeEquivalentTo(""), "while provisioning a volume for resizing")
				pv, err := m.cs.CoreV1().PersistentVolumes().Get(ctx, pvc.Spec.VolumeName, metav1.GetOptions{})
				if err != nil {
					framework.Failf("failed to get pv %s, %v", pvc.Spec.VolumeName, err)
				}
				if pv.Spec.CSI == nil || pv.Spec.CSI.NodeExpandSecretRef == nil {
					framework.Fail("creating pv without 'NodeExpandSecretRef'")
				}
				if pv.Spec.CSI.NodeExpandSecretRef.Namespace != f.Namespace.Name || pv.Spec.CSI.NodeExpandSecretRef.Name != secretName {
					framework.Failf("failed to set node expand secret ref, namespace: %s name: %s", pv.Spec.CSI.NodeExpandSecretRef.Namespace, pv.Spec.CSI.NodeExpandSecretRef.Name)
				}

				ginkgo.By("Expanding current pvc")
				newSize := resource.MustParse("6Gi")
				newPVC, err := testsuites.ExpandPVCSize(ctx, pvc, newSize, m.cs)
				framework.ExpectNoError(err, "While updating pvc for more size")
				pvc = newPVC
				gomega.Expect(pvc).NotTo(gomega.BeNil())

				pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
				if pvcSize.Cmp(newSize) != 0 {
					framework.Failf("error updating pvc size %q", pvc.Name)
				}

				ginkgo.By("Waiting for persistent volume resize to finish")
				err = testsuites.WaitForControllerVolumeResize(ctx, pvc, m.cs, csiResizeWaitPeriod)
				framework.ExpectNoError(err, "While waiting for PV resize to finish")

				ginkgo.By("Waiting for PVC resize to finish")
				pvc, err = testsuites.WaitForFSResize(ctx, pvc, m.cs)
				framework.ExpectNoError(err, "while waiting for PVC to finish")

				ginkgo.By("Waiting for all remaining expected CSI calls")
				err = wait.Poll(time.Second, csiResizeWaitPeriod, func() (done bool, err error) {
					var index int
					_, index, err = compareCSICalls(ctx, trackedCalls, test.expectedCalls, m.driver.GetCalls)
					if err != nil {
						return true, err
					}
					if index == 0 {
						// No CSI call received yet
						return false, nil
					}
					if len(test.expectedCalls) == index {
						// all calls received
						return true, nil
					}
					return false, nil
				})
				framework.ExpectNoError(err, "while waiting for all CSI calls")

				pvcConditions := pvc.Status.Conditions
				gomega.Expect(pvcConditions).To(gomega.BeEmpty(), "pvc should not have conditions")
			})
		}
	})
	ginkgo.Context("CSI online volume expansion", func() {
		tests := []struct {
			name          string
			disableAttach bool
		}{
			{
				name: "should expand volume without restarting pod if attach=on, nodeExpansion=on",
			},
			{
				name:          "should expand volume without restarting pod if attach=off, nodeExpansion=on",
				disableAttach: true,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func(ctx context.Context) {
				var err error
				params := testParameters{enableResizing: true, enableNodeExpansion: true}
				if test.disableAttach {
					params.disableAttach = true
					params.registerDriver = true
				}

				m.init(ctx, params)
				ginkgo.DeferCleanup(m.cleanup)

				sc, pvc, pod := m.createPod(ctx, pvcReference)
				gomega.Expect(pod).NotTo(gomega.BeNil(), "while creating pod for resizing")

				if !*sc.AllowVolumeExpansion {
					framework.Fail("failed creating sc with allowed expansion")
				}

				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod1: %v", err)

				ginkgo.By("Expanding current pvc")
				newSize := resource.MustParse("6Gi")
				newPVC, err := testsuites.ExpandPVCSize(ctx, pvc, newSize, m.cs)
				framework.ExpectNoError(err, "While updating pvc for more size")
				pvc = newPVC
				gomega.Expect(pvc).NotTo(gomega.BeNil())

				pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
				if pvcSize.Cmp(newSize) != 0 {
					framework.Failf("error updating pvc size %q", pvc.Name)
				}

				ginkgo.By("Waiting for persistent volume resize to finish")
				err = testsuites.WaitForControllerVolumeResize(ctx, pvc, m.cs, csiResizeWaitPeriod)
				framework.ExpectNoError(err, "While waiting for PV resize to finish")

				ginkgo.By("Waiting for PVC resize to finish")
				pvc, err = testsuites.WaitForFSResize(ctx, pvc, m.cs)
				framework.ExpectNoError(err, "while waiting for PVC to finish")

				pvcConditions := pvc.Status.Conditions
				gomega.Expect(pvcConditions).To(gomega.BeEmpty(), "pvc should not have conditions")

			})
		}
	})

	f.Context("Expansion with recovery", func() {
		tests := []recoveryTest{
			{
				name:                       "should record target size in allocated resources",
				pvcRequestSize:             "4Gi",
				allocatedResource:          "4Gi",
				disableControllerExpansion: false,
				simulatedCSIDriverError:    expansionSuccess,
				expectedResizeStatus:       "",
				fullResourceQuota: &v1.ResourceQuota{
					Spec: v1.ResourceQuotaSpec{
						Hard: v1.ResourceList{
							pvcSizeQuotaKey:  resource.MustParse("20Gi"),
							pvcCountQuotaKey: resource.MustParse("5"),
						},
					},
				},
				expectedQuotaUsage: &v1.ResourceQuota{
					Status: v1.ResourceQuotaStatus{
						Used: v1.ResourceList{
							pvcSizeQuotaKey:  resource.MustParse("4Gi"),
							pvcCountQuotaKey: resource.MustParse("1"),
						},
					},
				},
			},
			{
				name:                       "should allow recovery if controller expansion fails with infeasible error",
				pvcRequestSize:             "11Gi", // expansion to 11Gi will cause expansion to fail on controller
				allocatedResource:          "11Gi",
				disableControllerExpansion: false,
				simulatedCSIDriverError:    expansionFailedOnControllerWithInfeasibleError,
				expectedResizeStatus:       v1.PersistentVolumeClaimControllerResizeInfeasible,
				recoverySize:               resource.MustParse("4Gi"),
				fullResourceQuota: &v1.ResourceQuota{
					Spec: v1.ResourceQuotaSpec{
						Hard: v1.ResourceList{
							pvcSizeQuotaKey:  resource.MustParse("20Gi"),
							pvcCountQuotaKey: resource.MustParse("5"),
						},
					},
				},
				expectedQuotaUsage: &v1.ResourceQuota{
					Status: v1.ResourceQuotaStatus{
						Used: v1.ResourceList{
							pvcSizeQuotaKey:  resource.MustParse("4Gi"),
							pvcCountQuotaKey: resource.MustParse("1"),
						},
					},
				},
			},
			{
				name:                       "should allow recovery if controller expansion fails with final error",
				pvcRequestSize:             "11Gi", // expansion to 11Gi will cause expansion to fail on controller
				allocatedResource:          "11Gi",
				disableControllerExpansion: false,
				simulatedCSIDriverError:    expansionFailedOnControllerWithFinalError,
				expectedResizeStatus:       v1.PersistentVolumeClaimControllerResizeInProgress,
				recoverySize:               resource.MustParse("4Gi"),
				fullResourceQuota: &v1.ResourceQuota{
					Spec: v1.ResourceQuotaSpec{
						Hard: v1.ResourceList{
							pvcSizeQuotaKey:  resource.MustParse("20Gi"),
							pvcCountQuotaKey: resource.MustParse("5"),
						},
					},
				},
				expectedQuotaUsage: &v1.ResourceQuota{
					Status: v1.ResourceQuotaStatus{
						Used: v1.ResourceList{
							pvcSizeQuotaKey:  resource.MustParse("4Gi"),
							pvcCountQuotaKey: resource.MustParse("1"),
						},
					},
				},
			},
			{
				name:                       "recovery should not be possible in partially expanded volumes",
				pvcRequestSize:             "9Gi", // expansion to 9Gi will cause expansion to fail on node
				allocatedResource:          "9Gi",
				disableControllerExpansion: false,
				simulatedCSIDriverError:    expansionFailedOnNodeWithInfeasibleError,
				expectedResizeStatus:       v1.PersistentVolumeClaimNodeResizeInfeasible,
				recoverySize:               resource.MustParse("5Gi"),
			},
			{
				name:                       "recovery should be possible for node-only expanded volumes with infeasible error",
				pvcRequestSize:             "9Gi", // expansion to 9Gi will cause expansion to fail on node
				allocatedResource:          "9Gi",
				disableControllerExpansion: true,
				simulatedCSIDriverError:    expansionFailedOnNodeWithInfeasibleError,
				expectedResizeStatus:       v1.PersistentVolumeClaimNodeResizeInfeasible,
				recoverySize:               resource.MustParse("5Gi"),
				fullResourceQuota: &v1.ResourceQuota{
					Spec: v1.ResourceQuotaSpec{
						Hard: v1.ResourceList{
							pvcSizeQuotaKey:  resource.MustParse("20Gi"),
							pvcCountQuotaKey: resource.MustParse("5"),
						},
					},
				},
				expectedQuotaUsage: &v1.ResourceQuota{
					Status: v1.ResourceQuotaStatus{
						Used: v1.ResourceList{
							pvcSizeQuotaKey:  resource.MustParse("5Gi"),
							pvcCountQuotaKey: resource.MustParse("1"),
						},
					},
				},
			},
			{
				name:                       "recovery should be possible for node-only expanded volumes with final error",
				pvcRequestSize:             "9Gi", // expansion to 9Gi will cause expansion to fail on node
				allocatedResource:          "9Gi",
				disableControllerExpansion: true,
				simulatedCSIDriverError:    expansionFailedOnNodeWithFinalError,
				expectedResizeStatus:       v1.PersistentVolumeClaimNodeResizeInProgress,
				recoverySize:               resource.MustParse("5Gi"),
				fullResourceQuota: &v1.ResourceQuota{
					Spec: v1.ResourceQuotaSpec{
						Hard: v1.ResourceList{
							pvcSizeQuotaKey:  resource.MustParse("20Gi"),
							pvcCountQuotaKey: resource.MustParse("5"),
						},
					},
				},
				expectedQuotaUsage: &v1.ResourceQuota{
					Status: v1.ResourceQuotaStatus{
						Used: v1.ResourceList{
							pvcSizeQuotaKey:  resource.MustParse("5Gi"),
							pvcCountQuotaKey: resource.MustParse("1"),
						},
					},
				},
			},
		}

		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func(ctx context.Context) {
				var err error
				params := testParameters{enableResizing: true, enableNodeExpansion: true, disableControllerExpansion: test.disableControllerExpansion}

				if test.simulatedCSIDriverError != expansionSuccess {
					params.hooks = createExpansionHook(test.simulatedCSIDriverError)
				}

				m.init(ctx, params)
				ginkgo.DeferCleanup(m.cleanup)
				var currentQuota *v1.ResourceQuota

				if test.fullResourceQuota != nil {
					currentQuota = m.createResourceQuota(ctx, test.fullResourceQuota)
				}

				sc, pvc, pod := m.createPod(ctx, pvcReference)
				gomega.Expect(pod).NotTo(gomega.BeNil(), "while creating pod for resizing")

				if !*sc.AllowVolumeExpansion {
					framework.Fail("failed creating sc with allowed expansion")
				}

				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod1: %v", err)

				ginkgo.By("Expanding current pvc")
				newSize := resource.MustParse(test.pvcRequestSize)
				newPVC, err := testsuites.ExpandPVCSize(ctx, pvc, newSize, m.cs)
				framework.ExpectNoError(err, "While updating pvc for more size")
				pvc = newPVC
				gomega.Expect(pvc).NotTo(gomega.BeNil())

				pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
				if pvcSize.Cmp(newSize) != 0 {
					framework.Failf("error updating pvc size %q", pvc.Name)
				}

				if test.simulatedCSIDriverError == expansionSuccess {
					validateExpansionSuccess(ctx, pvc, m, test, test.allocatedResource)
				} else {
					validateRecoveryBehaviour(ctx, pvc, m, test)
				}

				if test.expectedQuotaUsage != nil {
					validateQuotaUsage(ctx, m, currentQuota, test.expectedQuotaUsage)
				}
			})
		}
	})
})

func validateQuotaUsage(ctx context.Context, m *mockDriverSetup, currentQuota, expectedQuota *v1.ResourceQuota) {
	ginkgo.By("Waiting for resource quota usage to be updated")
	var (
		quota     *v1.ResourceQuota
		usedCount resource.Quantity
		usedSize  resource.Quantity
	)

	expectedCount := expectedQuota.Status.Used[pvcCountQuotaKey]
	expectedUsedSize := expectedQuota.Status.Used[pvcSizeQuotaKey]

	gomega.Eventually(func() error {
		q, err := m.cs.CoreV1().ResourceQuotas(currentQuota.Namespace).Get(ctx, currentQuota.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get resource quota %s/%s: %w", currentQuota.Namespace, currentQuota.Name, err)
		}
		if q.Status.Used == nil {
			return fmt.Errorf("resource quota %s/%s has nil Status.Used", currentQuota.Namespace, currentQuota.Name)
		}

		quota = q
		usedCount = quota.Status.Used[pvcCountQuotaKey]
		usedSize = quota.Status.Used[pvcSizeQuotaKey]

		if usedCount.Cmp(expectedCount) != 0 || usedSize.Cmp(expectedUsedSize) != 0 {
			return fmt.Errorf(
				"resource quota usage did not converge; currentlyUsed: %s/%s, expected: %s/%s",
				usedCount.String(), usedSize.String(), expectedCount.String(), expectedUsedSize.String(),
			)
		}
		return nil
	}, csiResizeWaitPeriod, resizePollInterval).Should(gomega.Succeed())
}

func validateRecoveryBehaviour(ctx context.Context, pvc *v1.PersistentVolumeClaim, m *mockDriverSetup, test recoveryTest) {
	var err error
	ginkgo.By("Waiting for resizer to set allocated resource")
	err = waitForAllocatedResource(ctx, pvc, m, test.allocatedResource)
	framework.ExpectNoError(err, "While waiting for allocated resource to be updated")

	if test.expectedResizeStatus == v1.PersistentVolumeClaimNodeResizeInfeasible {
		ginkgo.By("Waiting for kubelet to fail expansion on the node")
		err = waitForResizeToFailOnNode(ctx, pvc, m.cs)
		framework.ExpectNoError(err, "While waiting for resize status to be set")
	} else {
		ginkgo.By("Waiting for resizer to set resize status")
		err = waitForResizeStatus(ctx, pvc, m.cs, test.expectedResizeStatus)
		framework.ExpectNoError(err, "While waiting for resize status to be set")
	}

	ginkgo.By("Recover pvc size")
	newPVC, err := testsuites.ExpandPVCSize(ctx, pvc, test.recoverySize, m.cs)
	framework.ExpectNoError(err, "While updating pvc for more size")
	pvc = newPVC
	gomega.Expect(pvc).NotTo(gomega.BeNil())

	pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
	if pvcSize.Cmp(test.recoverySize) != 0 {
		framework.Failf("error updating pvc size %q", pvc.Name)
	}

	// If expansion failed on controller (infeasible or final), recovery should be possible.
	// Wait for the recovery resize to settle before checking quota.
	if test.simulatedCSIDriverError == expansionFailedOnControllerWithInfeasibleError ||
		test.simulatedCSIDriverError == expansionFailedOnControllerWithFinalError {
		validateExpansionSuccess(ctx, pvc, m, test, test.recoverySize.String())
		return
	}

	// if expansion failed on node with final error but volume was only expanded on the node
	// then recovery should be possible
	if test.disableControllerExpansion &&
		(test.simulatedCSIDriverError == expansionFailedOnNodeWithInfeasibleError ||
			test.simulatedCSIDriverError == expansionFailedOnNodeWithFinalError) {
		validateExpansionSuccess(ctx, pvc, m, test, test.recoverySize.String())
		return
	}

	// if expansion succeeded on controller but failed on the node
	if test.simulatedCSIDriverError == expansionFailedOnNodeWithInfeasibleError {
		ginkgo.By("Wait for expansion to fail on node again")
		err = waitForResizeToFailOnNode(ctx, pvc, m.cs)
		framework.ExpectNoError(err, "While waiting for resize status to be set")

		ginkgo.By("verify allocated resources after recovery")
		pvc, err = m.cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(context.TODO(), pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "while fetching pvc")
		actualAllocatedResource := pvc.Status.AllocatedResources.Storage()

		if actualAllocatedResource.Equal(test.recoverySize) {
			framework.Failf("unexpected allocated resource size %s after node expansion failure", actualAllocatedResource.String())
		}

		if !actualAllocatedResource.Equal(resource.MustParse(test.allocatedResource)) {
			framework.Failf("expected allocated resources to be %s got %s", test.allocatedResource, actualAllocatedResource.String())
		}
	}
}

func validateExpansionSuccess(ctx context.Context, pvc *v1.PersistentVolumeClaim, m *mockDriverSetup, test recoveryTest, expectedAllocatedSize string) {
	var err error
	ginkgo.By(fmt.Sprintf("Waiting for PV %s to be expanded to %s", pvc.Spec.VolumeName, test.recoverySize.String()))
	err = testsuites.WaitForControllerVolumeResize(ctx, pvc, m.cs, csiResizeWaitPeriod)
	framework.ExpectNoError(err, "While waiting for PV resize to finish")

	ginkgo.By(fmt.Sprintf("Waiting for PVC %s to be expanded to %s", pvc.Name, test.recoverySize.String()))
	pvc, err = testsuites.WaitForFSResize(ctx, pvc, m.cs)
	framework.ExpectNoError(err, "while waiting for PVC to finish")

	pvcConditions := pvc.Status.Conditions
	gomega.Expect(pvcConditions).To(gomega.BeEmpty(), "pvc should not have conditions")
	allocatedResource := pvc.Status.AllocatedResources.Storage()
	gomega.Expect(allocatedResource).NotTo(gomega.BeNil())
	expectedAllocatedResource := resource.MustParse(expectedAllocatedSize)
	if allocatedResource.Cmp(expectedAllocatedResource) != 0 {
		framework.Failf("expected allocated Resources to be %s got %s", expectedAllocatedResource.String(), allocatedResource.String())
	}

	resizeStatus := pvc.Status.AllocatedResourceStatuses[v1.ResourceStorage]
	gomega.Expect(resizeStatus).To(gomega.BeZero(), "resize status should be empty")
}

func waitForResizeStatus(ctx context.Context, pvc *v1.PersistentVolumeClaim, c clientset.Interface, expectedState v1.ClaimResourceStatus) error {
	var actualResizeStatus v1.ClaimResourceStatus

	waitErr := wait.PollUntilContextTimeout(ctx, resizePollInterval, csiResizeWaitPeriod, true, func(pollContext context.Context) (bool, error) {
		var err error
		updatedPVC, err := c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pollContext, pvc.Name, metav1.GetOptions{})

		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for checking for resize status: %w", pvc.Name, err)
		}

		actualResizeStatus = updatedPVC.Status.AllocatedResourceStatuses[v1.ResourceStorage]
		return (actualResizeStatus == expectedState), nil
	})
	if waitErr != nil {
		return fmt.Errorf("error while waiting for resize status to sync to %v, actualStatus %s: %w", expectedState, actualResizeStatus, waitErr)
	}
	return nil
}

func waitForResizeToFailOnNode(ctx context.Context, pvc *v1.PersistentVolumeClaim, c clientset.Interface) error {
	var finalConditions []v1.PersistentVolumeClaimCondition
	waitErr := wait.PollUntilContextTimeout(ctx, resizePollInterval, csiResizeWaitPeriod, true, func(pollContext context.Context) (bool, error) {
		var err error
		updatedPVC, err := c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pollContext, pvc.Name, metav1.GetOptions{})

		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for checking for resize status: %w", pvc.Name, err)
		}
		pvcConditions := updatedPVC.Status.Conditions
		for _, cond := range pvcConditions {
			if cond.Type == v1.PersistentVolumeClaimNodeResizeError {
				return true, nil
			}
		}
		finalConditions = pvcConditions
		return false, nil
	})

	if waitErr != nil {
		return fmt.Errorf("error while waiting for resize condition sync to NodeResizeError, actualStatus %+v: %w", finalConditions, waitErr)
	}
	return nil
}

func waitForAllocatedResource(ctx context.Context, pvc *v1.PersistentVolumeClaim, m *mockDriverSetup, expectedSize string) error {
	expectedQuantity := resource.MustParse(expectedSize)
	waitErr := wait.PollUntilContextTimeout(ctx, resizePollInterval, csiResizeWaitPeriod, true, func(pollContext context.Context) (bool, error) {
		var err error
		updatedPVC, err := m.cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pollContext, pvc.Name, metav1.GetOptions{})

		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for checking for resize status: %w", pvc.Name, err)
		}
		actualAllocatedSize := updatedPVC.Status.AllocatedResources.Storage()
		if actualAllocatedSize != nil && actualAllocatedSize.Equal(expectedQuantity) {
			return true, nil
		}
		return false, nil

	})
	if waitErr != nil {
		return fmt.Errorf("error while waiting for allocatedSize to sync to %s: %v", expectedSize, waitErr)
	}
	return nil
}

func createExpansionHook(expectedExpansionStatus expansionStatus) *drivers.Hooks {
	return &drivers.Hooks{
		Pre: func(ctx context.Context, method string, request interface{}) (reply interface{}, err error) {
			switch expectedExpansionStatus {
			case expansionFailedMissingStagingPath:
				expansionRequest, ok := request.(*csipbv1.NodeExpandVolumeRequest)
				if ok {
					stagingPath := expansionRequest.StagingTargetPath
					if stagingPath == "" {
						return nil, status.Error(codes.InvalidArgument, "invalid node expansion request, missing staging path")
					}

				}
			case expansionFailedOnControllerWithInfeasibleError:
				expansionRequest, ok := request.(*csipbv1.ControllerExpandVolumeRequest)
				if ok {
					requestedSize := resource.NewQuantity(expansionRequest.CapacityRange.RequiredBytes, resource.BinarySI)
					if requestedSize.Cmp(maxControllerSizeLimit) > 0 {
						return nil, status.Error(codes.InvalidArgument, "invalid expansion request")
					}
				}
			case expansionFailedOnControllerWithFinalError:
				// This simulates a condition that a final, but not infeasible error is returned when expansion fails in the controller.
				expansionRequest, ok := request.(*csipbv1.ControllerExpandVolumeRequest)
				if ok {
					requestedSize := resource.NewQuantity(expansionRequest.CapacityRange.RequiredBytes, resource.BinarySI)
					if requestedSize.Cmp(maxControllerSizeLimit) > 0 {
						return nil, status.Error(codes.PermissionDenied, "permission denied for expansion")
					}
				}
			case expansionFailedOnNodeWithInfeasibleError:
				expansionRequest, ok := request.(*csipbv1.NodeExpandVolumeRequest)
				if ok {
					requestedSize := resource.NewQuantity(expansionRequest.CapacityRange.RequiredBytes, resource.BinarySI)
					if requestedSize.Cmp(maxNodeExpansionLimit) > 0 {
						return nil, status.Error(codes.InvalidArgument, "invalid node expansion request")
					}

				}
			case expansionFailedOnNodeWithFinalError:
				expansionRequest, ok := request.(*csipbv1.NodeExpandVolumeRequest)
				if ok {
					requestedSize := resource.NewQuantity(expansionRequest.CapacityRange.RequiredBytes, resource.BinarySI)
					if requestedSize.Cmp(maxNodeExpansionLimit) > 0 {
						return nil, status.Error(codes.PermissionDenied, "permission denied for expansion")
					}
				}
			}

			return nil, nil
		},
	}
}
