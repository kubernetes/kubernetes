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

package csi_mock

import (
	"context"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock volume expansion", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-expansion")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	ginkgo.Context("CSI Volume expansion", func() {
		tests := []struct {
			name                    string
			nodeExpansionRequired   bool
			disableAttach           bool
			disableResizingOnDriver bool
			expectFailure           bool
		}{
			{
				name:                  "should expand volume without restarting pod if nodeExpansion=off",
				nodeExpansionRequired: false,
			},
			{
				name:                  "should expand volume by restarting pod if attach=on, nodeExpansion=on",
				nodeExpansionRequired: true,
			},
			{
				name:                  "should expand volume by restarting pod if attach=off, nodeExpansion=on",
				disableAttach:         true,
				nodeExpansionRequired: true,
			},
			{
				name:                    "should not expand volume if resizingOnDriver=off, resizingOnSC=on",
				disableResizingOnDriver: true,
				expectFailure:           true,
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

				m.init(tp)
				ginkgo.DeferCleanup(m.cleanup)

				sc, pvc, pod := m.createPod(pvcReference)
				gomega.Expect(pod).NotTo(gomega.BeNil(), "while creating pod for resizing")

				if !*sc.AllowVolumeExpansion {
					framework.Fail("failed creating sc with allowed expansion")
				}

				err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod1: %v", err)

				ginkgo.By("Expanding current pvc")
				newSize := resource.MustParse("6Gi")
				newPVC, err := testsuites.ExpandPVCSize(pvc, newSize, m.cs)
				framework.ExpectNoError(err, "While updating pvc for more size")
				pvc = newPVC
				gomega.Expect(pvc).NotTo(gomega.BeNil())

				pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
				if pvcSize.Cmp(newSize) != 0 {
					framework.Failf("error updating pvc size %q", pvc.Name)
				}
				if test.expectFailure {
					err = testsuites.WaitForResizingCondition(pvc, m.cs, csiResizingConditionWait)
					framework.ExpectError(err, "unexpected resizing condition on PVC")
					return
				}

				ginkgo.By("Waiting for persistent volume resize to finish")
				err = testsuites.WaitForControllerVolumeResize(pvc, m.cs, csiResizeWaitPeriod)
				framework.ExpectNoError(err, "While waiting for CSI PV resize to finish")

				checkPVCSize := func() {
					ginkgo.By("Waiting for PVC resize to finish")
					pvc, err = testsuites.WaitForFSResize(pvc, m.cs)
					framework.ExpectNoError(err, "while waiting for PVC resize to finish")

					pvcConditions := pvc.Status.Conditions
					framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
				}

				// if node expansion is not required PVC should be resized as well
				if !test.nodeExpansionRequired {
					checkPVCSize()
				} else {
					ginkgo.By("Checking for conditions on pvc")
					npvc, err := testsuites.WaitForPendingFSResizeCondition(pvc, m.cs)
					framework.ExpectNoError(err, "While waiting for pvc to have fs resizing condition")
					pvc = npvc

					inProgressConditions := pvc.Status.Conditions
					if len(inProgressConditions) > 0 {
						framework.ExpectEqual(inProgressConditions[0].Type, v1.PersistentVolumeClaimFileSystemResizePending, "pvc must have fs resizing condition")
					}

					ginkgo.By("Deleting the previously created pod")
					err = e2epod.DeletePodWithWait(m.cs, pod)
					framework.ExpectNoError(err, "while deleting pod for resizing")

					ginkgo.By("Creating a new pod with same volume")
					pod2, err := m.createPodWithPVC(pvc)
					gomega.Expect(pod2).NotTo(gomega.BeNil(), "while creating pod for csi resizing")
					framework.ExpectNoError(err, "while recreating pod for resizing")

					checkPVCSize()
				}
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

				m.init(params)
				ginkgo.DeferCleanup(m.cleanup)

				sc, pvc, pod := m.createPod(pvcReference)
				gomega.Expect(pod).NotTo(gomega.BeNil(), "while creating pod for resizing")

				if !*sc.AllowVolumeExpansion {
					framework.Fail("failed creating sc with allowed expansion")
				}

				err = e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod1: %v", err)

				ginkgo.By("Expanding current pvc")
				newSize := resource.MustParse("6Gi")
				newPVC, err := testsuites.ExpandPVCSize(pvc, newSize, m.cs)
				framework.ExpectNoError(err, "While updating pvc for more size")
				pvc = newPVC
				gomega.Expect(pvc).NotTo(gomega.BeNil())

				pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
				if pvcSize.Cmp(newSize) != 0 {
					framework.Failf("error updating pvc size %q", pvc.Name)
				}

				ginkgo.By("Waiting for persistent volume resize to finish")
				err = testsuites.WaitForControllerVolumeResize(pvc, m.cs, csiResizeWaitPeriod)
				framework.ExpectNoError(err, "While waiting for PV resize to finish")

				ginkgo.By("Waiting for PVC resize to finish")
				pvc, err = testsuites.WaitForFSResize(pvc, m.cs)
				framework.ExpectNoError(err, "while waiting for PVC to finish")

				pvcConditions := pvc.Status.Conditions
				framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")

			})
		}
	})
})
