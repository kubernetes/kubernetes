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
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock volume node stage", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-node-stage")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	f.Context("CSI NodeStage error cases", f.WithSlow(), func() {
		trackedCalls := []string{
			"NodeStageVolume",
			"NodeUnstageVolume",
		}

		tests := []struct {
			name             string
			expectPodRunning bool
			expectedCalls    []csiCall

			// Called for each NodeStateVolume calls, with counter incremented atomically before
			// the invocation (i.e. first value will be 1).
			nodeStageHook func(counter int64) error
		}{
			{
				// This is already tested elsewhere, adding simple good case here to test the test framework.
				name:             "should call NodeUnstage after NodeStage success",
				expectPodRunning: true,
				expectedCalls: []csiCall{
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK, deletePod: true},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
			},
			{
				// Kubelet should repeat NodeStage as long as the pod exists
				name:             "should retry NodeStage after NodeStage final error",
				expectPodRunning: true,
				expectedCalls: []csiCall{
					// This matches all 3 NodeStage calls with InvalidArgument error
					{expectedMethod: "NodeStageVolume", expectedError: codes.InvalidArgument},
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK, deletePod: true},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
				// Fail first 3 NodeStage requests, 4th succeeds
				nodeStageHook: func(counter int64) error {
					if counter < 4 {
						return status.Error(codes.InvalidArgument, "fake error")
					}
					return nil
				},
			},
			{
				// Kubelet should repeat NodeStage as long as the pod exists
				name:             "should retry NodeStage after NodeStage ephemeral error",
				expectPodRunning: true,
				expectedCalls: []csiCall{
					// This matches all 3 NodeStage calls with DeadlineExceeded error
					{expectedMethod: "NodeStageVolume", expectedError: codes.DeadlineExceeded},
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK, deletePod: true},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
				// Fail first 3 NodeStage requests, 4th succeeds
				nodeStageHook: func(counter int64) error {
					if counter < 4 {
						return status.Error(codes.DeadlineExceeded, "fake error")
					}
					return nil
				},
			},
			{
				// After NodeUnstage with ephemeral error, the driver may continue staging the volume.
				// Kubelet should call NodeUnstage to make sure the volume is really unstaged after
				// the pod is deleted.
				name:             "should call NodeUnstage after NodeStage ephemeral error",
				expectPodRunning: false,
				expectedCalls: []csiCall{
					// Delete the pod before NodeStage succeeds - it should get "uncertain" because of ephemeral error
					// This matches all repeated NodeStage calls with DeadlineExceeded error (due to exp. backoff).
					{expectedMethod: "NodeStageVolume", expectedError: codes.DeadlineExceeded, deletePod: true},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
				nodeStageHook: func(counter int64) error {
					return status.Error(codes.DeadlineExceeded, "fake error")
				},
			},
			{
				// After NodeUnstage with final error, kubelet can be sure the volume is not staged.
				// The test checks that NodeUnstage is *not* called.
				name:             "should not call NodeUnstage after NodeStage final error",
				expectPodRunning: false,
				expectedCalls: []csiCall{
					// Delete the pod before NodeStage succeeds - it should get "globally unmounted" because of final error.
					// This matches all repeated NodeStage calls with InvalidArgument error (due to exp. backoff).
					{expectedMethod: "NodeStageVolume", expectedError: codes.InvalidArgument, deletePod: true},
				},
				// nodeStageScript: `INVALIDARGUMENT;`,
				nodeStageHook: func(counter int64) error {
					return status.Error(codes.InvalidArgument, "fake error")
				},
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func(ctx context.Context) {
				var hooks *drivers.Hooks
				if test.nodeStageHook != nil {
					hooks = createPreHook("NodeStageVolume", test.nodeStageHook)
				}
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

				ginkgo.By("Waiting for expected CSI calls")
				// Watch for all calls up to deletePod = true
				timeoutCtx, cancel := context.WithTimeout(ctx, csiPodRunningTimeout)
				defer cancel()
				for {
					if timeoutCtx.Err() != nil {
						framework.Failf("timed out waiting for the CSI call that indicates that the pod can be deleted: %v", test.expectedCalls)
					}
					time.Sleep(1 * time.Second)
					_, index, err := compareCSICalls(timeoutCtx, trackedCalls, test.expectedCalls, m.driver.GetCalls)
					framework.ExpectNoError(err, "while waiting for initial CSI calls")
					if index == 0 {
						// No CSI call received yet
						continue
					}
					// Check the last *received* call wanted the pod to be deleted
					if test.expectedCalls[index-1].deletePod {
						break
					}
				}

				if test.expectPodRunning {
					ginkgo.By("Waiting for pod to be running")
					err := e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
					framework.ExpectNoError(err, "Failed to start pod: %v", err)
				}

				ginkgo.By("Deleting the previously created pod")
				err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
				framework.ExpectNoError(err, "while deleting")

				ginkgo.By("Waiting for all remaining expected CSI calls")
				err = wait.Poll(time.Second, csiUnstageWaitTimeout, func() (done bool, err error) {
					_, index, err := compareCSICalls(ctx, trackedCalls, test.expectedCalls, m.driver.GetCalls)
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
			})
		}
	})

	f.Context("CSI NodeUnstage error cases", f.WithSlow(), func() {
		trackedCalls := []string{
			"NodeStageVolume",
			"NodeUnstageVolume",
		}

		// Each test starts two pods in sequence.
		// The first pod always runs successfully, but NodeUnstage hook can set various error conditions.
		// The test then checks how NodeStage of the second pod is called.
		tests := []struct {
			name          string
			expectedCalls []csiCall

			// Called for each NodeStageVolume calls, with counter incremented atomically before
			// the invocation (i.e. first value will be 1) and index of deleted pod (the first pod
			// has index 1)
			nodeUnstageHook func(counter, pod int64) error
		}{
			{
				// This is already tested elsewhere, adding simple good case here to test the test framework.
				name: "should call NodeStage after NodeUnstage success",
				expectedCalls: []csiCall{
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
			},
			{
				name: "two pods: should call NodeStage after previous NodeUnstage final error",
				expectedCalls: []csiCall{
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.InvalidArgument},
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
				nodeUnstageHook: func(counter, pod int64) error {
					if pod == 1 {
						return status.Error(codes.InvalidArgument, "fake final error")
					}
					return nil
				},
			},
			{
				name: "two pods: should call NodeStage after previous NodeUnstage transient error",
				expectedCalls: []csiCall{
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.DeadlineExceeded},
					{expectedMethod: "NodeStageVolume", expectedError: codes.OK},
					{expectedMethod: "NodeUnstageVolume", expectedError: codes.OK},
				},
				nodeUnstageHook: func(counter, pod int64) error {
					if pod == 1 {
						return status.Error(codes.DeadlineExceeded, "fake transient error")
					}
					return nil
				},
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func(ctx context.Context) {
				// Index of the last deleted pod. NodeUnstage calls are then related to this pod.
				var deletedPodNumber int64 = 1
				var hooks *drivers.Hooks
				if test.nodeUnstageHook != nil {
					hooks = createPreHook("NodeUnstageVolume", func(counter int64) error {
						pod := atomic.LoadInt64(&deletedPodNumber)
						return test.nodeUnstageHook(counter, pod)
					})
				}
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
				framework.ExpectNoError(err, "while waiting for the first pod to start")
				err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
				framework.ExpectNoError(err, "while deleting the first pod")

				// Create the second pod
				pod, err = m.createPodWithPVC(claim)
				framework.ExpectNoError(err, "while creating the second pod")
				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "while waiting for the second pod to start")
				// The second pod is running and kubelet can't call NodeUnstage of the first one.
				// Therefore incrementing the pod counter is safe here.
				atomic.AddInt64(&deletedPodNumber, 1)
				err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
				framework.ExpectNoError(err, "while deleting the second pod")

				ginkgo.By("Waiting for all remaining expected CSI calls")
				err = wait.Poll(time.Second, csiUnstageWaitTimeout, func() (done bool, err error) {
					_, index, err := compareCSICalls(ctx, trackedCalls, test.expectedCalls, m.driver.GetCalls)
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
			})
		}
	})

})
