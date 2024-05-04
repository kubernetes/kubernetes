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
	"errors"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	cachetools "k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock volume storage capacity", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-capacity")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	ginkgo.Context("storage capacity", func() {
		tests := []struct {
			name              string
			resourceExhausted bool
			lateBinding       bool
			topology          bool
		}{
			{
				name: "unlimited",
			},
			{
				name:              "exhausted, immediate binding",
				resourceExhausted: true,
			},
			{
				name:              "exhausted, late binding, no topology",
				resourceExhausted: true,
				lateBinding:       true,
			},
			{
				name:              "exhausted, late binding, with topology",
				resourceExhausted: true,
				lateBinding:       true,
				topology:          true,
			},
		}

		createVolume := "CreateVolume"
		deleteVolume := "DeleteVolume"
		// publishVolume := "NodePublishVolume"
		// unpublishVolume := "NodeUnpublishVolume"
		// stageVolume := "NodeStageVolume"
		// unstageVolume := "NodeUnstageVolume"

		// These calls are assumed to occur in this order for
		// each test run. NodeStageVolume and
		// NodePublishVolume should also be deterministic and
		// only get called once, but sometimes kubelet calls
		// both multiple times, which breaks this test
		// (https://github.com/kubernetes/kubernetes/issues/90250).
		// Therefore they are temporarily commented out until
		// that issue is resolved.
		//
		// NodeUnpublishVolume and NodeUnstageVolume are racing
		// with DeleteVolume, so we cannot assume a deterministic
		// order and have to ignore them
		// (https://github.com/kubernetes/kubernetes/issues/94108).
		deterministicCalls := []string{
			createVolume,
			// stageVolume,
			// publishVolume,
			// unpublishVolume,
			// unstageVolume,
			deleteVolume,
		}

		for _, t := range tests {
			test := t
			ginkgo.It(test.name, ginkgo.NodeTimeout(csiPodRunningTimeout), func(ctx context.Context) {
				var err error
				params := testParameters{
					lateBinding:    test.lateBinding,
					enableTopology: test.topology,

					// Not strictly necessary, but runs a bit faster this way
					// and for a while there also was a problem with a two minuted delay
					// due to a bug (https://github.com/kubernetes-csi/csi-test/pull/250).
					disableAttach:  true,
					registerDriver: true,
				}

				if test.resourceExhausted {
					params.hooks = createPreHook("CreateVolume", func(counter int64) error {
						if counter%2 != 0 {
							return status.Error(codes.ResourceExhausted, "fake error")
						}
						return nil
					})
				}

				m.init(ctx, params)
				ginkgo.DeferCleanup(m.cleanup)

				// In contrast to the raw watch, RetryWatcher is expected to deliver all events even
				// when the underlying raw watch gets closed prematurely
				// (https://github.com/kubernetes/kubernetes/pull/93777#discussion_r467932080).
				// This is important because below the test is going to make assertions about the
				// PVC state changes.
				initResource, err := f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).List(ctx, metav1.ListOptions{})
				framework.ExpectNoError(err, "Failed to fetch initial PVC resource")
				listWatcher := &cachetools.ListWatch{
					WatchFunc: func(listOptions metav1.ListOptions) (watch.Interface, error) {
						return f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Watch(ctx, listOptions)
					},
				}
				pvcWatch, err := watchtools.NewRetryWatcher(initResource.GetResourceVersion(), listWatcher)
				framework.ExpectNoError(err, "create PVC watch")
				defer pvcWatch.Stop()

				sc, claim, pod := m.createPod(ctx, pvcReference)
				gomega.Expect(pod).NotTo(gomega.BeNil(), "while creating pod")
				bindingMode := storagev1.VolumeBindingImmediate
				if test.lateBinding {
					bindingMode = storagev1.VolumeBindingWaitForFirstConsumer
				}
				gomega.Expect(*sc.VolumeBindingMode).To(gomega.Equal(bindingMode), "volume binding mode")

				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "failed to start pod")
				err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
				framework.ExpectNoError(err, "failed to delete pod")
				err = m.cs.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(ctx, claim.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "failed to delete claim")

				normal := []csiCall{}
				for _, method := range deterministicCalls {
					normal = append(normal, csiCall{expectedMethod: method})
				}
				expected := normal
				// When simulating limited capacity,
				// we expect exactly two CreateVolume
				// calls because the first one should
				// have failed.
				if test.resourceExhausted {
					expected = []csiCall{
						{expectedMethod: createVolume, expectedError: codes.ResourceExhausted},
					}
					expected = append(expected, normal...)
				}

				var calls []drivers.MockCSICall
				err = wait.PollUntilContextCancel(ctx, time.Second, true, func(ctx context.Context) (done bool, err error) {
					c, index, err := compareCSICalls(ctx, deterministicCalls, expected, m.driver.GetCalls)
					if err != nil {
						return true, fmt.Errorf("error waiting for expected CSI calls: %w", err)
					}
					calls = c
					if index == 0 {
						// No CSI call received yet
						return false, nil
					}
					if len(expected) == index {
						// all calls received
						return true, nil
					}
					return false, nil
				})
				framework.ExpectNoError(err, "while waiting for all CSI calls")

				// The capacity error is dealt with in two different ways.
				//
				// For delayed binding, the external-provisioner should unset the node annotation
				// to give the scheduler the opportunity to reschedule the pod onto a different
				// node.
				//
				// For immediate binding, the external-scheduler must keep retrying.
				//
				// Unfortunately, the call log is the same in both cases. We have to collect
				// additional evidence that rescheduling really happened. What we have observed
				// above is how the PVC changed over time. Now we can analyze that.
				ginkgo.By("Checking PVC events")
				nodeAnnotationSet := false
				nodeAnnotationReset := false
				watchFailed := false
			loop:
				for {
					select {
					case event, ok := <-pvcWatch.ResultChan():
						if !ok {
							watchFailed = true
							break loop
						}

						framework.Logf("PVC event %s: %#v", event.Type, event.Object)
						switch event.Type {
						case watch.Modified:
							pvc, ok := event.Object.(*v1.PersistentVolumeClaim)
							if !ok {
								framework.Failf("PVC watch sent %#v instead of a PVC", event.Object)
							}
							_, set := pvc.Annotations["volume.kubernetes.io/selected-node"]
							if set {
								nodeAnnotationSet = true
							} else if nodeAnnotationSet {
								nodeAnnotationReset = true
							}
						case watch.Deleted:
							break loop
						case watch.Error:
							watchFailed = true
							break
						}
					case <-ctx.Done():
						framework.Failf("Timeout while waiting to observe PVC list")
					}
				}

				// More tests when capacity is limited.
				if test.resourceExhausted {
					for _, call := range calls {
						if call.Method == createVolume {
							gomega.Expect(call.Error).To(gomega.ContainSubstring("code = ResourceExhausted"), "first CreateVolume error in\n%s", calls)
							break
						}
					}

					switch {
					case watchFailed:
						// If the watch failed or stopped prematurely (which can happen at any time), then we cannot
						// verify whether the annotation was set as expected. This is still considered a successful
						// test.
						framework.Logf("PVC watch delivered incomplete data, cannot check annotation")
					case test.lateBinding:
						gomega.Expect(nodeAnnotationSet).To(gomega.BeTrue(), "selected-node should have been set")
						// Whether it gets reset depends on whether we have topology enabled. Without
						// it, rescheduling is unnecessary.
						if test.topology {
							gomega.Expect(nodeAnnotationReset).To(gomega.BeTrue(), "selected-node should have been set")
						} else {
							gomega.Expect(nodeAnnotationReset).To(gomega.BeFalse(), "selected-node should not have been reset")
						}
					default:
						gomega.Expect(nodeAnnotationSet).To(gomega.BeFalse(), "selected-node should not have been set")
						gomega.Expect(nodeAnnotationReset).To(gomega.BeFalse(), "selected-node should not have been reset")
					}
				}
			})
		}
	})

	// These tests *only* work on a cluster which has the CSIStorageCapacity feature enabled.
	ginkgo.Context("CSIStorageCapacity", func() {
		var (
			err error
			yes = true
			no  = false
		)
		// Tests that expect a failure are slow because we have to wait for a while
		// to be sure that the volume isn't getting created.
		tests := []struct {
			name            string
			storageCapacity *bool
			capacities      []string
			expectFailure   bool
		}{
			{
				name: "CSIStorageCapacity unused",
			},
			{
				name:            "CSIStorageCapacity disabled",
				storageCapacity: &no,
			},
			{
				name:            "CSIStorageCapacity used, no capacity",
				storageCapacity: &yes,
				expectFailure:   true,
			},
			{
				name:            "CSIStorageCapacity used, insufficient capacity",
				storageCapacity: &yes,
				expectFailure:   true,
				capacities:      []string{"1Mi"},
			},
			{
				name:            "CSIStorageCapacity used, have capacity",
				storageCapacity: &yes,
				capacities:      []string{"100Gi"},
			},
			// We could add more test cases here for
			// various situations, but covering those via
			// the scheduler binder unit tests is faster.
		}
		for _, t := range tests {
			test := t
			ginkgo.It(t.name, ginkgo.NodeTimeout(f.Timeouts.PodStart), func(ctx context.Context) {
				scName := "mock-csi-storage-capacity-" + f.UniqueName
				m.init(ctx, testParameters{
					registerDriver:  true,
					scName:          scName,
					storageCapacity: test.storageCapacity,
					lateBinding:     true,
				})
				ginkgo.DeferCleanup(m.cleanup)

				// The storage class uses a random name, therefore we have to create it first
				// before adding CSIStorageCapacity objects for it.
				for _, capacityStr := range test.capacities {
					capacityQuantity := resource.MustParse(capacityStr)
					capacity := &storagev1.CSIStorageCapacity{
						ObjectMeta: metav1.ObjectMeta{
							GenerateName: "fake-capacity-",
						},
						// Empty topology, usable by any node.
						StorageClassName: scName,
						NodeTopology:     &metav1.LabelSelector{},
						Capacity:         &capacityQuantity,
					}
					createdCapacity, err := f.ClientSet.StorageV1().CSIStorageCapacities(f.Namespace.Name).Create(ctx, capacity, metav1.CreateOptions{})
					framework.ExpectNoError(err, "create CSIStorageCapacity %+v", *capacity)
					ginkgo.DeferCleanup(framework.IgnoreNotFound(f.ClientSet.StorageV1().CSIStorageCapacities(f.Namespace.Name).Delete), createdCapacity.Name, metav1.DeleteOptions{})
				}

				// kube-scheduler may need some time before it gets the CSIDriver and CSIStorageCapacity objects.
				// Without them, scheduling doesn't run as expected by the test.
				syncDelay := 5 * time.Second
				time.Sleep(syncDelay)

				sc, _, pod := m.createPod(ctx, pvcReference) // late binding as specified above
				gomega.Expect(sc.Name).To(gomega.Equal(scName), "pre-selected storage class name not used")

				condition := anyOf(
					podRunning(ctx, f.ClientSet, pod.Name, pod.Namespace),
					// We only just created the CSIStorageCapacity objects, therefore
					// we have to ignore all older events, plus the syncDelay as our
					// safety margin.
					podHasStorage(ctx, f.ClientSet, pod.Name, pod.Namespace, time.Now().Add(syncDelay)),
				)
				err = wait.PollImmediateUntil(poll, condition, ctx.Done())
				if test.expectFailure {
					switch {
					case errors.Is(err, context.DeadlineExceeded),
						errors.Is(err, wait.ErrorInterrupted(errors.New("timed out waiting for the condition"))),
						errors.Is(err, errNotEnoughSpace):
						// Okay, we expected that.
					case err == nil:
						framework.Fail("pod unexpectedly started to run")
					default:
						framework.Failf("unexpected error while waiting for pod: %v", err)
					}
				} else {
					framework.ExpectNoError(err, "failed to start pod")
				}

				ginkgo.By("Deleting the previously created pod")
				err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
				framework.ExpectNoError(err, "while deleting")
			})
		}
	})
})
