/*
Copyright 2024 The Kubernetes Authors.

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
	"strings"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = utils.SIGDescribe("CSI Mock Node Volume Health", framework.WithFeatureGate(features.CSIVolumeHealth), func() {
	f := framework.NewDefaultFramework("csi-mock-node-volume-health")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	f.Context("CSI Mock Node Volume Health", f.WithSlow(), func() {
		// NodeGetVolumeHealth is probed by the volumehealth manager independently of
		// NodeGetVolumeStats, so call order is not guaranteed.
		trackedCalls := []string{
			"NodeGetVolumeStats",
			"NodeGetVolumeHealth",
		}
		tests := []struct {
			name                     string
			expectedCalls            []csiCall
			forbiddenCalls           []string
			nodeVolumeHealthRequired bool
			volumeHealth             *csipbv1.VolumeHealth_VolumeHealthEntry
			expectedHealth           v1.VolumeHealthCondition
		}{
			{
				name: "return normal volume health",
				expectedCalls: []csiCall{
					{
						expectedMethod: "NodeGetVolumeStats",
						expectedError:  codes.OK,
					},
					{
						expectedMethod: "NodeGetVolumeHealth",
						expectedError:  codes.OK,
					},
				},
				nodeVolumeHealthRequired: true,
				expectedHealth: v1.VolumeHealthCondition{
					Status:  v1.VolumeHealthHealthy,
					Reason:  "VolumeHealthy",
					Message: "Volume is healthy",
				},
			},
			{
				name: "return normal volume stats without volume health",
				expectedCalls: []csiCall{
					{
						expectedMethod: "NodeGetVolumeStats",
						expectedError:  codes.OK,
					},
				},
				forbiddenCalls:           []string{"NodeGetVolumeHealth"},
				nodeVolumeHealthRequired: false,
			},
			{
				name: "return abnormal volume health",
				expectedCalls: []csiCall{
					{
						expectedMethod: "NodeGetVolumeStats",
						expectedError:  codes.OK,
					},
					{
						expectedMethod: "NodeGetVolumeHealth",
						expectedError:  codes.OK,
					},
				},
				nodeVolumeHealthRequired: true,
				volumeHealth: &csipbv1.VolumeHealth_VolumeHealthEntry{
					Status:  csipbv1.VolumeHealthErrorType_INACCESSIBLE,
					Reason:  "AbnormalVolumeHealth",
					Message: "The target path of the volume doesn't exist",
				},
				expectedHealth: v1.VolumeHealthCondition{
					Status:  v1.VolumeHealthInaccessible,
					Reason:  "AbnormalVolumeHealth",
					Message: "The target path of the volume doesn't exist",
				},
			},
			{
				name: "persist unrecognized volume health as Unknown",
				expectedCalls: []csiCall{
					{
						expectedMethod: "NodeGetVolumeStats",
						expectedError:  codes.OK,
					},
					{
						expectedMethod: "NodeGetVolumeHealth",
						expectedError:  codes.OK,
					},
				},
				nodeVolumeHealthRequired: true,
				volumeHealth: &csipbv1.VolumeHealth_VolumeHealthEntry{
					Status:  csipbv1.VolumeHealthErrorType(999),
					Reason:  "FutureHealth",
					Message: "The driver reports an unrecognized volume health status",
				},
				expectedHealth: v1.VolumeHealthCondition{
					Status:  v1.VolumeHealthUnknown,
					Reason:  "FutureHealth",
					Message: "The driver reports an unrecognized volume health status",
				},
			},
		}
		for _, test := range tests {
			ginkgo.It(test.name, func(ctx context.Context) {
				m.init(ctx, testParameters{
					registerDriver:            true,
					enableNodeVolumeCondition: test.nodeVolumeHealthRequired,
					hooks:                     createVolumeHealthHook(test.volumeHealth),
				})
				ginkgo.DeferCleanup(m.cleanup)
				_, claim, pod := m.createPod(ctx, pvcReference)
				if pod == nil {
					return
				}
				// Wait for PVC to get bound to make sure the CSI driver is fully started.
				err := e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, f.ClientSet, f.Namespace.Name, claim.Name, time.Second, framework.ClaimProvisionTimeout)
				framework.ExpectNoError(err, "while waiting for PVC to get provisioned")

				ginkgo.By("Waiting for pod to be running")
				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "wait for running pod")
				ginkgo.By("Waiting for all remaining expected CSI calls")
				err = wait.PollUntilContextTimeout(ctx, time.Second, csiNodeVolumeStatWaitPeriod, true, func(c context.Context) (done bool, err error) {
					return expectedCSICallsSeen(ctx, trackedCalls, test.expectedCalls, test.forbiddenCalls, m.driver.GetCalls)
				})
				framework.ExpectNoError(err, "while waiting for all CSI calls")
				// try to use ```csi.NewMetricsCsi(pv.handler).GetMetrics()``` to get metrics from csimock driver but failed.
				// the mocked csidriver register doesn't regist itself to normal csidriver.
				if test.nodeVolumeHealthRequired {
					pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err, "get running pod")

					waitErr := wait.PollUntilContextTimeout(ctx, 5*time.Second, csiNodeVolumeStatWaitPeriod, true, func(ctx context.Context) (bool, error) {
						updated, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
						if err != nil {
							return false, err
						}
						return podVolumeHealthMatches(updated, test.expectedHealth), nil
					})
					framework.ExpectNoError(waitErr, "pod.status.volumeStatuses[*].healthStatus should reflect NodeGetVolumeHealth")
				}
			})
		}

	})

	f.Context("CSI Mock Node Storage Health", f.WithSlow(), func() {
		healthyCondition := storagev1.StorageHealthCondition{
			Status:  storagev1.StorageHealthy,
			Reason:  "StorageHealthy",
			Message: "Storage subsystems are healthy",
		}

		type storageHealthStep struct {
			description    string
			backendHealth  *csipbv1.NodeGetStorageHealthResponse_StorageBackendHealth
			expectedHealth storagev1.StorageHealthCondition
		}
		tests := []struct {
			name  string
			steps []storageHealthStep
		}{
			{
				name: "should mark CSINode unhealthy and then healthy",
				steps: []storageHealthStep{
					{
						description: "storage backend is unreachable",
						backendHealth: &csipbv1.NodeGetStorageHealthResponse_StorageBackendHealth{
							Status:  csipbv1.StorageHealthErrorType_STORAGE_UNREACHABLE,
							Reason:  "BackendUnavailable",
							Message: "The storage backend is unreachable",
						},
						expectedHealth: storagev1.StorageHealthCondition{
							Status:  storagev1.StorageUnreachable,
							Reason:  "BackendUnavailable",
							Message: "The storage backend is unreachable",
						},
					},
					{
						description:    "storage backend is healthy",
						expectedHealth: healthyCondition,
					},
				},
			},
			{
				name: "should persist unrecognized storage health as StorageHealthUnknown",
				steps: []storageHealthStep{
					{
						description: "storage backend reports an unrecognized status",
						backendHealth: &csipbv1.NodeGetStorageHealthResponse_StorageBackendHealth{
							Status:  csipbv1.StorageHealthErrorType(999),
							Reason:  "FutureHealth",
							Message: "The driver reports an unrecognized storage health status",
						},
						expectedHealth: storagev1.StorageHealthCondition{
							Status:  storagev1.StorageHealthUnknown,
							Reason:  "FutureHealth",
							Message: "The driver reports an unrecognized storage health status",
						},
					},
					{
						description:    "storage backend is healthy",
						expectedHealth: healthyCondition,
					},
				},
			},
		}
		for _, test := range tests {
			ginkgo.It(test.name, func(ctx context.Context) {
				var nodeName, driverName string
				// Register before m.init so this runs after the driver's deferred
				// teardown. DeferCleanup callbacks run in reverse registration order.
				ginkgo.DeferCleanup(func(ctx context.Context) {
					checkCSINodeDriverRemoved(ctx, f, nodeName, driverName)
				})

				var backendHealth atomic.Pointer[csipbv1.NodeGetStorageHealthResponse_StorageBackendHealth]
				backendHealth.Store(test.steps[0].backendHealth)
				m.init(ctx, testParameters{
					registerDriver:          true,
					enableNodeStorageHealth: true,
					hooks:                   createStorageHealthHook(&backendHealth),
				})
				ginkgo.DeferCleanup(m.cleanup)

				nodeName = m.config.ClientNodeSelection.Name
				driverName = m.config.GetUniqueDriverName()

				for _, step := range test.steps {
					backendHealth.Store(step.backendHealth)
					ginkgo.By(fmt.Sprintf("Waiting for CSINode to report %s: %s/%s", step.description, step.expectedHealth.Status, step.expectedHealth.Reason))
					err := wait.PollUntilContextTimeout(ctx, time.Second, csiNodeVolumeStatWaitPeriod, true, func(ctx context.Context) (bool, error) {
						csiNode, err := f.ClientSet.StorageV1().CSINodes().Get(ctx, nodeName, metav1.GetOptions{})
						if err != nil {
							return false, err
						}
						return csiNodeStorageHealthMatches(csiNode, driverName, step.expectedHealth), nil
					})
					framework.ExpectNoError(err, "CSINode.status.storageHealth should reflect NodeGetStorageHealth when %s", step.description)
				}
			})
		}
	})
})

// expectedCSICallsSeen reports whether every expected CSI method has been observed
// (order-independent) and that no forbidden methods appear. Health and stats are
// probed on independent kubelet paths, so ordered matching is not reliable.
func expectedCSICallsSeen(
	ctx context.Context,
	trackedCalls []string,
	expectedCalls []csiCall,
	forbiddenCalls []string,
	getCalls func(ctx context.Context) ([]drivers.MockCSICall, error),
) (bool, error) {
	allCalls, err := getCalls(ctx)
	if err != nil {
		framework.Logf("intermittent (?) log retrieval error, proceeding without output: %v", err)
		return false, nil
	}

	tracked := sets.NewString(trackedCalls...)
	forbidden := sets.NewString(forbiddenCalls...)
	seenOK := sets.NewString()
	for _, c := range allCalls {
		if !tracked.Has(c.Method) {
			continue
		}
		if forbidden.Has(c.Method) {
			return true, fmt.Errorf("unexpected CSI call %s (%d)", c.Method, c.FullError.Code)
		}
		for _, expected := range expectedCalls {
			if c.Method == expected.expectedMethod && c.FullError.Code == expected.expectedError {
				seenOK.Insert(expected.expectedMethod)
			}
		}
	}
	for _, expected := range expectedCalls {
		if !seenOK.Has(expected.expectedMethod) {
			return false, nil
		}
	}
	return true, nil
}

func podVolumeHealthMatches(pod *v1.Pod, expected v1.VolumeHealthCondition) bool {
	if len(pod.Status.VolumeStatuses) == 0 {
		return false
	}
	var entry *v1.VolumeHealthStatus
	for i := range pod.Status.VolumeStatuses {
		vh := &pod.Status.VolumeStatuses[i]
		if vh.Name != "" && vh.HealthStatus != nil {
			entry = vh.HealthStatus
			break
		}
	}
	return entry != nil && len(entry.HealthConditions) == 1 && entry.HealthConditions[0] == expected
}

// csiNodeStorageHealthMatches reports whether the driver's CSINode storage health
// consists of exactly the expected condition. LastTransitionTime is ignored.
func csiNodeStorageHealthMatches(csiNode *storagev1.CSINode, driverName string, expected storagev1.StorageHealthCondition) bool {
	for _, health := range csiNode.Status.StorageHealth {
		if health.Name != driverName {
			continue
		}
		if len(health.HealthConditions) != 1 {
			return false
		}
		got := health.HealthConditions[0]
		return got.Status == expected.Status &&
			got.Reason == expected.Reason &&
			got.Message == expected.Message &&
			got.AccessMode == nil &&
			got.VolumeMode == nil
	}
	return false
}

func checkCSINodeDriverRemoved(ctx context.Context, f *framework.Framework, nodeName, driverName string) {
	if driverName == "" {
		return
	}
	ginkgo.By("Waiting for the uninstalled driver's registration and storage health to be removed from CSINode")
	err := wait.PollUntilContextTimeout(ctx, time.Second, csiNodeVolumeStatWaitPeriod, true, func(ctx context.Context) (bool, error) {
		csiNode, err := f.ClientSet.StorageV1().CSINodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, driver := range csiNode.Spec.Drivers {
			if driver.Name == driverName {
				return false, nil
			}
		}
		for _, health := range csiNode.Status.StorageHealth {
			if health.Name == driverName {
				return false, nil
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "driver %q should be removed from CSINode %q spec.drivers and status.storageHealth after uninstall", driverName, nodeName)
}

func createVolumeHealthHook(health *csipbv1.VolumeHealth_VolumeHealthEntry) *drivers.Hooks {
	return &drivers.Hooks{
		Post: func(ctx context.Context, fullMethod string, request, reply interface{}, err error) (interface{}, error) {
			if !strings.Contains(fullMethod, "NodeGetVolumeHealth") {
				return reply, err
			}
			if resp, ok := reply.(*csipbv1.NodeGetVolumeHealthResponse); ok && health != nil {
				if resp.VolumeHealth == nil {
					resp.VolumeHealth = &csipbv1.VolumeHealth{}
				}
				resp.VolumeHealth.HealthStatuses = []*csipbv1.VolumeHealth_VolumeHealthEntry{health}
				return resp, nil
			}
			return reply, err
		},
	}
}

// createStorageHealthHook replaces the NodeGetStorageHealth response with the entry
// currently stored in backendHealth. A nil entry leaves the mock driver's empty
// response untouched, which kubelet reports as Healthy.
func createStorageHealthHook(backendHealth *atomic.Pointer[csipbv1.NodeGetStorageHealthResponse_StorageBackendHealth]) *drivers.Hooks {
	return &drivers.Hooks{
		Post: func(ctx context.Context, fullMethod string, request, reply interface{}, err error) (interface{}, error) {
			if !strings.Contains(fullMethod, "NodeGetStorageHealth") {
				return reply, err
			}
			entry := backendHealth.Load()
			if entry == nil {
				return reply, err
			}
			resp, ok := reply.(*csipbv1.NodeGetStorageHealthResponse)
			if !ok {
				return reply, err
			}
			resp.BackendHealth = []*csipbv1.NodeGetStorageHealthResponse_StorageBackendHealth{entry}
			return resp, err
		},
	}
}
