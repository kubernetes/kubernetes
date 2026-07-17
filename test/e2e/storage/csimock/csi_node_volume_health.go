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
			nodeAbnormalVolumeHealth bool
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
				nodeAbnormalVolumeHealth: false,
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
				nodeAbnormalVolumeHealth: false,
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
				nodeAbnormalVolumeHealth: true,
			},
		}
		for _, test := range tests {
			ginkgo.It(test.name, func(ctx context.Context) {
				m.init(ctx, testParameters{
					registerDriver:            true,
					enableNodeVolumeCondition: test.nodeVolumeHealthRequired,
					hooks:                     createVolumeHealthHook(test.nodeAbnormalVolumeHealth),
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
						return podVolumeHealthMatches(updated, test.nodeAbnormalVolumeHealth), nil
					})
					framework.ExpectNoError(waitErr, "pod.status.volumeHealth should reflect NodeGetVolumeHealth")
				}
			})
		}

	})

	f.Context("CSI Mock Node Storage Health", f.WithSlow(), func() {
		ginkgo.It("should mark CSINode unhealthy and then healthy", func(ctx context.Context) {
			var unhealthy atomic.Bool
			unhealthy.Store(true)
			m.init(ctx, testParameters{
				registerDriver:          true,
				enableNodeStorageHealth: true,
				hooks:                   createStorageHealthHook(&unhealthy),
			})
			ginkgo.DeferCleanup(m.cleanup)

			nodeName := m.config.ClientNodeSelection.Name
			driverName := m.config.GetUniqueDriverName()

			ginkgo.By("Waiting for CSINode to report the storage backend as unreachable")
			err := wait.PollUntilContextTimeout(ctx, time.Second, csiNodeVolumeStatWaitPeriod, true, func(ctx context.Context) (bool, error) {
				csiNode, err := f.ClientSet.StorageV1().CSINodes().Get(ctx, nodeName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				return csiNodeStorageHealthMatches(csiNode, driverName, true), nil
			})
			framework.ExpectNoError(err, "CSINode should report the driver's storage backend as unreachable")

			ginkgo.By("Making the storage backend healthy")
			unhealthy.Store(false)

			ginkgo.By("Waiting for CSINode to clear the storage health condition")
			err = wait.PollUntilContextTimeout(ctx, time.Second, csiNodeVolumeStatWaitPeriod, true, func(ctx context.Context) (bool, error) {
				csiNode, err := f.ClientSet.StorageV1().CSINodes().Get(ctx, nodeName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				return csiNodeStorageHealthMatches(csiNode, driverName, false), nil
			})
			framework.ExpectNoError(err, "CSINode should clear the driver's storage health condition")
		})
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

func podVolumeHealthMatches(pod *v1.Pod, abnormal bool) bool {
	var entry *v1.PodVolumeHealth
	for i := range pod.Status.VolumeHealth {
		vh := &pod.Status.VolumeHealth[i]
		if vh.Name != "" {
			entry = vh
			break
		}
	}
	if !abnormal {
		// Healthy: either no VolumeHealth entry, or an entry with empty conditions.
		return entry == nil || len(entry.HealthConditions) == 0
	}
	if entry == nil || len(entry.HealthConditions) == 0 {
		return false
	}

	for _, c := range entry.HealthConditions {
		if c.Status == v1.VolumeHealthInaccessible && c.Reason == "AbnormalVolumeHealth" {
			return true
		}
	}
	return false
}

func csiNodeStorageHealthMatches(csiNode *storagev1.CSINode, driverName string, unhealthy bool) bool {
	for _, condition := range csiNode.Status.StorageHealth {
		if condition.Name != driverName {
			continue
		}
		if !unhealthy {
			return false
		}
		if condition.Status == storagev1.StorageUnreachable && condition.Reason == "BackendUnavailable" {
			return true
		}
	}
	return !unhealthy
}

func createVolumeHealthHook(abnormalVolumeHealth bool) *drivers.Hooks {
	return &drivers.Hooks{
		Post: func(ctx context.Context, fullMethod string, request, reply interface{}, err error) (interface{}, error) {
			if !strings.Contains(fullMethod, "NodeGetVolumeHealth") {
				return reply, err
			}
			if resp, ok := reply.(*csipbv1.NodeGetVolumeHealthResponse); ok && abnormalVolumeHealth {
				if resp.VolumeHealth == nil {
					resp.VolumeHealth = &csipbv1.VolumeHealth{}
				}
				resp.VolumeHealth.HealthStatuses = []*csipbv1.VolumeHealth_VolumeHealthEntry{
					{
						Status:  csipbv1.VolumeHealthErrorType_INACCESSIBLE,
						Reason:  "AbnormalVolumeHealth",
						Message: "The target path of the volume doesn't exist",
					},
				}
				return resp, nil
			}
			return reply, err
		},
	}
}

func createStorageHealthHook(unhealthy *atomic.Bool) *drivers.Hooks {
	return &drivers.Hooks{
		Post: func(ctx context.Context, fullMethod string, request, reply interface{}, err error) (interface{}, error) {
			if !strings.Contains(fullMethod, "NodeGetStorageHealth") || !unhealthy.Load() {
				return reply, err
			}
			resp, ok := reply.(*csipbv1.NodeGetStorageHealthResponse)
			if !ok {
				return reply, err
			}
			resp.BackendHealth = []*csipbv1.NodeGetStorageHealthResponse_StorageBackendHealth{
				{
					Status:  csipbv1.StorageHealthErrorType_STORAGE_UNREACHABLE,
					Reason:  "BackendUnavailable",
					Message: "The storage backend is unreachable",
				},
			}
			return resp, err
		},
	}
}
