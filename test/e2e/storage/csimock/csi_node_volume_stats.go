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
	"time"

	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = utils.SIGDescribe("CSI Mock Node Volume Stats", func() {
	f := framework.NewDefaultFramework("csi-mock-node-volume-stats")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	f.Context("CSI Mock Node Volume Stats", f.WithSlow(), func() {
		trackedCalls := []string{
			"NodeGetVolumeStats",
		}
		tests := []struct {
			name                   string
			expectedCalls          []csiCall
			nodeVolumeStatRequired bool
			nodeGetVolumeStatsHook func(counter int64) error
		}{
			{
				name:                   "return abnormal volume stats",
				expectedCalls:          []csiCall{},
				nodeVolumeStatRequired: false,
				nodeGetVolumeStatsHook: func(counter int64) error {
					return nil
				},
			},
			{
				name: "return normal volume stats",
				expectedCalls: []csiCall{
					{
						expectedMethod: "NodeGetVolumeStats",
						expectedError:  codes.OK,
					},
				},
				nodeVolumeStatRequired: true,
				nodeGetVolumeStatsHook: func(counter int64) error {
					return nil
				},
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func(ctx context.Context) {
				ginkgo.By(fmt.Sprintf("volume stats: %+v", test))
				// Hooks appear to be required for enableNodeVolumeStat.
				hooks := createPreHook("NodeGetVolumeStats", test.nodeGetVolumeStatsHook)
				m.init(ctx, testParameters{
					registerDriver:       true,
					enableNodeVolumeStat: test.nodeVolumeStatRequired,
					hooks:                hooks,
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
				framework.ExpectNoError(err, "Failed to start pod: %v", err)

				ginkgo.By("Waiting for all remaining expected CSI calls")
				err = wait.PollUntilContextTimeout(ctx, time.Second, csiNodeVolumeStatWaitPeriod, true, func(c context.Context) (done bool, err error) {
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
				if test.nodeVolumeStatRequired {
					framework.ExpectNoError(err, "while waiting for all CSI calls")
				} else {
					gomega.Expect(err).To(gomega.HaveOccurred(), "an error should have occurred")
				}
			})
		}

	})
})
