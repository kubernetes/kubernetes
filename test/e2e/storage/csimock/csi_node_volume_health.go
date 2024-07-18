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
	"strings"
	"time"

	"google.golang.org/grpc/codes"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/features"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = utils.SIGDescribe("CSI Mock Node Volume Health", feature.CSIVolumeHealth, framework.WithFeatureGate(features.CSIVolumeHealth), func() {
	f := framework.NewDefaultFramework("csi-mock-node-volume-health")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	f.Context("CSI Mock Node Volume Health", f.WithSlow(), func() {
		trackedCalls := []string{
			"NodeGetVolumeStats",
		}
		tests := []struct {
			name                        string
			expectedCalls               []csiCall
			nodeVolumeConditionRequired bool
			nodeAbnormalVolumeCondition bool
		}{
			{
				name: "return normal volume stats",
				expectedCalls: []csiCall{
					{
						expectedMethod: "NodeGetVolumeStats",
						expectedError:  codes.OK,
					},
				},
				nodeVolumeConditionRequired: true,
				nodeAbnormalVolumeCondition: false,
			},
			{
				name: "return normal volume stats without volume condition",
				expectedCalls: []csiCall{
					{
						expectedMethod: "NodeGetVolumeStats",
						expectedError:  codes.OK,
					},
				},
				nodeVolumeConditionRequired: false,
				nodeAbnormalVolumeCondition: false,
			},
			{
				name: "return normal volume stats with abnormal volume condition",
				expectedCalls: []csiCall{
					{
						expectedMethod: "NodeGetVolumeStats",
						expectedError:  codes.OK,
					},
				},
				nodeVolumeConditionRequired: true,
				nodeAbnormalVolumeCondition: true,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func(ctx context.Context) {
				// Hooks appear to be required for enableNodeVolumeStat.
				m.init(ctx, testParameters{
					registerDriver:            true,
					enableNodeVolumeCondition: test.nodeVolumeConditionRequired,
					hooks:                     createGetVolumeStatsHook(test.nodeAbnormalVolumeCondition),
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
				// try to use ```csi.NewMetricsCsi(pv.handler).GetMetrics()``` to get metrics from csimock driver but failed.
				// the mocked csidriver register doesn't regist itself to normal csidriver.
				if test.nodeVolumeConditionRequired {
					pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err, "Failed to get pods: %v", err)
					grabber, err := e2emetrics.NewMetricsGrabber(ctx, f.ClientSet, nil, f.ClientConfig(), true, false, false, false, false, false)
					framework.ExpectNoError(err, "creating the metrics grabber")
					waitErr := wait.PollUntilContextTimeout(ctx, 30*time.Second, csiNodeVolumeStatWaitPeriod, true, func(ctx context.Context) (bool, error) {
						framework.Logf("Grabbing Kubelet metrics")
						// Grab kubelet metrics from the node the pod was scheduled on
						var err error
						kubeMetrics, err := grabber.GrabFromKubelet(ctx, pod.Spec.NodeName)
						if err != nil {
							framework.Logf("Error fetching kubelet metrics err: %v", err)
							return false, err
						}
						if !findVolumeConditionMetrics(f.Namespace.Name, claim.Name, kubeMetrics, test.nodeAbnormalVolumeCondition) {
							return false, nil
						}
						return true, nil
					})
					framework.ExpectNoError(waitErr, "call metrics should not have any error")
				}
				framework.ExpectNoError(err, "while waiting for all CSI calls")
			})
		}

	})
})

func findVolumeConditionMetrics(pvcNamespace, pvcName string, kubeMetrics e2emetrics.KubeletMetrics, nodeAbnormalVolumeCondition bool) bool {

	found := false
	framework.Logf("Looking for sample tagged with namespace `%s`, PVC `%s`", pvcNamespace, pvcName)
	for key, value := range kubeMetrics {
		for _, sample := range value {
			framework.Logf("Found sample %++v with key: %s", sample, key)
			samplePVC, ok := sample.Metric["persistentvolumeclaim"]
			if !ok {
				break
			}
			sampleNS, ok := sample.Metric["namespace"]
			if !ok {
				break
			}

			if string(samplePVC) == pvcName && string(sampleNS) == pvcNamespace && strings.Contains(key, kubeletmetrics.VolumeStatsHealthStatusAbnormalKey) {
				if (nodeAbnormalVolumeCondition && sample.Value.String() == "1") || (!nodeAbnormalVolumeCondition && sample.Value.String() == "0") {
					found = true
					break
				}
			}
		}
	}
	return found
}

func createGetVolumeStatsHook(abnormalVolumeCondition bool) *drivers.Hooks {
	return &drivers.Hooks{
		Pre: func(ctx context.Context, fullMethod string, request interface{}) (reply interface{}, err error) {
			if req, ok := request.(*csipbv1.NodeGetVolumeStatsRequest); ok {
				if abnormalVolumeCondition {
					req.VolumePath = "/tmp/csi/health/abnormal"
				}
			}
			return nil, nil
		},
	}

}
