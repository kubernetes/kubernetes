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
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock workload info", func() {
	// The CSIDriverRegistry feature gate is needed for this test in Kubernetes 1.12.
	f := framework.NewDefaultFramework("csi-mock-volumes-workload")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)
	ginkgo.Context("CSI workload information using mock driver", func() {
		var (
			err          error
			podInfoTrue  = true
			podInfoFalse = false
		)
		tests := []struct {
			name                   string
			podInfoOnMount         *bool
			deployClusterRegistrar bool
			expectPodInfo          bool
			expectEphemeral        bool
		}{
			{
				name:                   "should not be passed when podInfoOnMount=nil",
				podInfoOnMount:         nil,
				deployClusterRegistrar: true,
				expectPodInfo:          false,
				expectEphemeral:        false,
			},
			{
				name:                   "should be passed when podInfoOnMount=true",
				podInfoOnMount:         &podInfoTrue,
				deployClusterRegistrar: true,
				expectPodInfo:          true,
				expectEphemeral:        false,
			},
			{
				name:                   "contain ephemeral=true when using inline volume",
				podInfoOnMount:         &podInfoTrue,
				deployClusterRegistrar: true,
				expectPodInfo:          true,
				expectEphemeral:        true,
			},
			{
				name:                   "should not be passed when podInfoOnMount=false",
				podInfoOnMount:         &podInfoFalse,
				deployClusterRegistrar: true,
				expectPodInfo:          false,
				expectEphemeral:        false,
			},
			{
				name:                   "should not be passed when CSIDriver does not exist",
				deployClusterRegistrar: false,
				expectPodInfo:          false,
				expectEphemeral:        false,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(t.name, func(ctx context.Context) {
				m.init(ctx, testParameters{
					registerDriver: test.deployClusterRegistrar,
					podInfo:        test.podInfoOnMount})

				ginkgo.DeferCleanup(m.cleanup)

				withVolume := pvcReference
				if test.expectEphemeral {
					withVolume = csiEphemeral
				}
				_, _, pod := m.createPod(ctx, withVolume)
				if pod == nil {
					return
				}
				err = e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod: %v", err)

				// If we expect an ephemeral volume, the feature has to be enabled.
				// Otherwise need to check if we expect pod info, because the content
				// of that depends on whether the feature is enabled or not.
				csiInlineVolumesEnabled := test.expectEphemeral
				if test.expectPodInfo {
					ginkgo.By("checking for CSIInlineVolumes feature")
					csiInlineVolumesEnabled, err = testsuites.CSIInlineVolumesEnabled(ctx, m.cs, f.Timeouts, f.Namespace.Name)
					framework.ExpectNoError(err, "failed to test for CSIInlineVolumes")
				}

				ginkgo.By("Deleting the previously created pod")
				err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
				framework.ExpectNoError(err, "while deleting")

				ginkgo.By("Checking CSI driver logs")
				err = checkPodLogs(ctx, m.driver.GetCalls, pod, test.expectPodInfo, test.expectEphemeral, csiInlineVolumesEnabled, false, 1)
				framework.ExpectNoError(err)
			})
		}
	})

})
