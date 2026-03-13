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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock workload info", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-workload")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)
	ginkgo.Context("CSI workload information using mock driver", func() {
		tests := []struct {
			name                   string
			podInfoOnMount         bool
			deployClusterRegistrar bool
			expectPodInfo          bool
			expectEphemeral        bool
		}{
			{
				name:                   "should be passed when podInfoOnMount=true",
				podInfoOnMount:         true,
				deployClusterRegistrar: true,
				expectPodInfo:          true,
				expectEphemeral:        false,
			},
			{
				name:                   "contain ephemeral=true when using inline volume",
				podInfoOnMount:         true,
				deployClusterRegistrar: true,
				expectPodInfo:          true,
				expectEphemeral:        true,
			},
			{
				name:                   "should not be passed when podInfoOnMount=false",
				podInfoOnMount:         false,
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
					podInfo:        &test.podInfoOnMount})

				ginkgo.DeferCleanup(m.cleanup)

				waitUntilPodInfoInLog(ctx, m, test.expectPodInfo, test.expectEphemeral)

			})
		}
	})

	ginkgo.Context("CSI PodInfoOnMount Update", func() {
		tests := []struct {
			name              string
			oldPodInfoOnMount bool
			newPodInfoOnMount bool
		}{
			{
				name:              "should not be passed when update from true to false",
				oldPodInfoOnMount: true,
				newPodInfoOnMount: false,
			},
			{
				name:              "should be passed when update from false to true",
				oldPodInfoOnMount: false,
				newPodInfoOnMount: true,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(t.name, func(ctx context.Context) {
				m.init(ctx, testParameters{
					registerDriver: true,
					podInfo:        &test.oldPodInfoOnMount})

				ginkgo.DeferCleanup(m.cleanup)

				waitUntilPodInfoInLog(ctx, m, test.oldPodInfoOnMount, false)
				m.update(utils.PatchCSIOptions{PodInfo: &test.newPodInfoOnMount})
				waitUntilPodInfoInLog(ctx, m, test.newPodInfoOnMount, false)
			})
		}
	})

})

func waitUntilPodInfoInLog(ctx context.Context, m *mockDriverSetup, expectPodInfo, expectEphemeral bool) {
	var err error

	utils.WaitUntil(framework.Poll, framework.PodStartTimeout, func() bool {
		err = gomega.InterceptGomegaFailure(func() {
			withVolume := pvcReference
			if expectEphemeral {
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
			csiInlineVolumesEnabled := expectEphemeral
			if expectPodInfo {
				ginkgo.By("checking for CSIInlineVolumes feature")
				csiInlineVolumesEnabled, err = testsuites.CSIInlineVolumesEnabled(ctx, m.cs, m.f.Timeouts, m.f.Namespace.Name)
				framework.ExpectNoError(err, "failed to test for CSIInlineVolumes")
			}

			ginkgo.By("Deleting the previously created pod")
			err = e2epod.DeletePodWithWait(ctx, m.cs, pod)
			framework.ExpectNoError(err, "while deleting")

			ginkgo.By("Checking CSI driver logs")
			err = checkNodePublishVolume(ctx, m.driver.GetCalls, pod, expectPodInfo, expectEphemeral, csiInlineVolumesEnabled, false, false)
			framework.ExpectNoError(err)
		})

		return err == nil
	})

	framework.ExpectNoError(err, "failed: verifing PodInfo: %s", err)
}
