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
	"math/rand"
	"strconv"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock fsgroup as mount option", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-fsgroup-mount")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	ginkgo.Context("Delegate FSGroup to CSI driver [LinuxOnly]", func() {
		tests := []struct {
			name                   string
			enableVolumeMountGroup bool
		}{
			{
				name:                   "should pass FSGroup to CSI driver if it is set in pod and driver supports VOLUME_MOUNT_GROUP",
				enableVolumeMountGroup: true,
			},
			{
				name:                   "should not pass FSGroup to CSI driver if it is set in pod and driver supports VOLUME_MOUNT_GROUP",
				enableVolumeMountGroup: false,
			},
		}
		for _, t := range tests {
			t := t
			ginkgo.It(t.name, func(ctx context.Context) {
				var nodeStageFsGroup, nodePublishFsGroup string
				if framework.NodeOSDistroIs("windows") {
					e2eskipper.Skipf("FSGroupPolicy is only applied on linux nodes -- skipping")
				}
				m.init(ctx, testParameters{
					disableAttach:          true,
					registerDriver:         true,
					enableVolumeMountGroup: t.enableVolumeMountGroup,
					hooks:                  createFSGroupRequestPreHook(&nodeStageFsGroup, &nodePublishFsGroup),
				})
				ginkgo.DeferCleanup(m.cleanup)

				fsGroupVal := int64(rand.Int63n(20000) + 1024)
				fsGroup := &fsGroupVal
				fsGroupStr := strconv.FormatInt(fsGroupVal, 10 /* base */)

				_, _, pod := m.createPodWithFSGroup(ctx, fsGroup) /* persistent volume */

				err := e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "failed to start pod")

				if t.enableVolumeMountGroup {
					gomega.Expect(nodeStageFsGroup).To(gomega.Equal(fsGroupStr), "Expect NodeStageVolumeRequest.VolumeCapability.MountVolume.VolumeMountGroup to equal %q; got: %q", fsGroupStr, nodeStageFsGroup)
					gomega.Expect(nodePublishFsGroup).To(gomega.Equal(fsGroupStr), "Expect NodePublishVolumeRequest.VolumeCapability.MountVolume.VolumeMountGroup to equal %q; got: %q", fsGroupStr, nodePublishFsGroup)
				} else {
					gomega.Expect(nodeStageFsGroup).To(gomega.BeEmpty(), "Expect NodeStageVolumeRequest.VolumeCapability.MountVolume.VolumeMountGroup to be empty; got: %q", nodeStageFsGroup)
					gomega.Expect(nodePublishFsGroup).To(gomega.BeEmpty(), "Expect NodePublishVolumeRequest.VolumeCapability.MountVolume.VolumeMountGroup to empty; got: %q", nodePublishFsGroup)
				}
			})
		}
	})

})
