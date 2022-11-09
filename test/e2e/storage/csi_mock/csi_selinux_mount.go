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
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock selinux on mount", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-selinux")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	ginkgo.Context("SELinuxMount [LinuxOnly][Feature:SELinux][Feature:SELinuxMountReadWriteOncePod]", func() {
		// Make sure all options are set so system specific defaults are not used.
		seLinuxOpts := v1.SELinuxOptions{
			User:  "system_u",
			Role:  "object_r",
			Type:  "container_file_t",
			Level: "s0:c0,c1",
		}
		seLinuxMountOption := "context=\"system_u:object_r:container_file_t:s0:c0,c1\""

		tests := []struct {
			name                 string
			seLinuxEnabled       bool
			seLinuxSetInPod      bool
			mountOptions         []string
			volumeMode           v1.PersistentVolumeAccessMode
			expectedMountOptions []string
		}{
			{
				name:                 "should pass SELinux mount option for RWOP volume and Pod with SELinux context set",
				seLinuxEnabled:       true,
				seLinuxSetInPod:      true,
				volumeMode:           v1.ReadWriteOncePod,
				expectedMountOptions: []string{seLinuxMountOption},
			},
			{
				name:                 "should add SELinux mount option to existing mount options",
				seLinuxEnabled:       true,
				seLinuxSetInPod:      true,
				mountOptions:         []string{"noexec", "noatime"},
				volumeMode:           v1.ReadWriteOncePod,
				expectedMountOptions: []string{"noexec", "noatime", seLinuxMountOption},
			},
			{
				name:                 "should not pass SELinux mount option for RWO volume",
				seLinuxEnabled:       true,
				seLinuxSetInPod:      true,
				volumeMode:           v1.ReadWriteOnce,
				expectedMountOptions: nil,
			},
			{
				name:                 "should not pass SELinux mount option for Pod without SELinux context",
				seLinuxEnabled:       true,
				seLinuxSetInPod:      false,
				volumeMode:           v1.ReadWriteOncePod,
				expectedMountOptions: nil,
			},
			{
				name:                 "should not pass SELinux mount option for CSI driver that does not support SELinux mount",
				seLinuxEnabled:       false,
				seLinuxSetInPod:      true,
				volumeMode:           v1.ReadWriteOncePod,
				expectedMountOptions: nil,
			},
		}
		for _, t := range tests {
			t := t
			ginkgo.It(t.name, func(ctx context.Context) {
				if framework.NodeOSDistroIs("windows") {
					e2eskipper.Skipf("SELinuxMount is only applied on linux nodes -- skipping")
				}
				var nodeStageMountOpts, nodePublishMountOpts []string
				m.init(testParameters{
					disableAttach:      true,
					registerDriver:     true,
					enableSELinuxMount: &t.seLinuxEnabled,
					hooks:              createSELinuxMountPreHook(&nodeStageMountOpts, &nodePublishMountOpts),
				})
				defer m.cleanup()

				accessModes := []v1.PersistentVolumeAccessMode{t.volumeMode}
				var podSELinuxOpts *v1.SELinuxOptions
				if t.seLinuxSetInPod {
					// Make sure all options are set so system specific defaults are not used.
					podSELinuxOpts = &seLinuxOpts
				}

				_, _, pod := m.createPodWithSELinux(accessModes, t.mountOptions, podSELinuxOpts)
				err := e2epod.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "failed to start pod")

				framework.ExpectEqual(nodeStageMountOpts, t.expectedMountOptions, "Expect NodeStageVolumeRequest.VolumeCapability.MountVolume. to equal %q; got: %q", t.expectedMountOptions, nodeStageMountOpts)
				framework.ExpectEqual(nodePublishMountOpts, t.expectedMountOptions, "Expect NodePublishVolumeRequest.VolumeCapability.MountVolume.VolumeMountGroup to equal %q; got: %q", t.expectedMountOptions, nodeStageMountOpts)
			})
		}
	})
})
