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
	"fmt"
	"math/rand"
	"strconv"
	"time"

	"github.com/onsi/ginkgo/v2"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("CSI Mock volume fsgroup policies", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-fsgroup-policy")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	m := newMockDriverSetup(f)

	// These tests *only* work on a cluster which has the CSIVolumeFSGroupPolicy feature enabled.
	ginkgo.Context("CSI FSGroupPolicy [LinuxOnly]", func() {
		tests := []struct {
			name          string
			fsGroupPolicy storagev1.FSGroupPolicy
			modified      bool
		}{
			{
				name:          "should modify fsGroup if fsGroupPolicy=default",
				fsGroupPolicy: storagev1.ReadWriteOnceWithFSTypeFSGroupPolicy,
				modified:      true,
			},
			{
				name:          "should modify fsGroup if fsGroupPolicy=File",
				fsGroupPolicy: storagev1.FileFSGroupPolicy,
				modified:      true,
			},
			{
				name:          "should not modify fsGroup if fsGroupPolicy=None",
				fsGroupPolicy: storagev1.NoneFSGroupPolicy,
				modified:      false,
			},
		}
		for _, t := range tests {
			test := t
			ginkgo.It(test.name, func(ctx context.Context) {
				if framework.NodeOSDistroIs("windows") {
					e2eskipper.Skipf("FSGroupPolicy is only applied on linux nodes -- skipping")
				}
				m.init(ctx, testParameters{
					disableAttach:  true,
					registerDriver: true,
					fsGroupPolicy:  &test.fsGroupPolicy,
				})
				ginkgo.DeferCleanup(m.cleanup)

				// kube-scheduler may need some time before it gets the CSIDriver object.
				// Without them, scheduling doesn't run as expected by the test.
				syncDelay := 5 * time.Second
				time.Sleep(syncDelay)

				fsGroupVal := int64(rand.Int63n(20000) + 1024)
				fsGroup := &fsGroupVal

				_, _, pod := m.createPodWithFSGroup(ctx, fsGroup) /* persistent volume */

				mountPath := pod.Spec.Containers[0].VolumeMounts[0].MountPath
				dirName := mountPath + "/" + f.UniqueName
				fileName := dirName + "/" + f.UniqueName

				err := e2epod.WaitForPodNameRunningInNamespace(ctx, m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "failed to start pod")

				// Create the subdirectory to ensure that fsGroup propagates
				createDirectory := fmt.Sprintf("mkdir %s", dirName)
				_, _, err = e2evolume.PodExec(f, pod, createDirectory)
				framework.ExpectNoError(err, "failed: creating the directory: %s", err)

				// Inject the contents onto the mount
				createFile := fmt.Sprintf("echo '%s' > '%s'; sync", "filecontents", fileName)
				_, _, err = e2evolume.PodExec(f, pod, createFile)
				framework.ExpectNoError(err, "failed: writing the contents: %s", err)

				// Delete the created file. This step is mandatory, as the mock driver
				// won't clean up the contents automatically.
				defer func() {
					deleteDir := fmt.Sprintf("rm -fr %s", dirName)
					_, _, err = e2evolume.PodExec(f, pod, deleteDir)
					framework.ExpectNoError(err, "failed: deleting the directory: %s", err)
				}()

				// Ensure that the fsGroup matches what we expect
				if test.modified {
					utils.VerifyFSGroupInPod(f, fileName, strconv.FormatInt(*fsGroup, 10), pod)
				} else {
					utils.VerifyFSGroupInPod(f, fileName, "root", pod)
				}

				// The created resources will be removed by the cleanup() function,
				// so need to delete anything here.
			})
		}
	})
})
