/*
Copyright 2021 The Kubernetes Authors.

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

package windows

import (
	"context"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = SIGDescribe("[Feature:WindowsHostProcessContainers] [Excluded:WindowsDocker] [MinimumKubeletVersion:1.22] HostProcess containers", func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
		SkipUnlessWindowsHostProcessContainersEnabled()
	})

	f := framework.NewDefaultFramework("host-process-test-windows")

	ginkgo.It("should run as a process on the host/node", func() {

		ginkgo.By("selecting a Windows node")
		targetNode, err := findWindowsNode(f)
		framework.ExpectNoError(err, "Error finding Windows node")
		framework.Logf("Using node: %v", targetNode.Name)

		ginkgo.By("scheduling a pod with a container that verifies %COMPUTERNAME% matches selected node name")
		image := imageutils.GetConfig(imageutils.BusyBox)

		trueVar := true
		podName := "host-process-test-pod"
		user := "NT AUTHORITY\\Local service"
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{
					WindowsOptions: &v1.WindowsSecurityContextOptions{
						HostProcess:   &trueVar,
						RunAsUserName: &user,
					},
				},
				HostNetwork: true,
				Containers: []v1.Container{
					{
						Image:   image.GetE2EImage(),
						Name:    "computer-name-test",
						Command: []string{"cmd.exe", "/K", "IF", "NOT", "%COMPUTERNAME%", "==", targetNode.Name, "(", "exit", "-1", ")"},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				NodeName:      targetNode.Name,
			},
		}

		f.PodClient().Create(pod)

		ginkgo.By("Waiting for pod to run")
		f.PodClient().WaitForFinish(podName, 3*time.Minute)

		ginkgo.By("Then ensuring pod finished running successfully")
		p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(
			context.TODO(),
			podName,
			metav1.GetOptions{})

		framework.ExpectNoError(err, "Error retrieving pod")
		framework.ExpectEqual(p.Status.Phase, v1.PodSucceeded)
	})
})

func SkipUnlessWindowsHostProcessContainersEnabled() {
	if !framework.TestContext.FeatureGates[string(features.WindowsHostProcessContainers)] {
		e2eskipper.Skipf("Skipping test because feature 'WindowsHostProcessContainers' is not enabled")
	}
}
