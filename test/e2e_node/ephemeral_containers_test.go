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

package e2enode

import (
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("Ephemeral Containers [Feature:EphemeralContainers][NodeAlphaFeature:EphemeralContainers]", func() {
	f := framework.NewDefaultFramework("ephemeral-containers-test")
	var podClient *framework.PodClient
	ginkgo.BeforeEach(func() {
		podClient = f.PodClient()
	})

	ginkgo.It("will start an ephemeral container in an existing pod", func() {
		ginkgo.By("creating a target pod")
		pod := podClient.CreateSync(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "ephemeral-containers-target-pod"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "test-container-1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sleep"},
						Args:    []string{"10000"},
					},
				},
			},
		})

		ginkgo.By("adding an ephemeral container")
		ec := &v1.EphemeralContainer{
			EphemeralContainerCommon: v1.EphemeralContainerCommon{
				Name:    "debugger",
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"/bin/sh"},
				Stdin:   true,
				TTY:     true,
			},
		}
		podClient.AddEphemeralContainerSync(pod, ec, time.Minute)

		ginkgo.By("confirm that the container is really running")
		marco := f.ExecCommandInContainer(pod.Name, "debugger", "/bin/echo", "polo")
		framework.ExpectEqual(marco, "polo")
	})
})
