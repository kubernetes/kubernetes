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

package node

import (
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Ephemeral Containers [NodeFeature:EphemeralContainers]", func() {
	f := framework.NewDefaultFramework("ephemeral-containers-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline
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
		ecName := "debugger"
		ec := &v1.EphemeralContainer{
			EphemeralContainerCommon: v1.EphemeralContainerCommon{
				Name:    ecName,
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: e2epod.GenerateScriptCmd("while true; do echo polo; sleep 2; done"),
				Stdin:   true,
				TTY:     true,
			},
		}
		err := podClient.AddEphemeralContainerSync(pod, ec, time.Minute)
		// BEGIN TODO: Remove when EphemeralContainers feature gate is retired.
		if apierrors.IsNotFound(err) {
			e2eskipper.Skipf("Skipping test because EphemeralContainers feature disabled (error: %q)", err)
		}
		// END TODO: Remove when EphemeralContainers feature gate is retired.
		framework.ExpectNoError(err, "Failed to patch ephemeral containers in pod %q", format.Pod(pod))

		ginkgo.By("checking pod container endpoints")
		// Can't use anything depending on kubectl here because it's not available in the node test environment
		output := f.ExecCommandInContainer(pod.Name, ecName, "/bin/echo", "marco")
		gomega.Expect(output).To(gomega.ContainSubstring("marco"))
		log, err := e2epod.GetPodLogs(f.ClientSet, pod.Namespace, pod.Name, ecName)
		framework.ExpectNoError(err, "Failed to get logs for pod %q ephemeral container %q", format.Pod(pod), ecName)
		gomega.Expect(log).To(gomega.ContainSubstring("polo"))
	})
})
