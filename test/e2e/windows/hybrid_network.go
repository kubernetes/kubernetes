/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	linuxOS   = "linux"
	windowsOS = "windows"
)

var (
	windowsBusyBoximage = imageutils.GetE2EImage(imageutils.TestWebserver)
	linuxBusyBoxImage   = "docker.io/library/nginx:1.15-alpine"
)

var _ = SIGDescribe("Hybrid cluster network", func() {
	f := framework.NewDefaultFramework("hybrid-network")

	BeforeEach(func() {
		framework.SkipUnlessNodeOSDistroIs("windows")
	})

	Context("for all supported CNIs", func() {

		It("should have stable external networking for linux and windows pods", func() {

			By("creating a linux pod with external requests")
			linuxPod := CreateTestPod(f, linuxBusyBoxImage, linuxOS)
			cmd := []string{"/bin/sh", "-c", "nc -vz 8.8.8.8 53", "||", "nc -vz 8.8.4.4 53"}
			checkExternal(f, linuxPod, linuxOS, cmd)

			By("creating a windows pod with external requests")
			windowsPod := CreateTestPod(f, windowsBusyBoximage, windowsOS)
			cmd = []string{"powershell", "-command", "$r=invoke-webrequest www.google.com -usebasicparsing; echo $r.StatusCode"}
			out, msg := checkExternal(f, windowsPod, windowsOS, cmd)
			Expect(out).To(Equal("200"), msg)
		})

	})
})

func checkExternal(f *framework.Framework, podName string, os string, cmd []string]) (string, string) {
	By(fmt.Sprintf("checking external connectivity of %s-container in %s pod", os, podName))
	out, stderr, err := f.ExecCommandInContainerWithFullOutput(podName, os + "-container", cmd...)
	msg := fmt.Sprintf("cmd: %v, stdout: %q, stderr: %q", cmd, out, stderr)
	Expect(err).NotTo(HaveOccurred(), msg)
	return out, msg
}

func CreateTestPod(f *framework.Framework, image string, os string) string {
	ns := f.Namespace.Name
	cs := f.ClientSet

	podName := "pod-" + string(uuid.NewUUID())
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  os + "-container",
					Image: image,
				},
			},
			NodeSelector: map[string]string{
				"beta.kubernetes.io/os": os,
			},
		},
	}
	if os == linuxOS {
		pod.Spec.Tolerations = []v1.Toleration{
			{
				Key:      "key",
				Operator: v1.TolerationOpExists,
				Effect:   v1.TaintEffectNoSchedule,
			},
		}
	}
	_, err := cs.CoreV1().Pods(ns).Create(pod)
	framework.ExpectNoError(err)
	framework.ExpectNoError(f.WaitForPodRunning(podName))
	return podName
}
