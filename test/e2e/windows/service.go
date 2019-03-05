/*
Copyright 2019 The Kubernetes Authors.

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
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	defaultWindowsImage = "mcr.microsoft.com/windows/nanoserver:1809"
)

var (
	defaultWindowsPod = &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "windows-tester-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "ping-pod",
					Image:           defaultWindowsImage,
					Command:         []string{"cmd"},
					Args:            []string{"-c"},
					SecurityContext: securitycontext.ValidSecurityContextWithContainerDefaults(),
				},
			},
			RestartPolicy: v1.RestartPolicyOnFailure,
		},
	}
)

var _ = SIGDescribe("Services", func() {
	f := framework.NewDefaultFramework("services")

	var cs clientset.Interface

	BeforeEach(func() {
		cs = f.ClientSet
	})
	//Only for Windows containers
	It("should be able to create a functioning NodePort service for Windows", func() {
		serviceName := "nodeport-test"
		ns := f.Namespace.Name

		jig := framework.NewServiceTestJig(cs, serviceName)
		nodeIP := framework.PickNodeIP(jig.Client) // for later

		By("creating service " + serviceName + " with type=NodePort in namespace " + ns)
		service := jig.CreateTCPServiceOrFail(ns, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
		})
		jig.SanityCheckService(service, v1.ServiceTypeNodePort)
		nodePort := int(service.Spec.Ports[0].NodePort)

		By("creating Pod to be part of service " + serviceName)
		jig.RunOrFail(ns, nil)

		// Create a testing Pod so we can ping the nodeport from within the cluster
		By(fmt.Sprintf("creating testing Pod to curl http://%s:%d", nodeIP, nodePort))
		windowsSelector := map[string]string{"beta.kubernetes.io/os": "windows"}
		winPodSpec := defaultWindowsPod.DeepCopy()
		winPodSpec.Spec.NodeSelector = windowsSelector
		curl := fmt.Sprintf("curl.exe -s -o ./curl-output.txt -w \"%%{http_code}\" http://%s:%d", nodeIP, nodePort)
		winPodSpec.Spec.Containers[0].Args = []string{"cmd", "/c", curl}
		pod, err := cs.CoreV1().Pods(ns).Create(winPodSpec)
		Expect(err).NotTo(HaveOccurred())

		By("waiting for Pod to be running")
		err = f.WaitForPodRunning(pod.Name)
		Expect(err).NotTo(HaveOccurred(),
			"Error waiting for Pod %s to run", pod.Name)

		By("obtaining the logs of the command")
		logs, err := framework.GetPodLogs(cs, ns, pod.Name, pod.Spec.Containers[0].Name)
		Expect(err).NotTo(HaveOccurred(),
			"Error getting logs from Pod %s in namespace %s", pod.Name, ns)
		if !strings.Contains(logs, "200") {
			Fail("Error getting 200 from NodePort")
		}
		framework.Logf("Request made to NodePort and obtained: %s", logs)
	})

})
