/*
Copyright 2020 The Kubernetes Authors.

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

/* This test check that SecurityContext parameters specified at the
 * pod or the container level work as intended. These tests cannot be
 * run when the 'SecurityContextDeny' admission controller is not used
 * so they are skipped by default.
 */

package node

import (
	"crypto/rand"
	"fmt"
	"math/big"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

func generatePodName(base string) string {
	id, err := rand.Int(rand.Reader, big.NewInt(214748))
	if err != nil {
		return base
	}
	return fmt.Sprintf("%s-%d", base, id)
}

func testPod() *v1.Pod {
	podName := generatePodName("hostfqdn")
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Labels:      map[string]string{"name": podName},
			Annotations: map[string]string{},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test-container",
					Image: imageutils.GetE2EImage(imageutils.BusyBox),
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	return pod
}

var _ = SIGDescribe("Hostname of Pod [Feature:HostnameFQDN]", func() {
	f := framework.NewDefaultFramework("hostfqdn")

	/*
	   Release : v1.19
	   Testname: Create Pod without fully qualified domain name (FQDN)
	   Description: A Pod that does not define the subdomain field in it spec, does not have FQDN.
	*/
	ginkgo.It("a pod without subdomain field does not have FQDN [Feature:HostnameFQDN]", func() {
		pod := testPod()
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)"}
		output := []string{fmt.Sprintf("%s;%s", pod.ObjectMeta.Name, pod.ObjectMeta.Name)}
		// Create Pod
		f.TestContainerOutput("shotname only", pod, 0, output)
	})

	/*
	   Release : v1.19
	   Testname: Create Pod without FQDN, hostnameFQDN field set to true
	   Description: A Pod that does not define the subdomain field in it spec, does not have FQDN.
	                Hence, HostnameFQDN feature has no effect.
	*/
	ginkgo.It("a pod without FQDN is not affected by HostnameFQDN field [Feature:HostnameFQDN]", func() {
		pod := testPod()
		// Setting hostnameFQDN field to true should have no effect.
		pod.Spec.HostnameFQDN = true
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)"}
		output := []string{fmt.Sprintf("%s;%s", pod.ObjectMeta.Name, pod.ObjectMeta.Name)}
		// Create Pod
		f.TestContainerOutput("shotname only", pod, 0, output)
	})

	/*
	   Release : v1.19
	   Testname: Create Pod with FQDN, hostnameFQDN field not defined.
	   Description: A Pod that defines the subdomain field in it spec has FQDN.
	                hostname command returns shortname (pod name in this case), and hostname -f returns FQDN.
	*/
	ginkgo.It("a pod with subdomain field has FQDN, hostname is shortname [Feature:HostnameFQDN]", func() {
		pod := testPod()
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)"}
		subdomain := "t"
		// Set PodSpec subdomain field to generate FQDN for pod
		pod.Spec.Subdomain = subdomain
		// Expected Pod FQDN
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", pod.ObjectMeta.Name, subdomain, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		output := []string{fmt.Sprintf("%s;%s", pod.ObjectMeta.Name, hostFQDN)}
		// Create Pod
		f.TestContainerOutput("shotname and fqdn", pod, 0, output)
	})

	/*
		   Release : v1.19
		   Testname: Create Pod with FQDN, hostnameFQDN field set to true.
		   Description: A Pod that defines the subdomain field in it spec has FQDN. When hostnameFQDN: true, the
				hostname is set to be the FQDN. In this case, both commands hostname and hostname -f return
				the FQDN of the Pod.
	*/
	ginkgo.It("a pod with subdomain field has FQDN, when hostnameFQDN is set to true, the FQDN is set as hostname [Feature:HostnameFQDN]", func() {
		pod := testPod()
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)"}
		subdomain := "t"
		// Set PodSpec subdomain field to generate FQDN for pod
		pod.Spec.Subdomain = subdomain
		// Set PodSpec hostnameFQDN to set FQDN as hostname
		pod.Spec.HostnameFQDN = true
		// Expected Pod FQDN
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", pod.ObjectMeta.Name, subdomain, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		// Fail if FQDN is longer than 63 characters, otherwise the Pod will remain pending until test timeout.
		// In Linux, 63 characters is the limit of the hostname kernel field, which this test sets to the pod FQDN.
		framework.ExpectEqual(len(hostFQDN) < 64, true, "The FQDN of the Pod cannot be longer than 63 characters")
		output := []string{fmt.Sprintf("%s;%s", hostFQDN, hostFQDN)}
		// Create Pod
		f.TestContainerOutput("fqdn and fqdn", pod, 0, output)
	})

})
