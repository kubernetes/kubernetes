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

/* This test check that setHostnameAsFQDN PodSpec field works as
 * expected.
 */

package e2enode

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

var _ = SIGDescribe("Hostname of Pod [Feature:SetHostnameAsFQDN][NodeAlphaFeature:SetHostnameAsFQDN]", func() {
	f := framework.NewDefaultFramework("hostfqdn")

	/*
	   Release: v1.19
	   Testname: Create Pod without fully qualified domain name (FQDN)
	   Description: A Pod that does not define the subdomain field in it spec, does not have FQDN.
	*/
	ginkgo.It("a pod without subdomain field does not have FQDN", func() {
		pod := testPod()
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)';'"}
		output := []string{fmt.Sprintf("%s;%s;", pod.ObjectMeta.Name, pod.ObjectMeta.Name)}
		// Create Pod
		f.TestContainerOutput("shotname only", pod, 0, output)
	})

	/*
	   Release: v1.19
	   Testname: Create Pod without FQDN, setHostnameAsFQDN field set to true
	   Description: A Pod that does not define the subdomain field in it spec, does not have FQDN.
	                Hence, SetHostnameAsFQDN feature has no effect.
	*/
	ginkgo.It("a pod without FQDN is not affected by SetHostnameAsFQDN field", func() {
		pod := testPod()
		// Setting setHostnameAsFQDN field to true should have no effect.
		setHostnameAsFQDN := true
		pod.Spec.SetHostnameAsFQDN = &setHostnameAsFQDN
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)';'"}
		output := []string{fmt.Sprintf("%s;%s;", pod.ObjectMeta.Name, pod.ObjectMeta.Name)}
		// Create Pod
		f.TestContainerOutput("shotname only", pod, 0, output)
	})

	/*
	   Release: v1.19
	   Testname: Create Pod with FQDN, setHostnameAsFQDN field not defined.
	   Description: A Pod that defines the subdomain field in it spec has FQDN.
	                hostname command returns shortname (pod name in this case), and hostname -f returns FQDN.
	*/
	ginkgo.It("a pod with subdomain field has FQDN, hostname is shortname", func() {
		pod := testPod()
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)';'"}
		subdomain := "t"
		// Set PodSpec subdomain field to generate FQDN for pod
		pod.Spec.Subdomain = subdomain
		// Expected Pod FQDN
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", pod.ObjectMeta.Name, subdomain, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		output := []string{fmt.Sprintf("%s;%s;", pod.ObjectMeta.Name, hostFQDN)}
		// Create Pod
		f.TestContainerOutput("shotname and fqdn", pod, 0, output)
	})

	/*
	   Release: v1.19
	   Testname: Create Pod with FQDN, setHostnameAsFQDN field set to true.
	   Description: A Pod that defines the subdomain field in it spec has FQDN. When setHostnameAsFQDN: true, the
	                hostname is set to be the FQDN. In this case, both commands hostname and hostname -f return the FQDN of the Pod.
	*/
	ginkgo.It("a pod with subdomain field has FQDN, when setHostnameAsFQDN is set to true, the FQDN is set as hostname", func() {
		pod := testPod()
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)';'"}
		subdomain := "t"
		// Set PodSpec subdomain field to generate FQDN for pod
		pod.Spec.Subdomain = subdomain
		// Set PodSpec setHostnameAsFQDN to set FQDN as hostname
		setHostnameAsFQDN := true
		pod.Spec.SetHostnameAsFQDN = &setHostnameAsFQDN
		// Expected Pod FQDN
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", pod.ObjectMeta.Name, subdomain, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		// Fail if FQDN is longer than 64 characters, otherwise the Pod will remain pending until test timeout.
		// In Linux, 64 characters is the limit of the hostname kernel field, which this test sets to the pod FQDN.
		framework.ExpectEqual(len(hostFQDN) < 65, true, fmt.Sprintf("The FQDN of the Pod cannot be longer than 64 characters, requested %s which is %d characters long.", hostFQDN, len(hostFQDN)))
		output := []string{fmt.Sprintf("%s;%s;", hostFQDN, hostFQDN)}
		// Create Pod
		f.TestContainerOutput("fqdn and fqdn", pod, 0, output)
	})

})
