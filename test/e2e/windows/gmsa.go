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
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("[Feature:Windows] [Feature:WindowsGMSA] GMSA [Slow]", func() {
	f := framework.NewDefaultFramework("gmsa-test-windows")

	ginkgo.Describe("kubelet GMSA support", func() {
		ginkgo.Context("when creating a pod with correct GMSA credential specs", func() {
			ginkgo.It("passes the credential specs down to the Pod's containers", func() {
				defer ginkgo.GinkgoRecover()

				podName := "with-correct-gmsa-annotations"

				container1Name := "container1"
				podDomain := "acme.com"

				container2Name := "container2"
				container2Domain := "contoso.org"

				containers := make([]corev1.Container, 2)
				for i, name := range []string{container1Name, container2Name} {
					containers[i] = corev1.Container{
						Name:  name,
						Image: imageutils.GetPauseImageName(),
					}
				}

				pod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: podName,
						Annotations: map[string]string{
							"pod.alpha.windows.kubernetes.io/gmsa-credential-spec":                         generateDummyCredSpecs(podDomain),
							container2Name + ".container.alpha.windows.kubernetes.io/gmsa-credential-spec": generateDummyCredSpecs(container2Domain),
						},
					},
					Spec: corev1.PodSpec{
						Containers: containers,
					},
				}

				ginkgo.By("creating a pod with correct GMSA annotations")
				f.PodClient().CreateSync(pod)

				ginkgo.By("checking the domain reported by nltest in the containers")
				namespaceOption := fmt.Sprintf("--namespace=%s", f.Namespace.Name)
				for containerName, domain := range map[string]string{
					container1Name: podDomain,
					container2Name: container2Domain,
				} {
					var (
						output string
						err    error
					)

					containerOption := fmt.Sprintf("--container=%s", containerName)
					// even for bogus creds, `nltest /PARENTDOMAIN` simply returns the AD domain, which is enough for our purpose here.
					// note that the "eventually" part seems to be needed to account for the fact that powershell containers
					// are a bit slow to become responsive, even when docker reports them as running.
					gomega.Eventually(func() bool {
						output, err = framework.RunKubectl("exec", namespaceOption, podName, containerOption, "--", "nltest", "/PARENTDOMAIN")
						return err == nil
					}, 1*time.Minute, 1*time.Second).Should(gomega.BeTrue())

					if !strings.HasPrefix(output, domain) {
						framework.Failf("Expected %q to start with %q", output, domain)
					}

					expectedSubstr := "The command completed successfully"
					if !strings.Contains(output, expectedSubstr) {
						framework.Failf("Expected %q to contain %q", output, expectedSubstr)
					}
				}

				// If this was an e2e_node test, we could also check that the registry keys used to pass down the cred specs to Docker
				// have been properly cleaned up - but as of right now, e2e_node tests don't support Windows. We should migrate this
				// test to an e2e_node test when they start supporting Windows.
			})
		})
	})
})

func generateDummyCredSpecs(domain string) string {
	shortName := strings.ToUpper(strings.Split(domain, ".")[0])

	return fmt.Sprintf(`{
       "ActiveDirectoryConfig":{
          "GroupManagedServiceAccounts":[
             {
                "Name":"WebApplication",
                "Scope":"%s"
             },
             {
                "Name":"WebApplication",
                "Scope":"%s"
             }
          ]
       },
       "CmsPlugins":[
          "ActiveDirectory"
       ],
       "DomainJoinConfig":{
          "DnsName":"%s",
          "DnsTreeName":"%s",
          "Guid":"244818ae-87ca-4fcd-92ec-e79e5252348a",
          "MachineAccountName":"WebApplication",
          "NetBiosName":"%s",
          "Sid":"S-1-5-21-2126729477-2524175714-3194792973"
       }
    }`, shortName, domain, domain, domain, shortName)
}
