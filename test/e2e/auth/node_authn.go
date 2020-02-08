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

package auth

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
)

var _ = SIGDescribe("[Feature:NodeAuthenticator]", func() {

	f := framework.NewDefaultFramework("node-authn")
	var ns string
	var nodeIPs []string
	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name

		nodeList, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to list nodes in namespace: %s", ns)
		framework.ExpectNotEqual(len(nodeList.Items), 0)

		pickedNode := nodeList.Items[0]
		nodeIPs = e2enode.GetAddresses(&pickedNode, v1.NodeExternalIP)
		// The pods running in the cluster can see the internal addresses.
		nodeIPs = append(nodeIPs, e2enode.GetAddresses(&pickedNode, v1.NodeInternalIP)...)

		// make sure ServiceAccount admission controller is enabled, so secret generation on SA creation works
		saName := "default"
		sa, err := f.ClientSet.CoreV1().ServiceAccounts(ns).Get(context.TODO(), saName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to retrieve service account (%s:%s)", ns, saName)
		framework.ExpectNotEqual(len(sa.Secrets), 0)

	})

	ginkgo.It("The kubelet's main port 10250 should reject requests with no credentials", func() {
		pod := createNodeAuthTestPod(f)
		for _, nodeIP := range nodeIPs {
			// Anonymous authentication is disabled by default
			result := framework.RunHostCmdOrDie(ns, pod.Name, fmt.Sprintf("curl -sIk -o /dev/null -w '%s' https://%s:%v/metrics", "%{http_code}", nodeIP, ports.KubeletPort))
			gomega.Expect(result).To(gomega.Or(gomega.Equal("401"), gomega.Equal("403")), "the kubelet's main port 10250 should reject requests with no credentials")
		}
	})

	ginkgo.It("The kubelet can delegate ServiceAccount tokens to the API server", func() {
		ginkgo.By("create a new ServiceAccount for authentication")
		trueValue := true
		newSA := &v1.ServiceAccount{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: ns,
				Name:      "node-auth-newsa",
			},
			AutomountServiceAccountToken: &trueValue,
		}
		_, err := f.ClientSet.CoreV1().ServiceAccounts(ns).Create(context.TODO(), newSA, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create service account (%s:%s)", ns, newSA.Name)

		pod := createNodeAuthTestPod(f)

		for _, nodeIP := range nodeIPs {
			result := framework.RunHostCmdOrDie(ns,
				pod.Name,
				fmt.Sprintf("curl -sIk -o /dev/null -w '%s' --header \"Authorization: Bearer `%s`\" https://%s:%v/metrics",
					"%{http_code}",
					"cat /var/run/secrets/kubernetes.io/serviceaccount/token",
					nodeIP, ports.KubeletPort))
			gomega.Expect(result).To(gomega.Or(gomega.Equal("401"), gomega.Equal("403")), "the kubelet can delegate ServiceAccount tokens to the API server")
		}
	})
})

func createNodeAuthTestPod(f *framework.Framework) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-node-authn-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:    "test-node-authn",
				Image:   imageutils.GetE2EImage(imageutils.Agnhost),
				Command: []string{"sleep", "3600"},
			}},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	return f.PodClient().CreateSync(pod)
}
