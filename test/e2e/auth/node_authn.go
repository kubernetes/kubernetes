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
	"net"
	"strconv"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

var _ = SIGDescribe("[Feature:NodeAuthenticator]", func() {

	f := framework.NewDefaultFramework("node-authn")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline
	var ns string
	var nodeIPs []string
	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(f.ClientSet, 1)
		framework.ExpectNoError(err)

		family := v1.IPv4Protocol
		if framework.TestContext.ClusterIsIPv6() {
			family = v1.IPv6Protocol
		}

		nodeIPs := e2enode.GetAddressesByTypeAndFamily(&nodes.Items[0], v1.NodeInternalIP, family)
		framework.ExpectNotEqual(len(nodeIPs), 0)
	})

	ginkgo.It("The kubelet's main port 10250 should reject requests with no credentials", func() {
		pod := createNodeAuthTestPod(f)
		for _, nodeIP := range nodeIPs {
			// Anonymous authentication is disabled by default
			host := net.JoinHostPort(nodeIP, strconv.Itoa(ports.KubeletPort))
			result := framework.RunHostCmdOrDie(ns, pod.Name, fmt.Sprintf("curl -sIk -o /dev/null -w '%s' https://%s/metrics", "%{http_code}", host))
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
			host := net.JoinHostPort(nodeIP, strconv.Itoa(ports.KubeletPort))
			result := framework.RunHostCmdOrDie(ns,
				pod.Name,
				fmt.Sprintf("curl -sIk -o /dev/null -w '%s' --header \"Authorization: Bearer `%s`\" https://%s/metrics",
					"%{http_code}",
					"cat /var/run/secrets/kubernetes.io/serviceaccount/token",
					host))
			gomega.Expect(result).To(gomega.Or(gomega.Equal("401"), gomega.Equal("403")), "the kubelet can delegate ServiceAccount tokens to the API server")
		}
	})
})

func createNodeAuthTestPod(f *framework.Framework) *v1.Pod {
	pod := e2epod.NewAgnhostPod(f.Namespace.Name, "agnhost-pod", nil, nil, nil)
	pod.ObjectMeta.GenerateName = "test-node-authn-"
	return f.PodClient().CreateSync(pod)
}
