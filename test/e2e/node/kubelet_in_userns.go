/*
Copyright The Kubernetes Authors.

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
	"context"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Kubelet running in user namespace", "[LinuxOnly]", feature.KubeletInUserNamespace, framework.WithFeatureGate(kubefeatures.KubeletInUserNamespace), func() {
	f := framework.NewDefaultFramework("kubelet-in-userns")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	f.Context("when checking node system info", func() {
		/*
			Release: v1.37
			Testname: Kubelet, running in user namespace
			Description: When the kubelet is running in a user namespace, the Node's status.nodeInfo.runningInUserNamespace MUST be true.
		*/
		framework.It("should report RunningInUserNamespace", func(ctx context.Context) {
			hostExec := utils.NewHostExec(f)
			ginkgo.DeferCleanup(hostExec.Cleanup)

			node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
			framework.ExpectNoError(err)
			gomega.Expect(node.Status.NodeInfo.RunningInUserNamespace).NotTo(gomega.BeNil(), "expected Node.Status.NodeInfo.RunningInUserNamespace to be set")

			nodeLabelRunningInUserNS := node.Labels[v1.LabelRunningInUserNamespace]
			gomega.Expect(nodeLabelRunningInUserNS).NotTo(gomega.BeEmpty(), "expected node label %s to be set", v1.LabelRunningInUserNamespace)

			kubeletPid := pidOfKubelet(ctx, hostExec, node)
			cmd := "readlink /proc/" + strconv.Itoa(kubeletPid) + "/ns/user"
			kubeletUserNS, err := hostExec.IssueCommandWithResult(ctx, cmd, node)
			framework.ExpectNoError(err, "Checking kubelet user namespace")
			kubeletUserNS = strings.TrimSpace(kubeletUserNS)
			gomega.Expect(kubeletUserNS).To(gomega.MatchRegexp(`^user:\[\d+\]$`), "expected user namespace link to match format")

			// The magic number is defined as USER_NS_INIT_INO in kernel.
			// https://github.com/torvalds/linux/blob/v6.17/include/uapi/linux/nsfs.h#L48
			const initUserNSMagic = "user:[4026531837]"
			kubeletRunningInUserNS := kubeletUserNS != initUserNSMagic
			framework.Logf("kubelet user namespace: %s (running in non-initial UserNS: %v)",
				kubeletUserNS, kubeletRunningInUserNS)

			if kubeletRunningInUserNS {
				gomega.Expect(*node.Status.NodeInfo.RunningInUserNamespace).To(gomega.BeTrueBecause("kubelet is running in user namespace"))
				gomega.Expect(nodeLabelRunningInUserNS).To(gomega.Equal("true"), "expected node label %s to be 'true'", v1.LabelRunningInUserNamespace)
			} else {
				gomega.Expect(*node.Status.NodeInfo.RunningInUserNamespace).To(gomega.BeFalseBecause("kubelet is not running in user namespace"))
				gomega.Expect(nodeLabelRunningInUserNS).To(gomega.Equal("false"), "expected node label %s to be 'false'", v1.LabelRunningInUserNamespace)
			}
		})
	})
})
