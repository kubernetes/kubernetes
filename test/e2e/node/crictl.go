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

package node

import (
	"context"
	"fmt"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("crictl", func() {
	f := framework.NewDefaultFramework("crictl")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		// `crictl` is not available on all cloud providers.
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
	})

	ginkgo.It("should be able to run crictl on the node", func(ctx context.Context) {
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, f.ClientSet, maxNodes)
		framework.ExpectNoError(err)

		testCases := []string{
			"crictl version",
			"crictl info",
		}

		hostExec := utils.NewHostExec(f)

		for _, testCase := range testCases {
			for _, node := range nodes.Items {
				ginkgo.By(fmt.Sprintf("Testing %q on node %q ", testCase, node.GetName()))

				res, err := hostExec.Execute(ctx, testCase, &node)
				framework.ExpectNoError(err)

				if res.Stdout == "" && res.Stderr == "" {
					framework.Fail("output is empty")
				}
			}
		}
	})
})
