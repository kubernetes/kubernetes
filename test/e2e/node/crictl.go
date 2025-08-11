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
	"strings"
	"time"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("crictl", func() {
	f := framework.NewDefaultFramework("crictl")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		// `crictl` is not available on all cloud providers.
		e2eskipper.SkipUnlessProviderIs("gce")
	})

	ginkgo.It("should be able to run crictl on the node", func(ctx context.Context) {
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, f.ClientSet, maxNodes)
		framework.ExpectNoError(err)

		hostExec := utils.NewHostExec(f)

		for _, node := range nodes.Items {
			ginkgo.By(fmt.Sprintf("Testing `crictl version` on node %q ", node.GetName()))

			gomega.Eventually(ctx, func() error {
				// crictl is installed to /home/kubernetes/bin through configure.sh.
				res, err := hostExec.Execute(ctx, "/home/kubernetes/bin/crictl version", &node)
				if err != nil {
					return err
				}
				// crictl version example output:
				//
				// Version:  0.1.0
				// RuntimeName:  containerd
				// RuntimeVersion:  1.7.27
				// RuntimeApiVersion:  v1
				expectedOutput := "RuntimeVersion"
				if res.Code != 0 {
					return fmt.Errorf("exit code is not 0, exitCode=%d stdout=%q stderr=%q, might retry", res.Code, res.Stdout, res.Stderr)
				}
				if !strings.Contains(res.Stdout, expectedOutput) && !strings.Contains(res.Stderr, expectedOutput) {
					return fmt.Errorf("output contains stdout=%q stderr=%q expected to contain %q, might retry", res.Stdout, res.Stderr, expectedOutput)
				}
				return nil
			}, time.Minute, 5*time.Second).Should(gomega.Succeed())
		}
	})
})
