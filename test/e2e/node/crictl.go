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
	"fmt"
	"strings"

	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("crictl", func() {
	f := framework.NewDefaultFramework("crictl")

	ginkgo.BeforeEach(func() {
		// `crictl` is not available on all cloud providers.
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
		// The test requires $HOME/.ssh/id_rsa key to be present.
		e2eskipper.SkipUnlessSSHKeyPresent()
	})

	ginkgo.It("should be able to run crictl on the node", func() {
		// Get all nodes' external IPs.
		ginkgo.By("Getting all nodes' SSH-able IP addresses")
		hosts, err := e2essh.NodeSSHHosts(f.ClientSet)
		if err != nil {
			framework.Failf("Error getting node hostnames: %v", err)
		}

		testCases := []struct {
			cmd string
		}{
			{`sudo crictl version`},
			{`sudo crictl info`},
		}

		for _, testCase := range testCases {
			// Choose an arbitrary node to test.
			host := hosts[0]
			ginkgo.By(fmt.Sprintf("SSH'ing to node %q to run %q", host, testCase.cmd))

			result, err := e2essh.SSH(testCase.cmd, host, framework.TestContext.Provider)
			stdout, stderr := strings.TrimSpace(result.Stdout), strings.TrimSpace(result.Stderr)
			if err != nil {
				framework.Failf("Ran %q on %q, got error %v", testCase.cmd, host, err)
			}
			// Log the stdout/stderr output.
			// TODO: Verify the output.
			if len(stdout) > 0 {
				framework.Logf("Got stdout from %q:\n %s\n", host, strings.TrimSpace(stdout))
			}
			if len(stderr) > 0 {
				framework.Logf("Got stderr from %q:\n %s\n", host, strings.TrimSpace(stderr))
			}
		}
	})
})
