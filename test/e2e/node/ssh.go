/*
Copyright 2015 The Kubernetes Authors.

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
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"

	"github.com/onsi/ginkgo"
)

const maxNodes = 100

var _ = SIGDescribe("SSH", func() {

	f := framework.NewDefaultFramework("ssh")

	ginkgo.BeforeEach(func() {
		// When adding more providers here, also implement their functionality in e2essh.GetSigner(...).
		framework.SkipUnlessProviderIs(framework.ProvidersWithSSH...)

		// This test SSH's into the node for which it needs the $HOME/.ssh/id_rsa key to be present. So
		// we should skip if the environment does not have the key (not all CI systems support this use case)
		framework.SkipUnlessSSHKeyPresent()
	})

	ginkgo.It("should SSH to all nodes and run commands", func() {
		// Get all nodes' external IPs.
		ginkgo.By("Getting all nodes' SSH-able IP addresses")
		hosts, err := e2essh.NodeSSHHosts(f.ClientSet)
		if err != nil {
			e2elog.Failf("Error getting node hostnames: %v", err)
		}

		testCases := []struct {
			cmd            string
			checkStdout    bool
			expectedStdout string
			expectedStderr string
			expectedCode   int
			expectedError  error
		}{
			// Keep this test first - this variant runs on all nodes.
			{`echo "Hello from $(whoami)@$(hostname)"`, false, "", "", 0, nil},
			{`echo "foo" | grep "bar"`, true, "", "", 1, nil},
			{`echo "stdout" && echo "stderr" >&2 && exit 7`, true, "stdout", "stderr", 7, nil},
		}

		for i, testCase := range testCases {
			// Only run the first testcase against max 100 nodes. Run
			// the rest against the first node we find only, since
			// they're basically testing SSH semantics (and we don't
			// need to do that against each host in the cluster).
			nodes := len(hosts)
			if i > 0 {
				nodes = 1
			} else if nodes > maxNodes {
				nodes = maxNodes
			}
			testhosts := hosts[:nodes]
			ginkgo.By(fmt.Sprintf("SSH'ing to %d nodes and running %s", len(testhosts), testCase.cmd))

			for _, host := range testhosts {
				result, err := e2essh.SSH(testCase.cmd, host, framework.TestContext.Provider)
				stdout, stderr := strings.TrimSpace(result.Stdout), strings.TrimSpace(result.Stderr)
				if err != testCase.expectedError {
					e2elog.Failf("Ran %s on %s, got error %v, expected %v", testCase.cmd, host, err, testCase.expectedError)
				}
				if testCase.checkStdout && stdout != testCase.expectedStdout {
					e2elog.Failf("Ran %s on %s, got stdout '%s', expected '%s'", testCase.cmd, host, stdout, testCase.expectedStdout)
				}
				if stderr != testCase.expectedStderr {
					e2elog.Failf("Ran %s on %s, got stderr '%s', expected '%s'", testCase.cmd, host, stderr, testCase.expectedStderr)
				}
				if result.Code != testCase.expectedCode {
					e2elog.Failf("Ran %s on %s, got exit code %d, expected %d", testCase.cmd, host, result.Code, testCase.expectedCode)
				}
				// Show stdout, stderr for logging purposes.
				if len(stdout) > 0 {
					e2elog.Logf("Got stdout from %s: %s", host, strings.TrimSpace(stdout))
				}
				if len(stderr) > 0 {
					e2elog.Logf("Got stderr from %s: %s", host, strings.TrimSpace(stderr))
				}
			}
		}

		// Quickly test that SSH itself errors correctly.
		ginkgo.By("SSH'ing to a nonexistent host")
		if _, err = e2essh.SSH(`echo "hello"`, "i.do.not.exist", framework.TestContext.Provider); err == nil {
			e2elog.Failf("Expected error trying to SSH to nonexistent host.")
		}
	})
})
