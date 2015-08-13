/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"strings"

	client "k8s.io/kubernetes/pkg/client/unversioned"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("SSH", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())

		// When adding more providers here, also implement their functionality in util.go's getSigner(...).
		SkipUnlessProviderIs("gce", "gke")
	})

	It("should SSH to all nodes and run commands", func() {
		// Get all nodes' external IPs.
		By("Getting all nodes' SSH-able IP addresses")
		hosts, err := NodeSSHHosts(c)
		if err != nil {
			Failf("Error getting node hostnames: %v", err)
		}

		testCases := []struct {
			cmd            string
			checkStdout    bool
			expectedStdout string
			expectedStderr string
			expectedCode   int
			expectedError  error
		}{
			{`echo "Hello"`, true, "Hello", "", 0, nil},
			// Same as previous, but useful for test output diagnostics.
			{`echo "Hello from $(whoami)@$(hostname)"`, false, "", "", 0, nil},
			{`echo "foo" | grep "bar"`, true, "", "", 1, nil},
			{`echo "Out" && echo "Error" >&2 && exit 7`, true, "Out", "Error", 7, nil},
		}

		// Run commands on all nodes via SSH.
		for _, testCase := range testCases {
			By(fmt.Sprintf("SSH'ing to all nodes and running %s", testCase.cmd))
			for _, host := range hosts {
				stdout, stderr, code, err := SSH(testCase.cmd, host, testContext.Provider)
				stdout, stderr = strings.TrimSpace(stdout), strings.TrimSpace(stderr)
				if err != testCase.expectedError {
					Failf("Ran %s on %s, got error %v, expected %v", testCase.cmd, host, err, testCase.expectedError)
				}
				if testCase.checkStdout && stdout != testCase.expectedStdout {
					Failf("Ran %s on %s, got stdout '%s', expected '%s'", testCase.cmd, host, stdout, testCase.expectedStdout)
				}
				if stderr != testCase.expectedStderr {
					Failf("Ran %s on %s, got stderr '%s', expected '%s'", testCase.cmd, host, stderr, testCase.expectedStderr)
				}
				if code != testCase.expectedCode {
					Failf("Ran %s on %s, got exit code %d, expected %d", testCase.cmd, host, code, testCase.expectedCode)
				}
				// Show stdout, stderr for logging purposes.
				if len(stdout) > 0 {
					Logf("Got stdout from %s: %s", host, strings.TrimSpace(stdout))
				}
				if len(stderr) > 0 {
					Logf("Got stderr from %s: %s", host, strings.TrimSpace(stderr))
				}
			}
		}

		// Quickly test that SSH itself errors correctly.
		By("SSH'ing to a nonexistent host")
		if _, _, _, err = SSH(`echo "hello"`, "i.do.not.exist", testContext.Provider); err == nil {
			Failf("Expected error trying to SSH to nonexistent host.")
		}
	})
})
