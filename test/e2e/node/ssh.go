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
        "context"
        "fmt"
        "strings"

        "k8s.io/kubernetes/test/e2e/framework"
        e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
        e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
        admissionapi "k8s.io/pod-security-admission/api"

        "github.com/onsi/ginkgo/v2"
)

const maxNodes = 100

var _ = SIGDescribe("SSH", func() {

        f := framework.NewDefaultFramework("ssh")
        f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

        ginkgo.BeforeEach(func() {
                // When adding more providers here, also implement their functionality in e2essh.GetSigner(...).
                e2eskipper.SkipUnlessProviderIs(framework.ProvidersWithSSH...)

                // This test SSH's into the node for which it needs the $HOME/.ssh/id_rsa key to be present.
                e2eskipper.SkipUnlessSSHKeyPresent()
        })

        ginkgo.It("should SSH to all nodes and run commands", func(ctx context.Context) {
                ginkgo.By("Getting all nodes' SSH-able IP addresses")
                hosts, err := e2essh.NodeSSHHosts(ctx, f.ClientSet)
                if err != nil {
                        framework.Failf("Error getting node hostnames: %v", err)
                }

                ginkgo.By(fmt.Sprintf("Found %d SSH'able hosts", len(hosts)))

                testCases := []struct {
                        cmd            string
                        checkStdout    bool
                        expectedStdout string
                        expectedStderr string
                        expectedCode   int
                        expectedError  error
                }{
                        {`echo "Hello from $(whoami)@$(hostname)"`, false, "", "", 0, nil},
                        {`echo "foo" | grep "bar"`, true, "", "", 1, nil},
                        {`echo "stdout" && echo "stderr" >&2 && exit 7`, true, "stdout", "stderr", 7, nil},
                }

                for i, testCase := range testCases {
                        nodes := len(hosts)
                        if i > 0 {
                                nodes = 1
                        } else if nodes > maxNodes {
                                nodes = maxNodes
                        }

                        testhosts := hosts[:nodes]
                        ginkgo.By(fmt.Sprintf("SSH'ing to %d nodes and running %s", len(testhosts), testCase.cmd))

                        for _, host := range testhosts {
                                result, err := e2essh.SSH(ctx, testCase.cmd, host, framework.TestContext.Provider)
                                stdout := strings.TrimSpace(result.Stdout)
                                stderr := strings.TrimSpace(result.Stderr)

                                if err != testCase.expectedError {
                                        framework.Failf("Ran %s on %s, got error %v, expected %v",
                                                testCase.cmd, host, err, testCase.expectedError)
                                }
                                if testCase.checkStdout && stdout != testCase.expectedStdout {
                                        framework.Failf("Ran %s on %s, got stdout '%s', expected '%s'",
                                                testCase.cmd, host, stdout, testCase.expectedStdout)
                                }
                                if stderr != testCase.expectedStderr {
                                        framework.Failf("Ran %s on %s, got stderr '%s', expected '%s'",
                                                testCase.cmd, host, stderr, testCase.expectedStderr)
                                }
                                if result.Code != testCase.expectedCode {
                                        framework.Failf("Ran %s on %s, got exit code %d, expected %d",
                                                testCase.cmd, host, result.Code, testCase.expectedCode)
                                }
                        }
                }
        })
})
