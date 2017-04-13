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

package e2e

import (
	"fmt"
	"io"
	"os/exec"
	"regexp"
	"strconv"
	"syscall"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	utilversion "k8s.io/kubernetes/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	podName = "pfpod"
)

var (
	portForwardRegexp        = regexp.MustCompile("Forwarding from 127.0.0.1:([0-9]+) -> 80")
	portForwardPortToStdOutV = utilversion.MustParseSemantic("v1.3.0-alpha.4")
)

func makePortForwardCommand(ns, podName string, remotePort int) *portForwardCommand {
	return &portForwardCommand{
		ns:         ns,
		podName:    podName,
		remotePort: remotePort,
	}
}

type portForwardCommand struct {
	ns         string
	podName    string
	cmd        *exec.Cmd
	remotePort int
	port       int
}

func (c *portForwardCommand) String() string {
	return "'kubectl port-forward'"
}

// Stop attempts to gracefully stop `kubectl port-forward`, only killing it if necessary.
// This helps avoid spdy goroutine leaks in the Kubelet.
func (c *portForwardCommand) Stop() {
	// SIGINT signals that kubectl port-forward should gracefully terminate
	if err := c.cmd.Process.Signal(syscall.SIGINT); err != nil {
		framework.Logf("error sending SIGINT to kubectl port-forward: %v", err)
	}

	// try to wait for a clean exit
	done := make(chan error)
	go func() {
		done <- c.cmd.Wait()
	}()

	expired := time.NewTimer(wait.ForeverTestTimeout)
	defer expired.Stop()

	select {
	case err := <-done:
		if err == nil {
			// success
			return
		}
		framework.Logf("error waiting for kubectl port-forward to exit: %v", err)
	case <-expired.C:
		framework.Logf("timed out waiting for kubectl port-forward to exit")
	}

	framework.Logf("trying to forcibly kill kubectl port-forward")
	framework.TryKill(c.cmd)
}

func (c *portForwardCommand) Run() int {
	cmd := framework.KubectlCmd("port-forward", fmt.Sprintf("--namespace=%v", c.ns), c.podName, fmt.Sprintf(":%d", c.remotePort))
	// This is somewhat ugly but is the only way to retrieve the port that was picked
	// by the port-forward command. We don't want to hard code the port as we have no
	// way of guaranteeing we can pick one that isn't in use, particularly on Jenkins.
	framework.Logf("starting port-forward command and streaming output")
	stdout, stderr, err := framework.StartCmdAndStreamOutput(cmd)
	if err != nil {
		framework.Failf("Failed to start port-forward command: %v", err)
	}

	buf := make([]byte, 128)

	// After v1.3.0-alpha.4 (#17030), kubectl port-forward outputs port
	// info to stdout, not stderr, so for version-skewed tests, look there
	// instead.
	var portOutput io.ReadCloser
	if useStdOut, err := framework.KubectlVersionGTE(portForwardPortToStdOutV); err != nil {
		framework.Failf("Failed to get kubectl version: %v", err)
	} else if useStdOut {
		portOutput = stdout
	} else {
		portOutput = stderr
	}

	var n int
	framework.Logf("reading from `kubectl port-forward` command's stdout")
	if n, err = portOutput.Read(buf); err != nil {
		framework.Failf("Failed to read from kubectl port-forward stdout: %v", err)
	}
	portForwardOutput := string(buf[:n])
	match := portForwardRegexp.FindStringSubmatch(portForwardOutput)
	if len(match) != 2 {
		framework.Failf("Failed to parse kubectl port-forward output: %s", portForwardOutput)
	}

	listenPort, err := strconv.Atoi(match[1])
	if err != nil {
		framework.Failf("Error converting %s to an int: %v", match[1], err)
	}
	c.cmd = cmd
	c.port = listenPort
	return listenPort
}

var _ = framework.KubeDescribe("Port forwarding", func() {
	f := framework.NewDefaultFramework("port-forwarding")
	var command framework.PortForwardCommand

	BeforeEach(func() {
		command = makePortForwardCommand(f.Namespace.Name, podName, 80)
	})

	framework.KubeDescribe("With a server listening on 0.0.0.0", func() {
		framework.KubeDescribe("that expects a client request", func() {
			It("should support a client that connects, sends no data, and disconnects", func() {
				framework.DoTestMustConnectSendNothing(command, "0.0.0.0", f)
			})
			It("should support a client that connects, sends data, and disconnects", func() {
				framework.DoTestMustConnectSendDisconnect(command, "0.0.0.0", f)
			})
		})

		framework.KubeDescribe("that expects no client request", func() {
			It("should support a client that connects, sends data, and disconnects", func() {
				framework.DoTestConnectSendDisconnect(command, "0.0.0.0", f)
			})
		})

		It("should support forwarding over websockets", func() {
			framework.DoTestOverWebSockets("0.0.0.0", f)
		})
	})

	framework.KubeDescribe("With a server listening on localhost", func() {
		framework.KubeDescribe("that expects a client request", func() {
			It("should support a client that connects, sends no data, and disconnects [Conformance]", func() {
				framework.DoTestMustConnectSendNothing(command, "localhost", f)
			})
			It("should support a client that connects, sends data, and disconnects [Conformance]", func() {
				framework.DoTestMustConnectSendDisconnect(command, "localhost", f)
			})
		})

		framework.KubeDescribe("that expects no client request", func() {
			It("should support a client that connects, sends data, and disconnects [Conformance]", func() {
				framework.DoTestConnectSendDisconnect(command, "localhost", f)
			})
		})

		It("should support forwarding over websockets", func() {
			framework.DoTestOverWebSockets("localhost", f)
		})
	})
})
