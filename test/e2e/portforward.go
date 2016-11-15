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
	"io/ioutil"
	"net"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	podName = "pfpod"
)

// TODO support other ports besides 80
var (
	portForwardRegexp        = regexp.MustCompile("Forwarding from 127.0.0.1:([0-9]+) -> 80")
	portForwardPortToStdOutV = version.MustParse("v1.3.0-alpha.4")
)

func pfPod(expectedClientData, chunks, chunkSize, chunkIntervalMillis string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   podName,
			Labels: map[string]string{"name": podName},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "portforwardtester",
					Image: "gcr.io/google_containers/portforwardtester:1.0",
					Env: []api.EnvVar{
						{
							Name:  "BIND_PORT",
							Value: "80",
						},
						{
							Name:  "EXPECTED_CLIENT_DATA",
							Value: expectedClientData,
						},
						{
							Name:  "CHUNKS",
							Value: chunks,
						},
						{
							Name:  "CHUNK_SIZE",
							Value: chunkSize,
						},
						{
							Name:  "CHUNK_INTERVAL",
							Value: chunkIntervalMillis,
						},
					},
				},
			},
			RestartPolicy: api.RestartPolicyNever,
		},
	}
}

type portForwardCommand struct {
	cmd  *exec.Cmd
	port int
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

func runPortForward(ns, podName string, port int) *portForwardCommand {
	cmd := framework.KubectlCmd("port-forward", fmt.Sprintf("--namespace=%v", ns), podName, fmt.Sprintf(":%d", port))
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

	return &portForwardCommand{
		cmd:  cmd,
		port: listenPort,
	}
}

var _ = framework.KubeDescribe("Port forwarding", func() {
	f := framework.NewDefaultFramework("port-forwarding")

	framework.KubeDescribe("With a server that expects a client request", func() {
		It("should support a client that connects, sends no data, and disconnects [Conformance]", func() {
			By("creating the target pod")
			pod := pfPod("abc", "1", "1", "1")
			if _, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod); err != nil {
				framework.Failf("Couldn't create pod: %v", err)
			}
			if err := f.WaitForPodRunning(pod.Name); err != nil {
				framework.Failf("Pod did not start running: %v", err)
			}
			defer func() {
				logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
				if err != nil {
					framework.Logf("Error getting pod log: %v", err)
				} else {
					framework.Logf("Pod log:\n%s", logs)
				}
			}()

			By("Running 'kubectl port-forward'")
			cmd := runPortForward(f.Namespace.Name, pod.Name, 80)
			defer cmd.Stop()

			By("Dialing the local port")
			conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", cmd.port))
			if err != nil {
				framework.Failf("Couldn't connect to port %d: %v", cmd.port, err)
			}

			By("Closing the connection to the local port")
			conn.Close()

			By("Waiting for the target pod to stop running")
			if err := f.WaitForPodNoLongerRunning(pod.Name); err != nil {
				framework.Failf("Pod did not stop running: %v", err)
			}

			By("Verifying logs")
			logOutput, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
			if err != nil {
				framework.Failf("Error retrieving pod logs: %v", err)
			}
			verifyLogMessage(logOutput, "Accepted client connection")
			verifyLogMessage(logOutput, "Expected to read 3 bytes from client, but got 0 instead")
		})

		It("should support a client that connects, sends data, and disconnects [Conformance]", func() {
			By("creating the target pod")
			pod := pfPod("abc", "10", "10", "100")
			if _, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod); err != nil {
				framework.Failf("Couldn't create pod: %v", err)
			}
			if err := f.WaitForPodRunning(pod.Name); err != nil {
				framework.Failf("Pod did not start running: %v", err)
			}
			defer func() {
				logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
				if err != nil {
					framework.Logf("Error getting pod log: %v", err)
				} else {
					framework.Logf("Pod log:\n%s", logs)
				}
			}()

			By("Running 'kubectl port-forward'")
			cmd := runPortForward(f.Namespace.Name, pod.Name, 80)
			defer cmd.Stop()

			By("Dialing the local port")
			addr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("127.0.0.1:%d", cmd.port))
			if err != nil {
				framework.Failf("Error resolving tcp addr: %v", err)
			}
			conn, err := net.DialTCP("tcp", nil, addr)
			if err != nil {
				framework.Failf("Couldn't connect to port %d: %v", cmd.port, err)
			}
			defer func() {
				By("Closing the connection to the local port")
				conn.Close()
			}()

			By("Sending the expected data to the local port")
			fmt.Fprint(conn, "abc")

			By("Closing the write half of the client's connection")
			conn.CloseWrite()

			By("Reading data from the local port")
			fromServer, err := ioutil.ReadAll(conn)
			if err != nil {
				framework.Failf("Unexpected error reading data from the server: %v", err)
			}

			if e, a := strings.Repeat("x", 100), string(fromServer); e != a {
				framework.Failf("Expected %q from server, got %q", e, a)
			}

			By("Waiting for the target pod to stop running")
			if err := f.WaitForPodNoLongerRunning(pod.Name); err != nil {
				framework.Failf("Pod did not stop running: %v", err)
			}

			By("Verifying logs")
			logOutput, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
			if err != nil {
				framework.Failf("Error retrieving pod logs: %v", err)
			}
			verifyLogMessage(logOutput, "^Accepted client connection$")
			verifyLogMessage(logOutput, "^Received expected client data$")
			verifyLogMessage(logOutput, "^Done$")
		})
	})
	framework.KubeDescribe("With a server that expects no client request", func() {
		It("should support a client that connects, sends no data, and disconnects [Conformance]", func() {
			By("creating the target pod")
			pod := pfPod("", "10", "10", "100")
			if _, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod); err != nil {
				framework.Failf("Couldn't create pod: %v", err)
			}
			if err := f.WaitForPodRunning(pod.Name); err != nil {
				framework.Failf("Pod did not start running: %v", err)
			}
			defer func() {
				logs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
				if err != nil {
					framework.Logf("Error getting pod log: %v", err)
				} else {
					framework.Logf("Pod log:\n%s", logs)
				}
			}()

			By("Running 'kubectl port-forward'")
			cmd := runPortForward(f.Namespace.Name, pod.Name, 80)
			defer cmd.Stop()

			By("Dialing the local port")
			conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", cmd.port))
			if err != nil {
				framework.Failf("Couldn't connect to port %d: %v", cmd.port, err)
			}
			defer func() {
				By("Closing the connection to the local port")
				conn.Close()
			}()

			By("Reading data from the local port")
			fromServer, err := ioutil.ReadAll(conn)
			if err != nil {
				framework.Failf("Unexpected error reading data from the server: %v", err)
			}

			if e, a := strings.Repeat("x", 100), string(fromServer); e != a {
				framework.Failf("Expected %q from server, got %q", e, a)
			}

			By("Waiting for the target pod to stop running")
			if err := f.WaitForPodNoLongerRunning(pod.Name); err != nil {
				framework.Failf("Pod did not stop running: %v", err)
			}

			By("Verifying logs")
			logOutput, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
			if err != nil {
				framework.Failf("Error retrieving pod logs: %v", err)
			}
			verifyLogMessage(logOutput, "Accepted client connection")
			verifyLogMessage(logOutput, "Done")
		})
	})
})

func verifyLogMessage(log, expected string) {
	re := regexp.MustCompile(expected)
	lines := strings.Split(log, "\n")
	for i := range lines {
		if re.MatchString(lines[i]) {
			return
		}
	}
	framework.Failf("Missing %q from log: %s", expected, log)
}
