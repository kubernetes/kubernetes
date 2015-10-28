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
	"io/ioutil"
	"net"
	"os/exec"
	"regexp"
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/api"

	. "github.com/onsi/ginkgo"
)

const (
	podName = "pfpod"
)

// TODO support other ports besides 80
var portForwardRegexp = regexp.MustCompile("Forwarding from 127.0.0.1:([0-9]+) -> 80")

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

func runPortForward(ns, podName string, port int) (*exec.Cmd, int) {
	cmd := kubectlCmd("port-forward", fmt.Sprintf("--namespace=%v", ns), podName, fmt.Sprintf(":%d", port))
	// This is somewhat ugly but is the only way to retrieve the port that was picked
	// by the port-forward command. We don't want to hard code the port as we have no
	// way of guaranteeing we can pick one that isn't in use, particularly on Jenkins.
	Logf("starting port-forward command and streaming output")
	stdout, stderr, err := startCmdAndStreamOutput(cmd)
	if err != nil {
		Failf("Failed to start port-forward command: %v", err)
	}
	defer stdout.Close()
	defer stderr.Close()

	buf := make([]byte, 128)
	var n int
	Logf("reading from `kubectl port-forward` command's stderr")
	if n, err = stderr.Read(buf); err != nil {
		Failf("Failed to read from kubectl port-forward stderr: %v", err)
	}
	portForwardOutput := string(buf[:n])
	match := portForwardRegexp.FindStringSubmatch(portForwardOutput)
	if len(match) != 2 {
		Failf("Failed to parse kubectl port-forward output: %s", portForwardOutput)
	}

	listenPort, err := strconv.Atoi(match[1])
	if err != nil {
		Failf("Error converting %s to an int: %v", match[1], err)
	}

	return cmd, listenPort
}

var _ = Describe("Port forwarding", func() {
	framework := NewFramework("port-forwarding")

	Describe("With a server that expects a client request", func() {
		It("should support a client that connects, sends no data, and disconnects", func() {
			By("creating the target pod")
			pod := pfPod("abc", "1", "1", "1")
			framework.Client.Pods(framework.Namespace.Name).Create(pod)
			framework.WaitForPodRunning(pod.Name)

			By("Running 'kubectl port-forward'")
			cmd, listenPort := runPortForward(framework.Namespace.Name, pod.Name, 80)
			defer tryKill(cmd)

			By("Dialing the local port")
			conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", listenPort))
			if err != nil {
				Failf("Couldn't connect to port %d: %v", listenPort, err)
			}

			By("Closing the connection to the local port")
			conn.Close()

			logOutput := runKubectl("logs", fmt.Sprintf("--namespace=%v", framework.Namespace.Name), "-f", podName)
			verifyLogMessage(logOutput, "Accepted client connection")
			verifyLogMessage(logOutput, "Expected to read 3 bytes from client, but got 0 instead")
		})

		It("should support a client that connects, sends data, and disconnects", func() {
			By("creating the target pod")
			pod := pfPod("abc", "10", "10", "100")
			framework.Client.Pods(framework.Namespace.Name).Create(pod)
			framework.WaitForPodRunning(pod.Name)

			By("Running 'kubectl port-forward'")
			cmd, listenPort := runPortForward(framework.Namespace.Name, pod.Name, 80)
			defer tryKill(cmd)

			By("Dialing the local port")
			addr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("127.0.0.1:%d", listenPort))
			if err != nil {
				Failf("Error resolving tcp addr: %v", err)
			}
			conn, err := net.DialTCP("tcp", nil, addr)
			if err != nil {
				Failf("Couldn't connect to port %d: %v", listenPort, err)
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
				Failf("Unexpected error reading data from the server: %v", err)
			}

			if e, a := strings.Repeat("x", 100), string(fromServer); e != a {
				Failf("Expected %q from server, got %q", e, a)
			}

			logOutput := runKubectl("logs", fmt.Sprintf("--namespace=%v", framework.Namespace.Name), "-f", podName)
			verifyLogMessage(logOutput, "^Accepted client connection$")
			verifyLogMessage(logOutput, "^Received expected client data$")
			verifyLogMessage(logOutput, "^Done$")
		})
	})
	Describe("With a server that expects no client request", func() {
		It("should support a client that connects, sends no data, and disconnects", func() {
			By("creating the target pod")
			pod := pfPod("", "10", "10", "100")
			framework.Client.Pods(framework.Namespace.Name).Create(pod)
			framework.WaitForPodRunning(pod.Name)

			By("Running 'kubectl port-forward'")
			cmd, listenPort := runPortForward(framework.Namespace.Name, pod.Name, 80)
			defer tryKill(cmd)

			By("Dialing the local port")
			conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", listenPort))
			if err != nil {
				Failf("Couldn't connect to port %d: %v", listenPort, err)
			}
			defer func() {
				By("Closing the connection to the local port")
				conn.Close()
			}()

			By("Reading data from the local port")
			fromServer, err := ioutil.ReadAll(conn)
			if err != nil {
				Failf("Unexpected error reading data from the server: %v", err)
			}

			if e, a := strings.Repeat("x", 100), string(fromServer); e != a {
				Failf("Expected %q from server, got %q", e, a)
			}

			logOutput := runKubectl("logs", fmt.Sprintf("--namespace=%v", framework.Namespace.Name), "-f", podName)
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
	Failf("Missing %q from log: %s", expected, log)
}
