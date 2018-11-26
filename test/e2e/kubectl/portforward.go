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

// OWNER = sig/cli

package kubectl

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"net"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/net/websocket"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	podName = "pfpod"
)

const (
	podCheckInterval     = 1 * time.Second
	postStartWaitTimeout = 2 * time.Minute
)

// TODO support other ports besides 80
var (
	portForwardRegexp = regexp.MustCompile("Forwarding from (127.0.0.1|\\[::1\\]):([0-9]+) -> 80")
)

func pfPod(expectedClientData, chunks, chunkSize, chunkIntervalMillis string, bindAddress string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   podName,
			Labels: map[string]string{"name": podName},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "readiness",
					Image: imageutils.GetE2EImage(imageutils.Netexec),
					ReadinessProbe: &v1.Probe{
						Handler: v1.Handler{
							Exec: &v1.ExecAction{
								Command: []string{
									"sh", "-c", "netstat -na | grep LISTEN | grep -v 8080 | grep 80",
								}},
						},
						InitialDelaySeconds: 5,
						TimeoutSeconds:      60,
						PeriodSeconds:       1,
					},
				},
				{
					Name:  "portforwardtester",
					Image: imageutils.GetE2EImage(imageutils.PortForwardTester),
					Env: []v1.EnvVar{
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
						{
							Name:  "BIND_ADDRESS",
							Value: bindAddress,
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}

func WaitForTerminatedContainer(f *framework.Framework, pod *v1.Pod, containerName string) error {
	return framework.WaitForPodCondition(f.ClientSet, f.Namespace.Name, pod.Name, "container terminated", framework.PodStartTimeout, func(pod *v1.Pod) (bool, error) {
		if len(testutils.TerminatedContainers(pod)[containerName]) > 0 {
			return true, nil
		}
		return false, nil
	})
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

// runPortForward runs port-forward, warning, this may need root functionality on some systems.
func runPortForward(ns, podName string, port int) *portForwardCommand {
	cmd := framework.KubectlCmd("port-forward", fmt.Sprintf("--namespace=%v", ns), podName, fmt.Sprintf(":%d", port))
	// This is somewhat ugly but is the only way to retrieve the port that was picked
	// by the port-forward command. We don't want to hard code the port as we have no
	// way of guaranteeing we can pick one that isn't in use, particularly on Jenkins.
	framework.Logf("starting port-forward command and streaming output")
	portOutput, _, err := framework.StartCmdAndStreamOutput(cmd)
	if err != nil {
		framework.Failf("Failed to start port-forward command: %v", err)
	}

	buf := make([]byte, 128)

	var n int
	framework.Logf("reading from `kubectl port-forward` command's stdout")
	if n, err = portOutput.Read(buf); err != nil {
		framework.Failf("Failed to read from kubectl port-forward stdout: %v", err)
	}
	portForwardOutput := string(buf[:n])
	match := portForwardRegexp.FindStringSubmatch(portForwardOutput)
	if len(match) != 3 {
		framework.Failf("Failed to parse kubectl port-forward output: %s", portForwardOutput)
	}

	listenPort, err := strconv.Atoi(match[2])
	if err != nil {
		framework.Failf("Error converting %s to an int: %v", match[1], err)
	}

	return &portForwardCommand{
		cmd:  cmd,
		port: listenPort,
	}
}

func doTestConnectSendDisconnect(bindAddress string, f *framework.Framework) {
	By("Creating the target pod")
	pod := pfPod("", "10", "10", "100", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod); err != nil {
		framework.Failf("Couldn't create pod: %v", err)
	}
	if err := f.WaitForPodReady(pod.Name); err != nil {
		framework.Failf("Pod did not start running: %v", err)
	}

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
	if err := WaitForTerminatedContainer(f, pod, "portforwardtester"); err != nil {
		framework.Failf("Container did not terminate: %v", err)
	}

	By("Verifying logs")
	Eventually(func() (string, error) {
		return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	}, postStartWaitTimeout, podCheckInterval).Should(SatisfyAll(
		ContainSubstring("Accepted client connection"),
		ContainSubstring("Done"),
	))
}

func doTestMustConnectSendNothing(bindAddress string, f *framework.Framework) {
	By("Creating the target pod")
	pod := pfPod("abc", "1", "1", "1", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod); err != nil {
		framework.Failf("Couldn't create pod: %v", err)
	}
	if err := f.WaitForPodReady(pod.Name); err != nil {
		framework.Failf("Pod did not start running: %v", err)
	}

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
	if err := WaitForTerminatedContainer(f, pod, "portforwardtester"); err != nil {
		framework.Failf("Container did not terminate: %v", err)
	}

	By("Verifying logs")
	Eventually(func() (string, error) {
		return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	}, postStartWaitTimeout, podCheckInterval).Should(SatisfyAll(
		ContainSubstring("Accepted client connection"),
		ContainSubstring("Expected to read 3 bytes from client, but got 0 instead"),
	))
}

func doTestMustConnectSendDisconnect(bindAddress string, f *framework.Framework) {
	By("Creating the target pod")
	pod := pfPod("abc", "10", "10", "100", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod); err != nil {
		framework.Failf("Couldn't create pod: %v", err)
	}
	if err := f.WaitForPodReady(pod.Name); err != nil {
		framework.Failf("Pod did not start running: %v", err)
	}

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
	if err := WaitForTerminatedContainer(f, pod, "portforwardtester"); err != nil {
		framework.Failf("Container did not terminate: %v", err)
	}

	By("Verifying logs")
	Eventually(func() (string, error) {
		return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	}, postStartWaitTimeout, podCheckInterval).Should(SatisfyAll(
		ContainSubstring("Accepted client connection"),
		ContainSubstring("Received expected client data"),
		ContainSubstring("Done"),
	))
}

func doTestOverWebSockets(bindAddress string, f *framework.Framework) {
	config, err := framework.LoadConfig()
	Expect(err).NotTo(HaveOccurred(), "unable to get base config")

	By("Creating the pod")
	pod := pfPod("def", "10", "10", "100", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod); err != nil {
		framework.Failf("Couldn't create pod: %v", err)
	}
	if err := f.WaitForPodReady(pod.Name); err != nil {
		framework.Failf("Pod did not start running: %v", err)
	}

	req := f.ClientSet.CoreV1().RESTClient().Get().
		Namespace(f.Namespace.Name).
		Resource("pods").
		Name(pod.Name).
		Suffix("portforward").
		Param("ports", "80")

	url := req.URL()
	ws, err := framework.OpenWebSocketForURL(url, config, []string{"v4.channel.k8s.io"})
	if err != nil {
		framework.Failf("Failed to open websocket to %s: %v", url.String(), err)
	}
	defer ws.Close()

	Eventually(func() error {
		channel, msg, err := wsRead(ws)
		if err != nil {
			return fmt.Errorf("Failed to read completely from websocket %s: %v", url.String(), err)
		}
		if channel != 0 {
			return fmt.Errorf("Got message from server that didn't start with channel 0 (data): %v", msg)
		}
		if p := binary.LittleEndian.Uint16(msg); p != 80 {
			return fmt.Errorf("Received the wrong port: %d", p)
		}
		return nil
	}, time.Minute, 10*time.Second).Should(BeNil())

	Eventually(func() error {
		channel, msg, err := wsRead(ws)
		if err != nil {
			return fmt.Errorf("Failed to read completely from websocket %s: %v", url.String(), err)
		}
		if channel != 1 {
			return fmt.Errorf("Got message from server that didn't start with channel 1 (error): %v", msg)
		}
		if p := binary.LittleEndian.Uint16(msg); p != 80 {
			return fmt.Errorf("Received the wrong port: %d", p)
		}
		return nil
	}, time.Minute, 10*time.Second).Should(BeNil())

	By("Sending the expected data to the local port")
	err = wsWrite(ws, 0, []byte("def"))
	if err != nil {
		framework.Failf("Failed to write to websocket %s: %v", url.String(), err)
	}

	By("Reading data from the local port")
	buf := bytes.Buffer{}
	expectedData := bytes.Repeat([]byte("x"), 100)
	Eventually(func() error {
		channel, msg, err := wsRead(ws)
		if err != nil {
			return fmt.Errorf("Failed to read completely from websocket %s: %v", url.String(), err)
		}
		if channel != 0 {
			return fmt.Errorf("Got message from server that didn't start with channel 0 (data): %v", msg)
		}
		buf.Write(msg)
		if bytes.Equal(expectedData, buf.Bytes()) {
			return fmt.Errorf("Expected %q from server, got %q", expectedData, buf.Bytes())
		}
		return nil
	}, time.Minute, 10*time.Second).Should(BeNil())

	By("Verifying logs")
	Eventually(func() (string, error) {
		return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	}, postStartWaitTimeout, podCheckInterval).Should(SatisfyAll(
		ContainSubstring("Accepted client connection"),
		ContainSubstring("Received expected client data"),
	))
}

var _ = SIGDescribe("Kubectl Port forwarding", func() {
	f := framework.NewDefaultFramework("port-forwarding")

	framework.KubeDescribe("With a server listening on 0.0.0.0", func() {
		framework.KubeDescribe("that expects a client request", func() {
			It("should support a client that connects, sends NO DATA, and disconnects", func() {
				doTestMustConnectSendNothing("0.0.0.0", f)
			})
			It("should support a client that connects, sends DATA, and disconnects", func() {
				doTestMustConnectSendDisconnect("0.0.0.0", f)
			})
		})

		framework.KubeDescribe("that expects NO client request", func() {
			It("should support a client that connects, sends DATA, and disconnects", func() {
				doTestConnectSendDisconnect("0.0.0.0", f)
			})
		})

		It("should support forwarding over websockets", func() {
			doTestOverWebSockets("0.0.0.0", f)
		})
	})

	// kubectl port-forward may need elevated privileges to do its job.
	framework.KubeDescribe("With a server listening on localhost", func() {
		framework.KubeDescribe("that expects a client request", func() {
			It("should support a client that connects, sends NO DATA, and disconnects", func() {
				doTestMustConnectSendNothing("localhost", f)
			})
			It("should support a client that connects, sends DATA, and disconnects", func() {
				doTestMustConnectSendDisconnect("localhost", f)
			})
		})

		framework.KubeDescribe("that expects NO client request", func() {
			It("should support a client that connects, sends DATA, and disconnects", func() {
				doTestConnectSendDisconnect("localhost", f)
			})
		})

		It("should support forwarding over websockets", func() {
			doTestOverWebSockets("localhost", f)
		})
	})
})

func wsRead(conn *websocket.Conn) (byte, []byte, error) {
	for {
		var data []byte
		err := websocket.Message.Receive(conn, &data)
		if err != nil {
			return 0, nil, err
		}

		if len(data) == 0 {
			continue
		}

		channel := data[0]
		data = data[1:]

		return channel, data, err
	}
}

func wsWrite(conn *websocket.Conn, channel byte, data []byte) error {
	frame := make([]byte, len(data)+1)
	frame[0] = channel
	copy(frame[1:], data)
	err := websocket.Message.Send(conn, frame)
	return err
}
