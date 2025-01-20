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
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"net/http"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/net/websocket"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2ewebsocket "k8s.io/kubernetes/test/e2e/framework/websocket"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
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
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"netexec"},
					ReadinessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
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
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"port-forward-tester"},
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

func pfNeverReadRequestBodyPod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "issue-74551",
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "server",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args: []string{
						"netexec",
						"--http-port=80",
					},
					ReadinessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							HTTPGet: &v1.HTTPGetAction{
								Path: "/healthz",
								Port: intstr.IntOrString{
									IntVal: int32(80),
								},
								Scheme: v1.URISchemeHTTP,
							},
						},
						InitialDelaySeconds: 5,
						TimeoutSeconds:      60,
						PeriodSeconds:       1,
					},
				},
			},
		},
	}
}

func testWebServerPod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   podName,
			Labels: map[string]string{"name": podName},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "testwebserver",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"test-webserver"},
					Ports: []v1.ContainerPort{{ContainerPort: int32(80)}},
					ReadinessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							HTTPGet: &v1.HTTPGetAction{
								Path: "/",
								Port: intstr.FromInt32(int32(80)),
							},
						},
						InitialDelaySeconds: 5,
						TimeoutSeconds:      3,
						FailureThreshold:    10,
					},
				},
			},
		},
	}
}

// WaitForTerminatedContainer waits till a given container be terminated for a given pod.
func WaitForTerminatedContainer(ctx context.Context, f *framework.Framework, pod *v1.Pod, containerName string) error {
	return e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "container terminated", framework.PodStartTimeout, func(pod *v1.Pod) (bool, error) {
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
	tk := e2ekubectl.NewTestKubeconfig(framework.TestContext.CertDir, framework.TestContext.Host, framework.TestContext.KubeConfig, framework.TestContext.KubeContext, framework.TestContext.KubectlPath, ns)
	cmd := tk.KubectlCmd("port-forward", fmt.Sprintf("--namespace=%v", ns), podName, fmt.Sprintf(":%d", port))
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
		framework.Failf("Error converting %s to an int: %v", match[2], err)
	}

	return &portForwardCommand{
		cmd:  cmd,
		port: listenPort,
	}
}

func doTestConnectSendDisconnect(ctx context.Context, bindAddress string, f *framework.Framework) {
	ginkgo.By("Creating the target pod")
	pod := pfPod("", "10", "10", "100", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		framework.Failf("Couldn't create pod: %v", err)
	}
	if err := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout); err != nil {
		framework.Failf("Pod did not start running: %v", err)
	}

	ginkgo.By("Running 'kubectl port-forward'")
	cmd := runPortForward(f.Namespace.Name, pod.Name, 80)
	defer cmd.Stop()

	ginkgo.By("Dialing the local port")
	conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", cmd.port))
	if err != nil {
		framework.Failf("Couldn't connect to port %d: %v", cmd.port, err)
	}
	defer func() {
		ginkgo.By("Closing the connection to the local port")
		conn.Close()
	}()

	ginkgo.By("Reading data from the local port")
	fromServer, err := io.ReadAll(conn)
	if err != nil {
		framework.Failf("Unexpected error reading data from the server: %v", err)
	}

	if e, a := strings.Repeat("x", 100), string(fromServer); e != a {
		framework.Failf("Expected %q from server, got %q", e, a)
	}

	ginkgo.By("Waiting for the target pod to stop running")
	if err := WaitForTerminatedContainer(ctx, f, pod, "portforwardtester"); err != nil {
		framework.Failf("Container did not terminate: %v", err)
	}

	ginkgo.By("Verifying logs")
	gomega.Eventually(ctx, func() (string, error) {
		return e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	}, postStartWaitTimeout, podCheckInterval).Should(gomega.SatisfyAll(
		gomega.ContainSubstring("Accepted client connection"),
		gomega.ContainSubstring("Done"),
	))
}

func doTestMustConnectSendNothing(ctx context.Context, bindAddress string, f *framework.Framework) {
	ginkgo.By("Creating the target pod")
	pod := pfPod("abc", "1", "1", "1", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		framework.Failf("Couldn't create pod: %v", err)
	}
	if err := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout); err != nil {
		framework.Failf("Pod did not start running: %v", err)
	}

	ginkgo.By("Running 'kubectl port-forward'")
	cmd := runPortForward(f.Namespace.Name, pod.Name, 80)
	defer cmd.Stop()

	ginkgo.By("Dialing the local port")
	conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", cmd.port))
	if err != nil {
		framework.Failf("Couldn't connect to port %d: %v", cmd.port, err)
	}

	ginkgo.By("Closing the connection to the local port")
	conn.Close()

	ginkgo.By("Waiting for the target pod to stop running")
	if err := WaitForTerminatedContainer(ctx, f, pod, "portforwardtester"); err != nil {
		framework.Failf("Container did not terminate: %v", err)
	}

	ginkgo.By("Verifying logs")
	gomega.Eventually(ctx, func() (string, error) {
		return e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	}, postStartWaitTimeout, podCheckInterval).Should(gomega.SatisfyAll(
		gomega.ContainSubstring("Accepted client connection"),
		gomega.ContainSubstring("Expected to read 3 bytes from client, but got 0 instead"),
	))
}

func doTestMustConnectSendDisconnect(ctx context.Context, bindAddress string, f *framework.Framework) {
	ginkgo.By("Creating the target pod")
	pod := pfPod("abc", "10", "10", "100", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		framework.Failf("Couldn't create pod: %v", err)
	}
	if err := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout); err != nil {
		framework.Failf("Pod did not start running: %v", err)
	}

	ginkgo.By("Running 'kubectl port-forward'")
	cmd := runPortForward(f.Namespace.Name, pod.Name, 80)
	defer cmd.Stop()

	ginkgo.By("Dialing the local port")
	addr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("127.0.0.1:%d", cmd.port))
	if err != nil {
		framework.Failf("Error resolving tcp addr: %v", err)
	}
	conn, err := net.DialTCP("tcp", nil, addr)
	if err != nil {
		framework.Failf("Couldn't connect to port %d: %v", cmd.port, err)
	}
	defer func() {
		ginkgo.By("Closing the connection to the local port")
		conn.Close()
	}()

	ginkgo.By("Sending the expected data to the local port")
	fmt.Fprint(conn, "abc")

	ginkgo.By("Reading data from the local port")
	fromServer, err := io.ReadAll(conn)
	if err != nil {
		framework.Failf("Unexpected error reading data from the server: %v", err)
	}

	if e, a := strings.Repeat("x", 100), string(fromServer); e != a {
		podlogs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
		if err != nil {
			framework.Logf("Failed to get logs of portforwardtester pod: %v", err)
		} else {
			framework.Logf("Logs of portforwardtester pod: %v", podlogs)
		}
		framework.Failf("Expected %q from server, got %q", e, a)
	}

	ginkgo.By("Closing the write half of the client's connection")
	if err = conn.CloseWrite(); err != nil {
		framework.Failf("Couldn't close the write half of the client's connection: %v", err)
	}

	ginkgo.By("Waiting for the target pod to stop running")
	if err := WaitForTerminatedContainer(ctx, f, pod, "portforwardtester"); err != nil {
		framework.Failf("Container did not terminate: %v", err)
	}

	ginkgo.By("Verifying logs")
	gomega.Eventually(ctx, func() (string, error) {
		return e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	}, postStartWaitTimeout, podCheckInterval).Should(gomega.SatisfyAll(
		gomega.ContainSubstring("Accepted client connection"),
		gomega.ContainSubstring("Received expected client data"),
		gomega.ContainSubstring("Done"),
	))
}

func doTestOverWebSockets(ctx context.Context, bindAddress string, f *framework.Framework) {
	config, err := framework.LoadConfig()
	framework.ExpectNoError(err, "unable to get base config")

	ginkgo.By("Creating the pod")
	pod := pfPod("def", "10", "10", "100", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		framework.Failf("Couldn't create pod: %v", err)
	}
	if err := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout); err != nil {
		framework.Failf("Pod did not start running: %v", err)
	}

	req := f.ClientSet.CoreV1().RESTClient().Get().
		Namespace(f.Namespace.Name).
		Resource("pods").
		Name(pod.Name).
		Suffix("portforward").
		Param("ports", "80")

	url := req.URL()
	ws, err := e2ewebsocket.OpenWebSocketForURL(url, config, []string{"v4.channel.k8s.io"})
	if err != nil {
		framework.Failf("Failed to open websocket to %s: %v", url.String(), err)
	}
	defer ws.Close()

	gomega.Eventually(ctx, func() error {
		channel, msg, err := wsRead(ws)
		if err != nil {
			return fmt.Errorf("failed to read completely from websocket %s: %w", url.String(), err)
		}
		if channel != 0 {
			return fmt.Errorf("got message from server that didn't start with channel 0 (data): %v", msg)
		}
		if p := binary.LittleEndian.Uint16(msg); p != 80 {
			return fmt.Errorf("received the wrong port: %d", p)
		}
		return nil
	}, time.Minute, 10*time.Second).Should(gomega.Succeed())

	gomega.Eventually(ctx, func() error {
		channel, msg, err := wsRead(ws)
		if err != nil {
			return fmt.Errorf("failed to read completely from websocket %s: %w", url.String(), err)
		}
		if channel != 1 {
			return fmt.Errorf("got message from server that didn't start with channel 1 (error): %v", msg)
		}
		if p := binary.LittleEndian.Uint16(msg); p != 80 {
			return fmt.Errorf("received the wrong port: %d", p)
		}
		return nil
	}, time.Minute, 10*time.Second).Should(gomega.Succeed())

	ginkgo.By("Sending the expected data to the local port")
	err = wsWrite(ws, 0, []byte("def"))
	if err != nil {
		framework.Failf("Failed to write to websocket %s: %v", url.String(), err)
	}

	ginkgo.By("Reading data from the local port")
	buf := bytes.Buffer{}
	expectedData := bytes.Repeat([]byte("x"), 100)
	gomega.Eventually(ctx, func() error {
		channel, msg, err := wsRead(ws)
		if err != nil {
			return fmt.Errorf("failed to read completely from websocket %s: %w", url.String(), err)
		}
		if channel != 0 {
			return fmt.Errorf("got message from server that didn't start with channel 0 (data): %v", msg)
		}
		buf.Write(msg)
		if bytes.Equal(expectedData, buf.Bytes()) {
			return fmt.Errorf("expected %q from server, got %q", expectedData, buf.Bytes())
		}
		return nil
	}, time.Minute, 10*time.Second).Should(gomega.Succeed())

	ginkgo.By("Verifying logs")
	gomega.Eventually(ctx, func() (string, error) {
		return e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	}, postStartWaitTimeout, podCheckInterval).Should(gomega.SatisfyAll(
		gomega.ContainSubstring("Accepted client connection"),
		gomega.ContainSubstring("Received expected client data"),
	))
}

var _ = SIGDescribe("Kubectl Port forwarding", func() {
	f := framework.NewDefaultFramework("port-forwarding")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Describe("With a server listening on 0.0.0.0", func() {
		ginkgo.Describe("that expects a client request", func() {
			ginkgo.It("should support a client that connects, sends NO DATA, and disconnects", func(ctx context.Context) {
				doTestMustConnectSendNothing(ctx, "0.0.0.0", f)
			})
			ginkgo.It("should support a client that connects, sends DATA, and disconnects", func(ctx context.Context) {
				doTestMustConnectSendDisconnect(ctx, "0.0.0.0", f)
			})
		})

		ginkgo.Describe("that expects NO client request", func() {
			ginkgo.It("should support a client that connects, sends DATA, and disconnects", func(ctx context.Context) {
				doTestConnectSendDisconnect(ctx, "0.0.0.0", f)
			})
		})

		ginkgo.It("should support forwarding over websockets", func(ctx context.Context) {
			doTestOverWebSockets(ctx, "0.0.0.0", f)
		})
	})

	// kubectl port-forward may need elevated privileges to do its job.
	ginkgo.Describe("With a server listening on localhost", func() {
		ginkgo.Describe("that expects a client request", func() {
			ginkgo.It("should support a client that connects, sends NO DATA, and disconnects", func(ctx context.Context) {
				doTestMustConnectSendNothing(ctx, "localhost", f)
			})
			ginkgo.It("should support a client that connects, sends DATA, and disconnects", func(ctx context.Context) {
				doTestMustConnectSendDisconnect(ctx, "localhost", f)
			})
		})

		ginkgo.Describe("that expects NO client request", func() {
			ginkgo.It("should support a client that connects, sends DATA, and disconnects", func(ctx context.Context) {
				doTestConnectSendDisconnect(ctx, "localhost", f)
			})
		})

		ginkgo.It("should support forwarding over websockets", func(ctx context.Context) {
			doTestOverWebSockets(ctx, "localhost", f)
		})
	})

	ginkgo.Describe("with a pod being removed", func() {
		ginkgo.It("should stop port-forwarding", func(ctx context.Context) {
			ginkgo.By("Creating the target pod")
			pod := pfNeverReadRequestBodyPod()
			_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err, "couldn't create pod")

			err = e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout)
			framework.ExpectNoError(err, "pod did not start running")

			ginkgo.By("Running 'kubectl port-forward'")
			cmd := runPortForward(f.Namespace.Name, pod.Name, 80)
			defer cmd.Stop()

			ginkgo.By("Running port-forward client")
			reqChan := make(chan bool)
			errorChan := make(chan error)
			go func() {
				defer ginkgo.GinkgoRecover()

				// try to mock a big request, which should take some time
				for sentBodySize := 0; sentBodySize < 1024*1024*1024; {
					size := rand.Intn(4 * 1024 * 1024)
					url := fmt.Sprintf("http://localhost:%d/header", cmd.port)
					_, err := post(url, strings.NewReader(strings.Repeat("x", size)), 10*time.Second)
					if err != nil {
						errorChan <- err
					}
					ginkgo.By(fmt.Sprintf("Sent %d chunk of data", sentBodySize))
					if sentBodySize == 0 {
						close(reqChan)
					}
					sentBodySize += size
				}
			}()

			ginkgo.By("Remove the forwarded pod after the first client request")
			<-reqChan
			e2epod.DeletePodOrFail(ctx, f.ClientSet, f.Namespace.Name, pod.Name)

			ginkgo.By("Wait for client being interrupted")
			select {
			case err = <-errorChan:
			case <-time.After(f.Timeouts.PodDelete):
			}

			ginkgo.By("Check the client error")
			gomega.Expect(err).To(gomega.HaveOccurred())
			gomega.Expect(err.Error()).To(gomega.Or(
				// these two errors indicates remote connection is closed
				gomega.ContainSubstring("connection reset by peer"), gomega.ContainSubstring("EOF"),
				// this error indicates timeout when POST-ing data
				gomega.ContainSubstring("context deadline exceeded"),
				// this will happen when trying to write to a closed connection
				gomega.ContainSubstring("write: broken pipe"), gomega.ContainSubstring("closed network connection")))

			ginkgo.By("Check kubectl port-forward exit code")
			gomega.Expect(cmd.cmd.ProcessState.ExitCode()).To(gomega.BeNumerically("<", 0), "kubectl port-forward should finish with non-zero exit code")
		})
	})

	ginkgo.Describe("Shutdown client connection while the remote stream is writing data to the port-forward connection", func() {
		ginkgo.It("port-forward should keep working after detect broken connection", func(ctx context.Context) {
			ginkgo.By("Creating the target pod")
			pod := testWebServerPod()
			_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err, "couldn't create pod")

			err = e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout)
			framework.ExpectNoError(err, "pod did not start running")

			ginkgo.By("Running 'kubectl port-forward'")
			cmd := runPortForward(f.Namespace.Name, pod.Name, 80)
			defer cmd.Stop()

			ginkgo.By("Send a http request to verify port-forward working")
			client := http.Client{
				Timeout: 10 * time.Second,
			}
			resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/", cmd.port))
			framework.ExpectNoError(err, "couldn't get http response from port-forward")
			gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK), "unexpected status code")

			ginkgo.By("Dialing the local port")
			conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", cmd.port))
			framework.ExpectNoError(err, "couldn't connect to port %d", cmd.port)

			// use raw tcp connection to emulate client close connection without reading response
			ginkgo.By("Request agnhost binary file (40MB+)")
			requestLines := []string{"GET /agnhost HTTP/1.1", "Host: localhost", ""}
			for _, line := range requestLines {
				_, err := conn.Write(append([]byte(line), []byte("\r\n")...))
				framework.ExpectNoError(err, "couldn't write http request to local connection")
			}

			ginkgo.By("Read only one byte from the connection")
			_, err = conn.Read(make([]byte, 1))
			framework.ExpectNoError(err, "couldn't read from the local connection")

			ginkgo.By("Close client connection without reading remain data")
			err = conn.Close()
			framework.ExpectNoError(err, "couldn't close local connection")

			ginkgo.By("Send another http request through port-forward again")
			resp, err = client.Get(fmt.Sprintf("http://127.0.0.1:%d/", cmd.port))
			framework.ExpectNoError(err, "couldn't get http response from port-forward")
			gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK), "unexpected status code")
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

func post(url string, reader io.Reader, timeout time.Duration) (string, error) {
	client := &http.Client{
		Transport: utilnet.SetTransportDefaults(&http.Transport{}),
		Timeout:   timeout,
	}
	req, err := http.NewRequest(http.MethodPost, url, reader)
	if err != nil {
		return "", err
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close() //nolint: errcheck
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}
