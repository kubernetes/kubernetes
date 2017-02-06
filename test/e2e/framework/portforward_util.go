/*
Copyright 2017 The Kubernetes Authors.

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

package framework

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"regexp"
	"strconv"
	"strings"
	"time"

	"golang.org/x/net/websocket"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	utilversion "k8s.io/kubernetes/pkg/util/version"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	pfPodName = "pfpod"
)

var (
	portForwardRegexp        = regexp.MustCompile("Forwarding from 127.0.0.1:([0-9]+) -> 80")
	portForwardPortToStdOutV = utilversion.MustParseSemantic("v1.3.0-alpha.4")
)

// PortForwardCommand defines interface for portforwarding
type PortForwardCommand interface {
	Run() int
	Stop()
	String() string
}

type PortForwardOptions struct {
	Clientset    clientset.Interface
	PodName      string
	Namespace    string
	Ports        []string
	StopChannel  chan struct{}
	ReadyChannel chan struct{}

	Out    io.ReadWriter
	ErrOut io.ReadWriter
}

func MakePortForwardCommand(f *Framework) PortForwardCommand {
	return &PortForwardOptions{
		Clientset:    f.ClientSet,
		PodName:      pfPodName,
		Namespace:    f.Namespace.Name,
		Ports:        []string{":80"},
		StopChannel:  make(chan struct{}),
		ReadyChannel: make(chan struct{}),
		Out:          &bytes.Buffer{},
		ErrOut:       &bytes.Buffer{},
	}
}

// Run starts portfowarding, waits until it will be started and returns local port
func (p *PortForwardOptions) Run() int {
	config, err := LoadConfig()
	Expect(err).NotTo(HaveOccurred(), "failed to load restclient config")
	req := p.Clientset.Core().RESTClient().Post().
		Resource("pods").
		Name(p.PodName).
		Namespace(p.Namespace).
		SubResource("portforward")
	dialer, err := remotecommand.NewExecutor(config, "POST", req.URL())
	Expect(err).NotTo(HaveOccurred())
	fw, err := portforward.New(dialer, p.Ports, p.StopChannel, p.ReadyChannel, p.Out, p.ErrOut)
	Expect(err).NotTo(HaveOccurred())
	go func() {
		Expect(fw.ForwardPorts()).NotTo(HaveOccurred())
	}()
	select {
	case <-time.After(wait.ForeverTestTimeout):
		Failf("Timed out on wait to start portforwarding.")
	case <-p.ReadyChannel:
	}
	return FindPortFromOutput(p.Out)
}

// Stop portforwarding
func (p *PortForwardOptions) Stop() {
	close(p.StopChannel)
}

// String returns description for port forward command
func (p *PortForwardOptions) String() string {
	return fmt.Sprintf("port forwarding for Pod: %s on ports  %v", p.PodName, p.Ports)
}

// FindPortFromOutput get data from provided reader and returns matched port
func FindPortFromOutput(out io.Reader) int {
	buf := make([]byte, 128)
	n, err := out.Read(buf)
	if err != nil {
		Failf("Failed to read from portforward output: %v", err)
	}
	portForwardOutput := string(buf[:n])
	match := portForwardRegexp.FindStringSubmatch(portForwardOutput)
	if len(match) != 2 {
		Failf("Failed to parse portforward output: %s", portForwardOutput)
	}
	listenPort, err := strconv.Atoi(match[1])
	if err != nil {
		Failf("Error converting %s to an int: %v", match[1], err)
	}
	return listenPort
}

// PortForwardPod returns common pod for portforwarding tests
func PortForwardPod(expectedClientData, chunks, chunkSize, chunkIntervalMillis string, bindAddress string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   pfPodName,
			Labels: map[string]string{"name": pfPodName},
		},
		Spec: v1.PodSpec{
			NodeName: TestContext.NodeName,
			Containers: []v1.Container{
				{
					Name:  "readiness",
					Image: "gcr.io/google_containers/netexec:1.7",
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
					Image: "gcr.io/google_containers/portforwardtester:1.2",
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

func WaitForTerminatedContainer(f *Framework, pod *v1.Pod, containerName string) error {
	return WaitForPodCondition(f.ClientSet, f.Namespace.Name, pod.Name, "container terminated", PodStartTimeout, func(pod *v1.Pod) (bool, error) {
		if len(testutils.TerminatedContainers(pod)[containerName]) > 0 {
			return true, nil
		}
		return false, nil
	})
}

func DoTestConnectSendDisconnect(command PortForwardCommand, bindAddress string, f *Framework) {
	By("creating the target pod")
	pod := PortForwardPod("", "10", "10", "100", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod); err != nil {
		Failf("Couldn't create pod: %v", err)
	}
	if err := f.WaitForPodReady(pod.Name); err != nil {
		Failf("Pod did not start running: %v", err)
	}
	defer func() {
		logs, err := GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
		if err != nil {
			Logf("Error getting pod log: %v", err)
		} else {
			Logf("Pod log:\n%s", logs)
		}
	}()

	By("Running " + command.String())
	port := command.Run()
	defer command.Stop()

	By("Dialing the local port")
	conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		Failf("Couldn't connect to port %d: %v", port, err)
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

	By("Waiting for the target pod to stop running")
	if err := WaitForTerminatedContainer(f, pod, "portforwardtester"); err != nil {
		Failf("Container did not terminate: %v", err)
	}

	By("Verifying logs")
	logOutput, err := GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	if err != nil {
		Failf("Error retrieving pod logs: %v", err)
	}
	verifyLogMessage(logOutput, "Accepted client connection")
	verifyLogMessage(logOutput, "Done")
}

func DoTestMustConnectSendNothing(command PortForwardCommand, bindAddress string, f *Framework) {
	By("creating the target pod")
	pod := PortForwardPod("abc", "1", "1", "1", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod); err != nil {
		Failf("Couldn't create pod: %v", err)
	}
	if err := f.WaitForPodReady(pod.Name); err != nil {
		Failf("Pod did not start running: %v", err)
	}
	defer func() {
		logs, err := GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
		if err != nil {
			Logf("Error getting pod log: %v", err)
		} else {
			Logf("Pod log:\n%s", logs)
		}
	}()

	By("Running " + command.String())
	port := command.Run()
	defer command.Stop()

	By("Dialing the local port")
	conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		Failf("Couldn't connect to port %d: %v", port, err)
	}

	By("Closing the connection to the local port")
	conn.Close()

	By("Waiting for the target pod to stop running")
	if err := WaitForTerminatedContainer(f, pod, "portforwardtester"); err != nil {
		Failf("Container did not terminate: %v", err)
	}

	By("Verifying logs")
	logOutput, err := GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	if err != nil {
		Failf("Error retrieving pod logs: %v", err)
	}
	verifyLogMessage(logOutput, "Accepted client connection")
	verifyLogMessage(logOutput, "Expected to read 3 bytes from client, but got 0 instead")
}

func DoTestMustConnectSendDisconnect(command PortForwardCommand, bindAddress string, f *Framework) {
	By("creating the target pod")
	pod := PortForwardPod("abc", "10", "10", "100", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod); err != nil {
		Failf("Couldn't create pod: %v", err)
	}
	if err := f.WaitForPodReady(pod.Name); err != nil {
		Failf("Pod did not start running: %v", err)
	}
	defer func() {
		logs, err := GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
		if err != nil {
			Logf("Error getting pod log: %v", err)
		} else {
			Logf("Pod log:\n%s", logs)
		}
	}()

	By("Running " + command.String())
	port := command.Run()
	defer command.Stop()

	By("Dialing the local port")
	addr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		Failf("Error resolving tcp addr: %v", err)
	}
	conn, err := net.DialTCP("tcp", nil, addr)
	if err != nil {
		Failf("Couldn't connect to port %d: %v", port, err)
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

	By("Waiting for the target pod to stop running")
	if err := WaitForTerminatedContainer(f, pod, "portforwardtester"); err != nil {
		Failf("Container did not terminate: %v", err)
	}

	By("Verifying logs")
	logOutput, err := GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	if err != nil {
		Failf("Error retrieving pod logs: %v", err)
	}
	verifyLogMessage(logOutput, "^Accepted client connection$")
	verifyLogMessage(logOutput, "^Received expected client data$")
	verifyLogMessage(logOutput, "^Done$")
}

func DoTestOverWebSockets(bindAddress string, f *Framework) {
	config, err := LoadConfig()
	Expect(err).NotTo(HaveOccurred(), "unable to get base config")

	By("creating the pod")
	pod := PortForwardPod("def", "10", "10", "100", fmt.Sprintf("%s", bindAddress))
	if _, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod); err != nil {
		Failf("Couldn't create pod: %v", err)
	}
	if err := f.WaitForPodReady(pod.Name); err != nil {
		Failf("Pod did not start running: %v", err)
	}
	defer func() {
		logs, err := GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
		if err != nil {
			Logf("Error getting pod log: %v", err)
		} else {
			Logf("Pod log:\n%s", logs)
		}
	}()

	req := f.ClientSet.Core().RESTClient().Get().
		Namespace(f.Namespace.Name).
		Resource("pods").
		Name(pod.Name).
		Suffix("portforward").
		Param("ports", "80")

	url := req.URL()
	ws, err := OpenWebSocketForURL(url, config, []string{"v4.channel.k8s.io"})
	if err != nil {
		Failf("Failed to open websocket to %s: %v", url.String(), err)
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

	By("sending the expected data to the local port")
	err = wsWrite(ws, 0, []byte("def"))
	if err != nil {
		Failf("Failed to write to websocket %s: %v", url.String(), err)
	}

	By("reading data from the local port")
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

	By("verifying logs")
	logOutput, err := GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	if err != nil {
		Failf("Error retrieving pod logs: %v", err)
	}
	verifyLogMessage(logOutput, "^Accepted client connection$")
	verifyLogMessage(logOutput, "^Received expected client data$")
}
