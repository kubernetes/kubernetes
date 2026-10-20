/*
Copyright The Kubernetes Authors.

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

package apimachinery

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	rcconstants "k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/tools/remotecommand"
	utilexec "k8s.io/client-go/util/exec"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// These tests exercise the WebSocket streaming protocols introduced by
// KEP-4006 end to end, with no fallback to SPDY: the RemoteCommand
// subprotocol v5.channel.k8s.io for exec and attach, and the SPDY-over-WebSocket
// tunnel SPDY/3.1+portforward.k8s.io for port forwarding. A cluster that does
// not serve these protocols fails them, unlike the kubectl-driven tests, which
// fall back to SPDY.
var _ = SIGDescribe("WebSocket streaming", func() {
	f := framework.NewDefaultFramework("websocket-streaming")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("should exec over the WebSocket v5.channel.k8s.io subprotocol with stdin close and exit code", func(ctx context.Context) {
		ginkgo.By("creating a pod to exec into")
		pod := e2epod.NewPodClient(f).CreateSync(ctx, e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "websocket-exec"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{
					Name:    "main",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c", "sleep 600"},
				}},
			},
		}))

		ginkgo.By("executing a command that reads stdin to EOF and exits non-zero, over v5 only")
		// cat only returns once the client half-closes stdin, which v5 signals
		// with its CLOSE message; the exit code arrives on the status channel.
		req := f.ClientSet.CoreV1().RESTClient().Post().
			Resource("pods").Namespace(f.Namespace.Name).Name(pod.Name).SubResource("exec").
			VersionedParams(&v1.PodExecOptions{
				Container: pod.Spec.Containers[0].Name,
				Command:   []string{"sh", "-c", "cat; echo stdin closed; exit 3"},
				Stdin:     true,
				Stdout:    true,
				Stderr:    true,
			}, scheme.ParameterCodec)
		executor, err := remotecommand.NewWebSocketExecutorForProtocols(f.ClientConfig(), http.MethodGet, req.URL().String(), rcconstants.StreamProtocolV5Name)
		framework.ExpectNoError(err, "creating the v5 WebSocket executor")

		streamCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
		defer cancel()
		var stdout, stderr bytes.Buffer
		err = executor.StreamWithContext(streamCtx, remotecommand.StreamOptions{
			Stdin:  strings.NewReader("abcd1234"),
			Stdout: &stdout,
			Stderr: &stderr,
		})

		var exitErr utilexec.CodeExitError
		if !errors.As(err, &exitErr) {
			framework.Failf("expected the command's exit code as a CodeExitError, got %v (stdout %q, stderr %q)", err, stdout.String(), stderr.String())
		}
		gomega.Expect(exitErr.ExitStatus()).To(gomega.Equal(3), "exit code from the status channel")
		gomega.Expect(stdout.String()).To(gomega.ContainSubstring("abcd1234"), "stdin was delivered to the command")
		gomega.Expect(stdout.String()).To(gomega.ContainSubstring("stdin closed"), "the CLOSE signal ended the command's stdin")
	})

	ginkgo.It("should attach over the WebSocket v5.channel.k8s.io subprotocol", func(ctx context.Context) {
		ginkgo.By("creating a pod whose container waits on stdin")
		pod := e2epod.NewPodClient(f).CreateSync(ctx, e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "websocket-attach"},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				Containers: []v1.Container{{
					Name:      "main",
					Image:     imageutils.GetE2EImage(imageutils.BusyBox),
					Command:   []string{"sh", "-c", "cat && echo stdin closed"},
					Stdin:     true,
					StdinOnce: true,
				}},
			},
		}))

		ginkgo.By("attaching over v5 only, sending stdin and closing it")
		req := f.ClientSet.CoreV1().RESTClient().Post().
			Resource("pods").Namespace(f.Namespace.Name).Name(pod.Name).SubResource("attach").
			VersionedParams(&v1.PodAttachOptions{
				Container: pod.Spec.Containers[0].Name,
				Stdin:     true,
				Stdout:    true,
				Stderr:    true,
			}, scheme.ParameterCodec)
		executor, err := remotecommand.NewWebSocketExecutorForProtocols(f.ClientConfig(), http.MethodGet, req.URL().String(), rcconstants.StreamProtocolV5Name)
		framework.ExpectNoError(err, "creating the v5 WebSocket executor")

		streamCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
		defer cancel()
		var stdout, stderr bytes.Buffer
		err = executor.StreamWithContext(streamCtx, remotecommand.StreamOptions{
			Stdin:  strings.NewReader("attached"),
			Stdout: &stdout,
			Stderr: &stderr,
		})
		framework.ExpectNoError(err, "attach session (stdout %q, stderr %q)", stdout.String(), stderr.String())
		gomega.Expect(stdout.String()).To(gomega.ContainSubstring("attached"), "stdin was delivered to the container")
		gomega.Expect(stdout.String()).To(gomega.ContainSubstring("stdin closed"), "the CLOSE signal ended the container's stdin")
	})

	ginkgo.Describe("port forwarding over the WebSocket tunnel", func() {
		ginkgo.It("should forward to a server listening on 0.0.0.0", func(ctx context.Context) {
			testPortForwardOverWebSocketTunnel(ctx, f, "0.0.0.0")
		})
		ginkgo.It("should forward to a server listening on localhost", func(ctx context.Context) {
			testPortForwardOverWebSocketTunnel(ctx, f, "localhost")
		})
	})
})

// portForwardTesterPod runs agnhost's port-forward-tester on port 80: it waits
// for the client to send expectedClientData, answers with chunks of "x", and
// closes. A second container's readiness probe gates on the port being bound.
func portForwardTesterPod(name, bindAddress, expectedClientData string, chunks, chunkSize int) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "readiness",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"netexec"},
					ReadinessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{Exec: &v1.ExecAction{
							Command: []string{"sh", "-c", "netstat -na | grep LISTEN | grep -v 8080 | grep 80"},
						}},
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
						{Name: "BIND_PORT", Value: "80"},
						{Name: "BIND_ADDRESS", Value: bindAddress},
						{Name: "EXPECTED_CLIENT_DATA", Value: expectedClientData},
						{Name: "CHUNKS", Value: fmt.Sprint(chunks)},
						{Name: "CHUNK_SIZE", Value: fmt.Sprint(chunkSize)},
						{Name: "CHUNK_INTERVAL", Value: "100"},
					},
				},
			},
		},
	}
}

// testPortForwardOverWebSocketTunnel dials pods/<name>/portforward with the
// SPDY-over-WebSocket tunneling dialer (subprotocol SPDY/3.1+portforward.k8s.io)
// and drives the port-forward streams directly, so the tunnel is the only path
// that can satisfy the test.
func testPortForwardOverWebSocketTunnel(ctx context.Context, f *framework.Framework, bindAddress string) {
	const (
		clientData = "def"
		chunks     = 10
		chunkSize  = 10
	)

	ginkgo.By("creating the port-forward tester pod")
	pod := portForwardTesterPod("websocket-portforward", bindAddress, clientData, chunks, chunkSize)
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating the pod")
	framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout), "waiting for the pod to be ready")

	ginkgo.By("dialing pods/portforward through the WebSocket tunnel")
	url := f.ClientSet.CoreV1().RESTClient().Post().
		Resource("pods").Namespace(f.Namespace.Name).Name(pod.Name).SubResource("portforward").URL()
	dialer, err := portforward.NewSPDYOverWebsocketDialer(url, f.ClientConfig())
	framework.ExpectNoError(err, "creating the tunneling dialer")
	conn, protocol, err := dialer.Dial(portforward.PortForwardProtocolV1Name)
	framework.ExpectNoError(err, "dialing %s over the WebSocket tunnel", url)
	defer conn.Close() //nolint:errcheck // nothing to do with a close error at the end of the test
	gomega.Expect(protocol).To(gomega.Equal(portforward.PortForwardProtocolV1Name), "port-forward protocol negotiated inside the tunnel")

	ginkgo.By("opening the error and data streams for port 80")
	headers := http.Header{}
	headers.Set(v1.StreamType, v1.StreamTypeError)
	headers.Set(v1.PortHeader, "80")
	headers.Set(v1.PortForwardRequestIDHeader, "0")
	errorStream, err := conn.CreateStream(headers)
	framework.ExpectNoError(err, "creating the error stream")
	headers.Set(v1.StreamType, v1.StreamTypeData)
	dataStream, err := conn.CreateStream(headers)
	framework.ExpectNoError(err, "creating the data stream")

	ginkgo.By("sending the expected client data and reading the response")
	_, err = dataStream.Write([]byte(clientData))
	framework.ExpectNoError(err, "writing to the data stream")

	type readResult struct {
		data []byte
		err  error
	}
	read := func(r io.Reader) <-chan readResult {
		ch := make(chan readResult, 1)
		go func() {
			data, err := io.ReadAll(r)
			ch <- readResult{data: data, err: err}
		}()
		return ch
	}
	dataCh, errCh := read(dataStream), read(errorStream)
	deadline := time.After(time.Minute)
	var data readResult
	select {
	case data = <-dataCh:
	case <-deadline:
		framework.Failf("timed out reading the forwarded data")
	}
	framework.ExpectNoError(data.err, "reading the data stream")
	gomega.Expect(data.data).To(gomega.Equal(bytes.Repeat([]byte("x"), chunks*chunkSize)), "data forwarded from the pod")
	select {
	case errResult := <-errCh:
		framework.ExpectNoError(errResult.err, "reading the error stream")
		gomega.Expect(string(errResult.data)).To(gomega.BeEmpty(), "port-forward error stream")
	case <-deadline:
		framework.Failf("timed out reading the error stream")
	}

	ginkgo.By("checking the pod saw the forwarded connection")
	gomega.Eventually(ctx, func() (string, error) {
		return e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "portforwardtester")
	}, time.Minute, 5*time.Second).Should(gomega.SatisfyAll(
		gomega.ContainSubstring("Accepted client connection"),
		gomega.ContainSubstring("Received expected client data"),
		gomega.ContainSubstring("Done"),
	))
}
