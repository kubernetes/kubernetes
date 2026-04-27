/*
Copyright 2025 The Kubernetes Authors.

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

package tests

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	restclient "k8s.io/client-go/rest"
	remoteclient "k8s.io/client-go/tools/remotecommand"
	clientspdy "k8s.io/client-go/transport/spdy"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	streaming "k8s.io/cri-streaming/pkg/streaming"
	remotecommandserver "k8s.io/cri-streaming/pkg/streaming/remotecommand"
	streamingspdy "k8s.io/streaming/pkg/httpstream/spdy"
)

const (
	execContainerID   = "cri-streaming-exec-container"
	attachContainerID = "cri-streaming-attach-container"

	execInput  = "exec-stdin"
	execOutput = "exec-stdout"
	execErr    = "exec-stderr"

	attachInput  = "attach-stdin"
	attachOutput = "attach-stdout"
	attachErr    = "attach-stderr"
)

type fakeStreamingRuntime struct{}

func (*fakeStreamingRuntime) Exec(ctx context.Context, containerID string, cmd []string, in io.Reader, out, errStream io.WriteCloser, tty bool, resize <-chan remotecommandserver.TerminalSize) error {
	if containerID != execContainerID {
		return fmt.Errorf("unexpected exec container ID: %q", containerID)
	}

	stdinData, err := io.ReadAll(in)
	if err != nil {
		return err
	}
	if string(stdinData) != execInput {
		return fmt.Errorf("unexpected exec stdin: %q", string(stdinData))
	}

	if _, err := io.WriteString(out, execOutput); err != nil {
		return err
	}
	if _, err := io.WriteString(errStream, execErr); err != nil {
		return err
	}

	return nil
}

func (*fakeStreamingRuntime) Attach(ctx context.Context, containerID string, in io.Reader, out, errStream io.WriteCloser, tty bool, resize <-chan remotecommandserver.TerminalSize) error {
	if containerID != attachContainerID {
		return fmt.Errorf("unexpected attach container ID: %q", containerID)
	}

	stdinData, err := io.ReadAll(in)
	if err != nil {
		return err
	}
	if string(stdinData) != attachInput {
		return fmt.Errorf("unexpected attach stdin: %q", string(stdinData))
	}

	if _, err := io.WriteString(out, attachOutput); err != nil {
		return err
	}
	if _, err := io.WriteString(errStream, attachErr); err != nil {
		return err
	}

	return nil
}

func (*fakeStreamingRuntime) PortForward(ctx context.Context, podSandboxID string, port int32, stream io.ReadWriteCloser) error {
	return errors.New("not implemented")
}

func newCRIStreamingTestServer(t *testing.T) (streaming.Server, *httptest.Server) {
	t.Helper()

	var server streaming.Server
	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		server.ServeHTTP(w, r)
	}))

	baseURL, err := url.Parse(httpServer.URL)
	if err != nil {
		httpServer.Close()
		t.Fatalf("failed to parse test server URL: %v", err)
	}

	config := streaming.DefaultConfig
	config.BaseURL = baseURL

	server, err = streaming.NewServer(config, &fakeStreamingRuntime{})
	if err != nil {
		httpServer.Close()
		t.Fatalf("failed to create cri streaming server: %v", err)
	}

	return server, httpServer
}

// Verifies that client-go's SPDY executor can successfully stream against
// the extracted cri-streaming server for both Exec and Attach endpoints.
func TestCRIStreamingSPDYExecAttachCompatibility(t *testing.T) {
	server, httpServer := newCRIStreamingTestServer(t)
	defer httpServer.Close()

	t.Run("exec", func(t *testing.T) {
		response, err := server.GetExec(&runtimeapi.ExecRequest{
			ContainerId: execContainerID,
			Cmd:         []string{"echo", "exec"},
			Stdin:       true,
			Stdout:      true,
			Stderr:      true,
		})
		if err != nil {
			t.Fatalf("failed to get exec URL: %v", err)
		}

		requestURL, err := url.Parse(response.Url)
		if err != nil {
			t.Fatalf("failed to parse exec URL: %v", err)
		}

		executor, err := remoteclient.NewSPDYExecutor(&restclient.Config{Host: requestURL.Host}, "POST", requestURL)
		if err != nil {
			t.Fatalf("failed to build exec executor: %v", err)
		}

		var stdout bytes.Buffer
		var stderr bytes.Buffer
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		err = executor.StreamWithContext(ctx, remoteclient.StreamOptions{
			Stdin:  strings.NewReader(execInput),
			Stdout: &stdout,
			Stderr: &stderr,
		})
		if err != nil {
			t.Fatalf("unexpected exec stream error: %v", err)
		}

		if got := stdout.String(); got != execOutput {
			t.Fatalf("unexpected exec stdout: %q", got)
		}
		if got := stderr.String(); got != execErr {
			t.Fatalf("unexpected exec stderr: %q", got)
		}
	})

	t.Run("attach", func(t *testing.T) {
		response, err := server.GetAttach(&runtimeapi.AttachRequest{
			ContainerId: attachContainerID,
			Stdin:       true,
			Stdout:      true,
			Stderr:      true,
		})
		if err != nil {
			t.Fatalf("failed to get attach URL: %v", err)
		}

		requestURL, err := url.Parse(response.Url)
		if err != nil {
			t.Fatalf("failed to parse attach URL: %v", err)
		}

		executor, err := remoteclient.NewSPDYExecutor(&restclient.Config{Host: requestURL.Host}, "POST", requestURL)
		if err != nil {
			t.Fatalf("failed to build attach executor: %v", err)
		}

		var stdout bytes.Buffer
		var stderr bytes.Buffer
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		err = executor.StreamWithContext(ctx, remoteclient.StreamOptions{
			Stdin:  strings.NewReader(attachInput),
			Stdout: &stdout,
			Stderr: &stderr,
		})
		if err != nil {
			t.Fatalf("unexpected attach stream error: %v", err)
		}

		if got := stdout.String(); got != attachOutput {
			t.Fatalf("unexpected attach stdout: %q", got)
		}
		if got := stderr.String(); got != attachErr {
			t.Fatalf("unexpected attach stderr: %q", got)
		}
	})
}

// Verifies client-go's compatibility upgrader adapter path by building a
// streaming SPDY roundtripper from k8s.io/streaming and adapting it into a
// client-go remotecommand executor.
func TestCRIStreamingSPDYExecCompatibilityWithStreamingUpgraderAdapter(t *testing.T) {
	server, httpServer := newCRIStreamingTestServer(t)
	defer httpServer.Close()

	response, err := server.GetExec(&runtimeapi.ExecRequest{
		ContainerId: execContainerID,
		Cmd:         []string{"echo", "exec"},
		Stdin:       true,
		Stdout:      true,
		Stderr:      true,
	})
	if err != nil {
		t.Fatalf("failed to get exec URL: %v", err)
	}

	requestURL, err := url.Parse(response.Url)
	if err != nil {
		t.Fatalf("failed to parse exec URL: %v", err)
	}

	roundTripper, err := streamingspdy.NewRoundTripperWithConfig(streamingspdy.RoundTripperConfig{
		PingPeriod: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("failed to build streaming roundtripper: %v", err)
	}

	executor, err := remoteclient.NewSPDYExecutorForTransports(
		roundTripper,
		clientspdy.NewUpgraderForStreaming(roundTripper),
		"POST",
		requestURL,
	)
	if err != nil {
		t.Fatalf("failed to build adapter-based executor: %v", err)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = executor.StreamWithContext(ctx, remoteclient.StreamOptions{
		Stdin:  strings.NewReader(execInput),
		Stdout: &stdout,
		Stderr: &stderr,
	})
	if err != nil {
		t.Fatalf("unexpected exec stream error: %v", err)
	}

	if got := stdout.String(); got != execOutput {
		t.Fatalf("unexpected exec stdout: %q", got)
	}
	if got := stderr.String(); got != execErr {
		t.Fatalf("unexpected exec stderr: %q", got)
	}
}
