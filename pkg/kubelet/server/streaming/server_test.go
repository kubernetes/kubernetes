/*
Copyright 2016 The Kubernetes Authors.

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

package streaming

import (
	"crypto/tls"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/client-go/pkg/api"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubeletportforward "k8s.io/kubernetes/pkg/kubelet/server/portforward"
	kubeletremotecommand "k8s.io/kubernetes/pkg/kubelet/server/remotecommand"
	"k8s.io/kubernetes/pkg/util/term"
)

const (
	testAddr         = "localhost:12345"
	testContainerID  = "container789"
	testPodSandboxID = "pod0987"
)

func TestGetExec(t *testing.T) {
	type testcase struct {
		cmd   []string
		tty   bool
		stdin bool
	}
	testcases := []testcase{
		{[]string{"echo", "foo"}, false, false},
		{[]string{"date"}, true, false},
		{[]string{"date"}, false, true},
		{[]string{"date"}, true, true},
	}
	serv, err := NewServer(Config{
		Addr: testAddr,
	}, nil)
	assert.NoError(t, err)

	tlsServer, err := NewServer(Config{
		Addr:      testAddr,
		TLSConfig: &tls.Config{},
	}, nil)
	assert.NoError(t, err)

	const pathPrefix = "cri/shim"
	prefixServer, err := NewServer(Config{
		Addr: testAddr,
		BaseURL: &url.URL{
			Scheme: "http",
			Host:   testAddr,
			Path:   "/" + pathPrefix + "/",
		},
	}, nil)
	assert.NoError(t, err)

	assertRequestToken := func(test testcase, cache *requestCache, token string) {
		req, ok := cache.Consume(token)
		require.True(t, ok, "token %s not found! testcase=%+v", token, test)
		assert.Equal(t, testContainerID, req.(*runtimeapi.ExecRequest).GetContainerId(), "testcase=%+v", test)
		assert.Equal(t, test.cmd, req.(*runtimeapi.ExecRequest).GetCmd(), "testcase=%+v", test)
		assert.Equal(t, test.tty, req.(*runtimeapi.ExecRequest).GetTty(), "testcase=%+v", test)
		assert.Equal(t, test.stdin, req.(*runtimeapi.ExecRequest).GetStdin(), "testcase=%+v", test)
	}
	containerID := testContainerID
	for _, test := range testcases {
		request := &runtimeapi.ExecRequest{
			ContainerId: &containerID,
			Cmd:         test.cmd,
			Tty:         &test.tty,
			Stdin:       &test.stdin,
		}
		{ // Non-TLS
			resp, err := serv.GetExec(request)
			assert.NoError(t, err, "testcase=%+v", test)
			expectedURL := "http://" + testAddr + "/exec/"
			assert.Contains(t, resp.GetUrl(), expectedURL, "testcase=%+v", test)
			token := strings.TrimPrefix(resp.GetUrl(), expectedURL)
			assertRequestToken(test, serv.(*server).cache, token)
		}

		{ // TLS
			resp, err := tlsServer.GetExec(request)
			assert.NoError(t, err, "testcase=%+v", test)
			expectedURL := "https://" + testAddr + "/exec/"
			assert.Contains(t, resp.GetUrl(), expectedURL, "testcase=%+v", test)
			token := strings.TrimPrefix(resp.GetUrl(), expectedURL)
			assertRequestToken(test, tlsServer.(*server).cache, token)
		}

		{ // Path prefix
			resp, err := prefixServer.GetExec(request)
			assert.NoError(t, err, "testcase=%+v", test)
			expectedURL := "http://" + testAddr + "/" + pathPrefix + "/exec/"
			assert.Contains(t, resp.GetUrl(), expectedURL, "testcase=%+v", test)
			token := strings.TrimPrefix(resp.GetUrl(), expectedURL)
			assertRequestToken(test, prefixServer.(*server).cache, token)
		}
	}
}

func TestGetAttach(t *testing.T) {
	type testcase struct {
		tty   bool
		stdin bool
	}
	testcases := []testcase{
		{false, false},
		{true, false},
		{false, true},
		{true, true},
	}
	serv, err := NewServer(Config{
		Addr: testAddr,
	}, nil)
	assert.NoError(t, err)

	tlsServer, err := NewServer(Config{
		Addr:      testAddr,
		TLSConfig: &tls.Config{},
	}, nil)
	assert.NoError(t, err)

	assertRequestToken := func(test testcase, cache *requestCache, token string) {
		req, ok := cache.Consume(token)
		require.True(t, ok, "token %s not found! testcase=%+v", token, test)
		assert.Equal(t, testContainerID, req.(*runtimeapi.AttachRequest).GetContainerId(), "testcase=%+v", test)
		assert.Equal(t, test.tty, req.(*runtimeapi.AttachRequest).GetTty(), "testcase=%+v", test)
		assert.Equal(t, test.stdin, req.(*runtimeapi.AttachRequest).GetStdin(), "testcase=%+v", test)
	}
	containerID := testContainerID
	for _, test := range testcases {
		request := &runtimeapi.AttachRequest{
			ContainerId: &containerID,
			Stdin:       &test.stdin,
			Tty:         &test.tty,
		}
		{ // Non-TLS
			resp, err := serv.GetAttach(request)
			assert.NoError(t, err, "testcase=%+v", test)
			expectedURL := "http://" + testAddr + "/attach/"
			assert.Contains(t, resp.GetUrl(), expectedURL, "testcase=%+v", test)
			token := strings.TrimPrefix(resp.GetUrl(), expectedURL)
			assertRequestToken(test, serv.(*server).cache, token)
		}

		{ // TLS
			resp, err := tlsServer.GetAttach(request)
			assert.NoError(t, err, "testcase=%+v", test)
			expectedURL := "https://" + testAddr + "/attach/"
			assert.Contains(t, resp.GetUrl(), expectedURL, "testcase=%+v", test)
			token := strings.TrimPrefix(resp.GetUrl(), expectedURL)
			assertRequestToken(test, tlsServer.(*server).cache, token)
		}
	}
}

func TestGetPortForward(t *testing.T) {
	podSandboxID := testPodSandboxID
	request := &runtimeapi.PortForwardRequest{
		PodSandboxId: &podSandboxID,
		Port:         []int32{1, 2, 3, 4},
	}

	{ // Non-TLS
		serv, err := NewServer(Config{
			Addr: testAddr,
		}, nil)
		assert.NoError(t, err)
		resp, err := serv.GetPortForward(request)
		assert.NoError(t, err)
		expectedURL := "http://" + testAddr + "/portforward/"
		assert.True(t, strings.HasPrefix(resp.GetUrl(), expectedURL))
		token := strings.TrimPrefix(resp.GetUrl(), expectedURL)
		req, ok := serv.(*server).cache.Consume(token)
		require.True(t, ok, "token %s not found!", token)
		assert.Equal(t, testPodSandboxID, req.(*runtimeapi.PortForwardRequest).GetPodSandboxId())
	}

	{ // TLS
		tlsServer, err := NewServer(Config{
			Addr:      testAddr,
			TLSConfig: &tls.Config{},
		}, nil)
		assert.NoError(t, err)
		resp, err := tlsServer.GetPortForward(request)
		assert.NoError(t, err)
		expectedURL := "https://" + testAddr + "/portforward/"
		assert.True(t, strings.HasPrefix(resp.GetUrl(), expectedURL))
		token := strings.TrimPrefix(resp.GetUrl(), expectedURL)
		req, ok := tlsServer.(*server).cache.Consume(token)
		require.True(t, ok, "token %s not found!", token)
		assert.Equal(t, testPodSandboxID, req.(*runtimeapi.PortForwardRequest).GetPodSandboxId())
	}
}

func TestServeExec(t *testing.T) {
	runRemoteCommandTest(t, "exec")
}

func TestServeAttach(t *testing.T) {
	runRemoteCommandTest(t, "attach")
}

func TestServePortForward(t *testing.T) {
	s, testServer := startTestServer(t)
	defer testServer.Close()

	podSandboxID := testPodSandboxID
	resp, err := s.GetPortForward(&runtimeapi.PortForwardRequest{
		PodSandboxId: &podSandboxID,
	})
	require.NoError(t, err)
	reqURL, err := url.Parse(resp.GetUrl())
	require.NoError(t, err)

	exec, err := remotecommand.NewExecutor(&restclient.Config{}, "POST", reqURL)
	require.NoError(t, err)
	streamConn, _, err := exec.Dial(kubeletportforward.PortForwardProtocolV1Name)
	require.NoError(t, err)
	defer streamConn.Close()

	// Create the streams.
	headers := http.Header{}
	// Error stream is required, but unused in this test.
	headers.Set(api.StreamType, api.StreamTypeError)
	headers.Set(api.PortHeader, strconv.Itoa(testPort))
	_, err = streamConn.CreateStream(headers)
	require.NoError(t, err)
	// Setup the data stream.
	headers.Set(api.StreamType, api.StreamTypeData)
	headers.Set(api.PortHeader, strconv.Itoa(testPort))
	stream, err := streamConn.CreateStream(headers)
	require.NoError(t, err)

	doClientStreams(t, "portforward", stream, stream, nil)
}

// Run the remote command test.
// commandType is either "exec" or "attach".
func runRemoteCommandTest(t *testing.T, commandType string) {
	s, testServer := startTestServer(t)
	defer testServer.Close()

	var reqURL *url.URL
	stdin := true
	containerID := testContainerID
	switch commandType {
	case "exec":
		resp, err := s.GetExec(&runtimeapi.ExecRequest{
			ContainerId: &containerID,
			Cmd:         []string{"echo"},
			Stdin:       &stdin,
		})
		require.NoError(t, err)
		reqURL, err = url.Parse(resp.GetUrl())
		require.NoError(t, err)
	case "attach":
		resp, err := s.GetAttach(&runtimeapi.AttachRequest{
			ContainerId: &containerID,
			Stdin:       &stdin,
		})
		require.NoError(t, err)
		reqURL, err = url.Parse(resp.GetUrl())
		require.NoError(t, err)
	}

	wg := sync.WaitGroup{}
	wg.Add(2)

	stdinR, stdinW := io.Pipe()
	stdoutR, stdoutW := io.Pipe()
	stderrR, stderrW := io.Pipe()

	go func() {
		defer wg.Done()
		exec, err := remotecommand.NewExecutor(&restclient.Config{}, "POST", reqURL)
		require.NoError(t, err)

		opts := remotecommand.StreamOptions{
			SupportedProtocols: kubeletremotecommand.SupportedStreamingProtocols,
			Stdin:              stdinR,
			Stdout:             stdoutW,
			Stderr:             stderrW,
			Tty:                false,
			TerminalSizeQueue:  nil,
		}
		require.NoError(t, exec.Stream(opts))
	}()

	go func() {
		defer wg.Done()
		doClientStreams(t, commandType, stdinW, stdoutR, stderrR)
	}()

	wg.Wait()

	// Repeat request with the same URL should be a 404.
	resp, err := http.Get(reqURL.String())
	require.NoError(t, err)
	assert.Equal(t, http.StatusNotFound, resp.StatusCode)
}

func startTestServer(t *testing.T) (Server, *httptest.Server) {
	var s Server
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.ServeHTTP(w, r)
	}))
	cleanup := true
	defer func() {
		if cleanup {
			testServer.Close()
		}
	}()

	testURL, err := url.Parse(testServer.URL)
	require.NoError(t, err)

	rt := newFakeRuntime(t)
	config := DefaultConfig
	config.BaseURL = testURL
	s, err = NewServer(config, rt)
	require.NoError(t, err)

	cleanup = false // Caller must close the test server.
	return s, testServer
}

const (
	testInput  = "abcdefg"
	testOutput = "fooBARbaz"
	testErr    = "ERROR!!!"
	testPort   = 12345
)

func newFakeRuntime(t *testing.T) *fakeRuntime {
	return &fakeRuntime{
		t: t,
	}
}

type fakeRuntime struct {
	t *testing.T
}

func (f *fakeRuntime) Exec(containerID string, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) error {
	assert.Equal(f.t, testContainerID, containerID)
	doServerStreams(f.t, "exec", stdin, stdout, stderr)
	return nil
}

func (f *fakeRuntime) Attach(containerID string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) error {
	assert.Equal(f.t, testContainerID, containerID)
	doServerStreams(f.t, "attach", stdin, stdout, stderr)
	return nil
}

func (f *fakeRuntime) PortForward(podSandboxID string, port int32, stream io.ReadWriteCloser) error {
	assert.Equal(f.t, testPodSandboxID, podSandboxID)
	assert.EqualValues(f.t, testPort, port)
	doServerStreams(f.t, "portforward", stream, stream, nil)
	return nil
}

// Send & receive expected input/output. Must be the inverse of doClientStreams.
// Function will block until the expected i/o is finished.
func doServerStreams(t *testing.T, prefix string, stdin io.Reader, stdout, stderr io.Writer) {
	if stderr != nil {
		writeExpected(t, "server stderr", stderr, prefix+testErr)
	}
	readExpected(t, "server stdin", stdin, prefix+testInput)
	writeExpected(t, "server stdout", stdout, prefix+testOutput)
}

// Send & receive expected input/output. Must be the inverse of doServerStreams.
// Function will block until the expected i/o is finished.
func doClientStreams(t *testing.T, prefix string, stdin io.Writer, stdout, stderr io.Reader) {
	if stderr != nil {
		readExpected(t, "client stderr", stderr, prefix+testErr)
	}
	writeExpected(t, "client stdin", stdin, prefix+testInput)
	readExpected(t, "client stdout", stdout, prefix+testOutput)
}

// Read and verify the expected string from the stream.
func readExpected(t *testing.T, streamName string, r io.Reader, expected string) {
	result := make([]byte, len(expected))
	_, err := io.ReadAtLeast(r, result, len(expected))
	assert.NoError(t, err, "stream %s", streamName)
	assert.Equal(t, expected, string(result), "stream %s", streamName)
}

// Write and verify success of the data over the stream.
func writeExpected(t *testing.T, streamName string, w io.Writer, data string) {
	n, err := io.WriteString(w, data)
	assert.NoError(t, err, "stream %s", streamName)
	assert.Equal(t, len(data), n, "stream %s", streamName)
}
