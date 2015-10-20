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

package remotecommand

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/httpstream/spdy"
)

func fakeExecServer(t *testing.T, i int, stdinData, stdoutData, stderrData, errorData string, tty bool, messageCount int) http.HandlerFunc {
	// error + stdin + stdout
	expectedStreams := 3
	if !tty {
		// stderr
		expectedStreams++
	}

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamCh := make(chan httpstream.Stream)

		upgrader := spdy.NewResponseUpgrader()
		conn, protocol := upgrader.UpgradeResponse(w, req, []string{StreamProtocolV2Name, StreamProtocolV1Name}, func(stream httpstream.Stream) error {
			streamCh <- stream
			return nil
		})
		// from this point on, we can no longer call methods on w
		if conn == nil {
			// The upgrader is responsible for notifying the client of any errors that
			// occurred during upgrading. All we can do is return here at this point
			// if we weren't successful in upgrading.
			return
		}
		defer conn.Close()
		_ = protocol

		var errorStream, stdinStream, stdoutStream, stderrStream httpstream.Stream
		receivedStreams := 0
	WaitForStreams:
		for {
			select {
			case stream := <-streamCh:
				streamType := stream.Headers().Get(api.StreamType)
				switch streamType {
				case api.StreamTypeError:
					errorStream = stream
					receivedStreams++
				case api.StreamTypeStdin:
					stdinStream = stream
					receivedStreams++
				case api.StreamTypeStdout:
					stdoutStream = stream
					receivedStreams++
				case api.StreamTypeStderr:
					stderrStream = stream
					receivedStreams++
				default:
					t.Errorf("%d: unexpected stream type: %q", i, streamType)
				}

				if receivedStreams == expectedStreams {
					break WaitForStreams
				}
			}
		}

		if len(errorData) > 0 {
			n, err := fmt.Fprint(errorStream, errorData)
			if err != nil {
				t.Errorf("%d: error writing to errorStream: %v", i, err)
			}
			if e, a := len(errorData), n; e != a {
				t.Errorf("%d: expected to write %d bytes to errorStream, but only wrote %d", i, e, a)
			}
			errorStream.Close()
		}

		if len(stdoutData) > 0 {
			for j := 0; j < messageCount; j++ {
				n, err := fmt.Fprint(stdoutStream, stdoutData)
				if err != nil {
					t.Errorf("%d: error writing to stdoutStream: %v", i, err)
				}
				if e, a := len(stdoutData), n; e != a {
					t.Errorf("%d: expected to write %d bytes to stdoutStream, but only wrote %d", i, e, a)
				}
			}
			stdoutStream.Close()
		}
		if len(stderrData) > 0 {
			for j := 0; j < messageCount; j++ {
				n, err := fmt.Fprint(stderrStream, stderrData)
				if err != nil {
					t.Errorf("%d: error writing to stderrStream: %v", i, err)
				}
				if e, a := len(stderrData), n; e != a {
					t.Errorf("%d: expected to write %d bytes to stderrStream, but only wrote %d", i, e, a)
				}
			}
			stderrStream.Close()
		}
		if len(stdinData) > 0 {
			data := make([]byte, len(stdinData))
			for j := 0; j < messageCount; j++ {
				n, err := io.ReadFull(stdinStream, data)
				if err != nil {
					t.Errorf("%d: error reading stdin stream: %v", i, err)
				}
				if e, a := len(stdinData), n; e != a {
					t.Errorf("%d: expected to read %d bytes from stdinStream, but only read %d", i, e, a)
				}
				if e, a := stdinData, string(data); e != a {
					t.Errorf("%d: stdin: expected %q, got %q", i, e, a)
				}
			}
			stdinStream.Close()
		}
	})
}

func TestRequestExecuteRemoteCommand(t *testing.T) {
	testCases := []struct {
		Stdin        string
		Stdout       string
		Stderr       string
		Error        string
		Tty          bool
		MessageCount int
	}{
		{
			Error: "bail",
		},
		{
			Stdin:  "a",
			Stdout: "b",
			Stderr: "c",
			// TODO bump this to a larger number such as 100 once
			// https://github.com/docker/spdystream/issues/55 is fixed and the Godep
			// is bumped. Sending multiple messages over stdin/stdout/stderr results
			// in more frames being spread across multiple spdystream frame workers.
			// This makes it more likely that the spdystream bug will be encountered,
			// where streams are closed as soon as a goaway frame is received, and
			// any pending frames that haven't been processed yet may not be
			// delivered (it's a race).
			MessageCount: 1,
		},
		{
			Stdin:  "a",
			Stdout: "b",
			Tty:    true,
		},
	}

	for i, testCase := range testCases {
		localOut := &bytes.Buffer{}
		localErr := &bytes.Buffer{}

		server := httptest.NewServer(fakeExecServer(t, i, testCase.Stdin, testCase.Stdout, testCase.Stderr, testCase.Error, testCase.Tty, testCase.MessageCount))

		url, _ := url.ParseRequestURI(server.URL)
		c := client.NewRESTClient(url, "x", nil, -1, -1)
		req := c.Post().Resource("testing")
		req.Param("command", "ls")
		req.Param("command", "/")
		conf := &client.Config{
			Host: server.URL,
		}
		e, err := NewExecutor(conf, "POST", req.URL())
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		err = e.Stream(strings.NewReader(strings.Repeat(testCase.Stdin, testCase.MessageCount)), localOut, localErr, testCase.Tty)
		hasErr := err != nil

		if len(testCase.Error) > 0 {
			if !hasErr {
				t.Errorf("%d: expected an error", i)
			} else {
				if e, a := testCase.Error, err.Error(); !strings.Contains(a, e) {
					t.Errorf("%d: expected error stream read '%v', got '%v'", i, e, a)
				}
			}

			server.Close()
			continue
		}

		if hasErr {
			t.Errorf("%d: unexpected error: %v", i, err)
			server.Close()
			continue
		}

		if len(testCase.Stdout) > 0 {
			if e, a := strings.Repeat(testCase.Stdout, testCase.MessageCount), localOut; e != a.String() {
				t.Errorf("%d: expected stdout data '%s', got '%s'", i, e, a)
			}
		}

		if testCase.Stderr != "" {
			if e, a := strings.Repeat(testCase.Stderr, testCase.MessageCount), localErr; e != a.String() {
				t.Errorf("%d: expected stderr data '%s', got '%s'", i, e, a)
			}
		}

		server.Close()
	}
}

// TODO: this test is largely cut and paste, refactor to share code
func TestRequestAttachRemoteCommand(t *testing.T) {
	testCases := []struct {
		Stdin  string
		Stdout string
		Stderr string
		Error  string
		Tty    bool
	}{
		{
			Error: "bail",
		},
		{
			Stdin:  "a",
			Stdout: "b",
			Stderr: "c",
		},
		{
			Stdin:  "a",
			Stdout: "b",
			Tty:    true,
		},
	}

	for i, testCase := range testCases {
		localOut := &bytes.Buffer{}
		localErr := &bytes.Buffer{}

		server := httptest.NewServer(fakeExecServer(t, i, testCase.Stdin, testCase.Stdout, testCase.Stderr, testCase.Error, testCase.Tty, 1))

		url, _ := url.ParseRequestURI(server.URL)
		c := client.NewRESTClient(url, "x", nil, -1, -1)
		req := c.Post().Resource("testing")

		conf := &client.Config{
			Host: server.URL,
		}
		e, err := NewExecutor(conf, "POST", req.URL())
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
			continue
		}
		err = e.Stream(strings.NewReader(testCase.Stdin), localOut, localErr, testCase.Tty)
		hasErr := err != nil

		if len(testCase.Error) > 0 {
			if !hasErr {
				t.Errorf("%d: expected an error", i)
			} else {
				if e, a := testCase.Error, err.Error(); !strings.Contains(a, e) {
					t.Errorf("%d: expected error stream read '%v', got '%v'", i, e, a)
				}
			}

			server.Close()
			continue
		}

		if hasErr {
			t.Errorf("%d: unexpected error: %v", i, err)
			server.Close()
			continue
		}

		if len(testCase.Stdout) > 0 {
			if e, a := testCase.Stdout, localOut; e != a.String() {
				t.Errorf("%d: expected stdout data '%s', got '%s'", i, e, a)
			}
		}

		if testCase.Stderr != "" {
			if e, a := testCase.Stderr, localErr; e != a.String() {
				t.Errorf("%d: expected stderr data '%s', got '%s'", i, e, a)
			}
		}

		server.Close()
	}
}

type fakeUpgrader struct {
	req           *http.Request
	resp          *http.Response
	conn          httpstream.Connection
	err, connErr  error
	checkResponse bool

	t *testing.T
}

func (u *fakeUpgrader) RoundTrip(req *http.Request) (*http.Response, error) {
	u.req = req
	return u.resp, u.err
}

func (u *fakeUpgrader) NewConnection(resp *http.Response) (httpstream.Connection, error) {
	if u.checkResponse && u.resp != resp {
		u.t.Errorf("response objects passed did not match: %#v", resp)
	}
	return u.conn, u.connErr
}

type fakeConnection struct {
	httpstream.Connection
}

// Dial is the common functionality between any stream based upgrader, regardless of protocol.
// This method ensures that someone can use a generic stream executor without being dependent
// on the core Kube client config behavior.
func TestDial(t *testing.T) {
	upgrader := &fakeUpgrader{
		t:             t,
		checkResponse: true,
		conn:          &fakeConnection{},
		resp: &http.Response{
			StatusCode: http.StatusSwitchingProtocols,
			Body:       ioutil.NopCloser(&bytes.Buffer{}),
		},
	}
	var called bool
	testFn := func(rt http.RoundTripper) http.RoundTripper {
		if rt != upgrader {
			t.Fatalf("unexpected round tripper: %#v", rt)
		}
		called = true
		return rt
	}
	exec, err := NewStreamExecutor(upgrader, testFn, "POST", &url.URL{Host: "something.com", Scheme: "https"})
	if err != nil {
		t.Fatal(err)
	}
	conn, protocol, err := exec.Dial([]string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if conn != upgrader.conn {
		t.Errorf("unexpected connection: %#v", conn)
	}
	if !called {
		t.Errorf("wrapper not called")
	}
	_ = protocol
}
