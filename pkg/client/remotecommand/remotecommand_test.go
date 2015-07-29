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
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/httpstream"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/httpstream/spdy"
)

func fakeExecServer(t *testing.T, i int, stdinData, stdoutData, stderrData, errorData string, tty bool) http.HandlerFunc {
	// error + stdin + stdout
	expectedStreams := 3
	if !tty {
		// stderr
		expectedStreams++
	}

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		streamCh := make(chan httpstream.Stream)

		upgrader := spdy.NewResponseUpgrader()
		conn := upgrader.UpgradeResponse(w, req, func(stream httpstream.Stream) error {
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
					stdinStream.Close()
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

				defer stream.Reset()

				if receivedStreams == expectedStreams {
					break WaitForStreams
				}
			}
		}

		if len(errorData) > 0 {
			fmt.Fprint(errorStream, errorData)
			errorStream.Close()
		}

		if len(stdoutData) > 0 {
			fmt.Fprint(stdoutStream, stdoutData)
			stdoutStream.Close()
		}
		if len(stderrData) > 0 {
			fmt.Fprint(stderrStream, stderrData)
			stderrStream.Close()
		}
		if len(stdinData) > 0 {
			data, err := ioutil.ReadAll(stdinStream)
			if err != nil {
				t.Errorf("%d: error reading stdin stream: %v", i, err)
			}
			if e, a := stdinData, string(data); e != a {
				t.Errorf("%d: stdin: expected %q, got %q", i, e, a)
			}
		}
	})
}

func TestRequestExecuteRemoteCommand(t *testing.T) {
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

		server := httptest.NewServer(fakeExecServer(t, i, testCase.Stdin, testCase.Stdout, testCase.Stderr, testCase.Error, testCase.Tty))

		url, _ := url.ParseRequestURI(server.URL)
		c := client.NewRESTClient(url, "x", nil, -1, -1)
		req := c.Post().Resource("testing")

		conf := &client.Config{
			Host: server.URL,
		}
		e := New(req, conf, []string{"ls", "/"}, strings.NewReader(testCase.Stdin), localOut, localErr, testCase.Tty)
		//e.upgrader = testCase.Upgrader
		err := e.Execute()
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

		server := httptest.NewServer(fakeExecServer(t, i, testCase.Stdin, testCase.Stdout, testCase.Stderr, testCase.Error, testCase.Tty))

		url, _ := url.ParseRequestURI(server.URL)
		c := client.NewRESTClient(url, "x", nil, -1, -1)
		req := c.Post().Resource("testing")

		conf := &client.Config{
			Host: server.URL,
		}
		e := NewAttach(req, conf, strings.NewReader(testCase.Stdin), localOut, localErr, testCase.Tty)
		//e.upgrader = testCase.Upgrader
		err := e.Execute()
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
