/*
Copyright 2020 The Kubernetes Authors.

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
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	remotecommandconsts "k8s.io/apimachinery/pkg/util/remotecommand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

type AttachFunc func(in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan TerminalSize) error
type streamContext struct {
	conn         io.Closer
	stdinStream  io.ReadCloser
	stdoutStream io.WriteCloser
	stderrStream io.WriteCloser
	writeStatus  func(status *apierrors.StatusError) error
}

type streamAndReply struct {
	httpstream.Stream
	replySent <-chan struct{}
}

type fakeMassiveDataPty struct{}

func (s *fakeMassiveDataPty) Read(p []byte) (int, error) {
	time.Sleep(time.Duration(1) * time.Second)
	return copy(p, []byte{}), errors.New("client crashed after 1 second")
}

func (s *fakeMassiveDataPty) Write(p []byte) (int, error) {
	time.Sleep(time.Duration(1) * time.Second)
	return len(p), errors.New("return err")
}

func fakeMassiveDataAttacher(stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan TerminalSize) error {

	copyDone := make(chan struct{}, 3)

	if stdin == nil {
		return errors.New("stdin is requested") // we need stdin to notice the conn break
	}

	go func() {
		io.Copy(ioutil.Discard, stdin)
		copyDone <- struct{}{}
	}()

	go func() {
		if stdout == nil {
			return
		}
		copyDone <- writeMassiveData(stdout)
	}()

	go func() {
		if stderr == nil {
			return
		}
		copyDone <- writeMassiveData(stderr)
	}()

	select {
	case <-copyDone:
		return nil
	}
}

func writeMassiveData(stdStream io.Writer) struct{} { // write to stdin or stdout
	for {
		_, err := io.Copy(stdStream, strings.NewReader("something"))
		if err != nil && err.Error() != "EOF" {
			break
		}
	}
	return struct{}{}
}

func TestSPDYExecutorStream(t *testing.T) {
	tests := []struct {
		name        string
		options     StreamOptions
		expectError string
		attacher    AttachFunc
	}{
		{
			name: "stdoutBlockTest",
			options: StreamOptions{
				Stdin:  &fakeMassiveDataPty{},
				Stdout: &fakeMassiveDataPty{},
			},
			expectError: "",
			attacher:    fakeMassiveDataAttacher,
		},
		{
			name: "stderrBlockTest",
			options: StreamOptions{
				Stdin:  &fakeMassiveDataPty{},
				Stderr: &fakeMassiveDataPty{},
			},
			expectError: "",
			attacher:    fakeMassiveDataAttacher,
		},
	}

	for _, test := range tests {
		server := newTestHTTPServer(test.attacher, &test.options)

		err := attach2Server(server.URL, test.options)
		gotError := ""
		if err != nil {
			gotError = err.Error()
		}
		if test.expectError != gotError {
			t.Errorf("%s: expected [%v], got [%v]", test.name, test.expectError, gotError)
		}

		server.Close()
	}

}

func newTestHTTPServer(f AttachFunc, options *StreamOptions) *httptest.Server {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		ctx, err := createHTTPStreams(writer, request, options)
		if err != nil {
			return
		}
		defer ctx.conn.Close()

		// handle input output
		err = f(ctx.stdinStream, ctx.stdoutStream, ctx.stderrStream, false, nil)
		if err != nil {
			ctx.writeStatus(apierrors.NewInternalError(err))
		} else {
			ctx.writeStatus(&apierrors.StatusError{ErrStatus: metav1.Status{
				Status: metav1.StatusSuccess,
			}})
		}
	}))
	return server
}

func attach2Server(rawURL string, options StreamOptions) error {
	uri, _ := url.Parse(rawURL)
	exec, err := NewSPDYExecutor(&rest.Config{Host: uri.Host}, "POST", uri)
	if err != nil {
		return err
	}

	e := make(chan error)
	go func(e chan error) {
		e <- exec.Stream(options)
	}(e)
	select {
	case err := <-e:
		return err
	case <-time.After(wait.ForeverTestTimeout):
		return errors.New("execute timeout")
	}
}

// simplify createHttpStreams , only support StreamProtocolV4Name
func createHTTPStreams(w http.ResponseWriter, req *http.Request, opts *StreamOptions) (*streamContext, error) {
	_, err := httpstream.Handshake(req, w, []string{remotecommandconsts.StreamProtocolV4Name})
	if err != nil {
		return nil, err
	}

	upgrader := spdy.NewResponseUpgrader()
	streamCh := make(chan streamAndReply)
	conn := upgrader.UpgradeResponse(w, req, func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streamCh <- streamAndReply{Stream: stream, replySent: replySent}
		return nil
	})
	ctx := &streamContext{
		conn: conn,
	}

	// wait for stream
	replyChan := make(chan struct{}, 4)
	defer close(replyChan)
	receivedStreams := 0
	expectedStreams := 1
	if opts.Stdout != nil {
		expectedStreams++
	}
	if opts.Stdin != nil {
		expectedStreams++
	}
	if opts.Stderr != nil {
		expectedStreams++
	}
WaitForStreams:
	for {
		select {
		case stream := <-streamCh:
			streamType := stream.Headers().Get(v1.StreamType)
			switch streamType {
			case v1.StreamTypeError:
				replyChan <- struct{}{}
				ctx.writeStatus = v4WriteStatusFunc(stream)
			case v1.StreamTypeStdout:
				replyChan <- struct{}{}
				ctx.stdoutStream = stream
			case v1.StreamTypeStdin:
				replyChan <- struct{}{}
				ctx.stdinStream = stream
			case v1.StreamTypeStderr:
				replyChan <- struct{}{}
				ctx.stderrStream = stream
			default:
				// add other stream ...
				return nil, errors.New("unimplemented stream type")
			}
		case <-replyChan:
			receivedStreams++
			if receivedStreams == expectedStreams {
				break WaitForStreams
			}
		}
	}

	return ctx, nil
}

func v4WriteStatusFunc(stream io.Writer) func(status *apierrors.StatusError) error {
	return func(status *apierrors.StatusError) error {
		bs, err := json.Marshal(status.Status())
		if err != nil {
			return err
		}
		_, err = stream.Write(bs)
		return err
	}
}
