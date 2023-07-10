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
	"bytes"
	"context"
	"crypto/rand"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	rctesting "k8s.io/client-go/tools/remotecommand/testing"
)

type AttachFunc func(in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan TerminalSize) error

type fakeEmptyDataPty struct {
}

func (s *fakeEmptyDataPty) Read(p []byte) (int, error) {
	return len(p), nil
}

func (s *fakeEmptyDataPty) Write(p []byte) (int, error) {
	return len(p), nil
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
		io.Copy(io.Discard, stdin)
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
		timeout     time.Duration
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
		{
			timeout: 500 * time.Millisecond,
			name:    "timeoutTest",
			options: StreamOptions{
				Stdin:  &fakeMassiveDataPty{},
				Stderr: &fakeMassiveDataPty{},
			},
			expectError: context.DeadlineExceeded.Error(),
			attacher:    fakeMassiveDataAttacher,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := newTestHTTPServer(test.attacher, &test.options)
			defer server.Close()

			ctx, cancel := context.Background(), func() {}
			if test.timeout > 0 {
				ctx, cancel = context.WithTimeout(ctx, test.timeout)
			}
			defer cancel()

			err := attach2Server(ctx, server.URL, test.options)

			gotError := ""
			if err != nil {
				gotError = err.Error()
			}
			if test.expectError != gotError {
				t.Errorf("%s: expected [%v], got [%v]", test.name, test.expectError, gotError)
			}
		})
	}
}

func convertStreamOptions(options *StreamOptions) rctesting.Options {
	testOptions := rctesting.Options{}
	if options.Stdin != nil {
		testOptions.Stdin = true
	}
	if options.Stdout != nil {
		testOptions.Stdout = true
	}
	if options.Stderr != nil {
		testOptions.Stderr = true
	}
	return testOptions
}

func newTestHTTPServer(f AttachFunc, options *StreamOptions) *httptest.Server {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		ctx, err := rctesting.CreateHTTPStreams(writer, request, convertStreamOptions(options))
		if err != nil {
			return
		}
		defer ctx.Conn.Close()

		// handle input output
		err = f(ctx.StdinStream, ctx.StdoutStream, ctx.StderrStream, false, nil)
		if err != nil {
			ctx.WriteStatus(apierrors.NewInternalError(err))
		} else {
			ctx.WriteStatus(&apierrors.StatusError{ErrStatus: metav1.Status{
				Status: metav1.StatusSuccess,
			}})
		}
	}))
	return server
}

func attach2Server(ctx context.Context, rawURL string, options StreamOptions) error {
	uri, _ := url.Parse(rawURL)
	exec, err := NewSPDYExecutor(&rest.Config{Host: uri.Host}, "POST", uri)
	if err != nil {
		return err
	}

	e := make(chan error, 1)
	go func(e chan error) {
		e <- exec.StreamWithContext(ctx, options)
	}(e)
	select {
	case err := <-e:
		return err
	case <-time.After(wait.ForeverTestTimeout):
		return errors.New("execute timeout")
	}
}

// writeDetector provides a helper method to block until the underlying writer written.
type writeDetector struct {
	written chan bool
	closed  bool
	io.Writer
}

func newWriterDetector(w io.Writer) *writeDetector {
	return &writeDetector{
		written: make(chan bool),
		Writer:  w,
	}
}

func (w *writeDetector) BlockUntilWritten() {
	<-w.written
}

func (w *writeDetector) Write(p []byte) (n int, err error) {
	if !w.closed {
		close(w.written)
		w.closed = true
	}
	return w.Writer.Write(p)
}

// `Executor.StreamWithContext` starts a goroutine in the background to do the streaming
// and expects the deferred close of the connection leads to the exit of the goroutine on cancellation.
// This test verifies that works.
func TestStreamExitsAfterConnectionIsClosed(t *testing.T) {
	writeDetector := newWriterDetector(&fakeEmptyDataPty{})
	options := StreamOptions{
		Stdin:  &fakeEmptyDataPty{},
		Stdout: writeDetector,
	}
	server := newTestHTTPServer(fakeMassiveDataAttacher, &options)

	ctx, cancelFn := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancelFn()

	uri, _ := url.Parse(server.URL)
	exec, err := NewSPDYExecutor(&rest.Config{Host: uri.Host}, "POST", uri)
	if err != nil {
		t.Fatal(err)
	}
	streamExec := exec.(*spdyStreamExecutor)

	conn, streamer, err := streamExec.newConnectionAndStream(ctx, options)
	if err != nil {
		t.Fatal(err)
	}

	errorChan := make(chan error)
	go func() {
		errorChan <- streamer.stream(conn)
	}()

	// Wait until stream goroutine starts.
	writeDetector.BlockUntilWritten()

	// Close the connection
	conn.Close()

	select {
	case <-time.After(1 * time.Second):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case <-errorChan:
		return
	}
}

func TestStreamRandomData(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx, err := rctesting.CreateHTTPStreams(w, req, rctesting.Options{
			Stdin:  true,
			Stdout: true,
		})
		if err != nil {
			t.Errorf("error on createHTTPStreams: %v", err)
			return
		}
		defer ctx.Conn.Close()

		io.Copy(ctx.StdoutStream, ctx.StdinStream)
	}))

	defer server.Close()

	uri, _ := url.Parse(server.URL)
	exec, err := NewSPDYExecutor(&rest.Config{Host: uri.Host}, "POST", uri)
	if err != nil {
		t.Fatal(err)
	}

	randomData := make([]byte, 1024*1024)
	if _, err := rand.Read(randomData); err != nil {
		t.Errorf("unexpected error reading random data: %v", err)
	}
	var stdout bytes.Buffer
	options := &StreamOptions{
		Stdin:  bytes.NewReader(randomData),
		Stdout: &stdout,
	}
	errorChan := make(chan error)
	go func() {
		errorChan <- exec.StreamWithContext(context.Background(), *options)
	}()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("expect stream to be closed after connection is closed.")
	case err := <-errorChan:
		if err != nil {
			t.Errorf("unexpected error")
		}
	}

	data, err := ioutil.ReadAll(bytes.NewReader(stdout.Bytes()))
	if err != nil {
		t.Errorf("error reading the stream: %v", err)
		return
	}
	if !bytes.Equal(randomData, data) {
		t.Errorf("unexpected data received: %d sent: %d", len(data), len(randomData))
	}

}
