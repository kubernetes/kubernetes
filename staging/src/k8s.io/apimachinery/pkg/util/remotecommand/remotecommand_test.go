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

package remotecommand

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
)

type attachFunc func(in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan TerminalSize) error
type streamAndReply struct {
	httpstream.Stream
	replySent <-chan struct{}
}

type fakeEmptyDataPty struct {
}

func (s *fakeEmptyDataPty) Read(p []byte) (int, error) {
	return len(p), nil
}

func (s *fakeEmptyDataPty) Write(p []byte) (int, error) {
	return len(p), nil
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

func newTestHTTPServer(f attachFunc, options *StreamOptions) *httptest.Server {
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

type StreamContext struct {
	conn         io.Closer
	stdinStream  io.ReadCloser
	stdoutStream io.WriteCloser
	stderrStream io.WriteCloser
	writeStatus  func(status *apierrors.StatusError) error
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

// simplify createHttpStreams , only support StreamProtocolV4Name
func createHTTPStreams(w http.ResponseWriter, req *http.Request, opts *StreamOptions) (*StreamContext, error) {
	_, err := httpstream.Handshake(req, w, []string{StreamProtocolV4Name})
	if err != nil {
		return nil, err
	}

	upgrader := spdy.NewResponseUpgrader()
	streamCh := make(chan streamAndReply)
	conn := upgrader.UpgradeResponse(w, req, func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streamCh <- streamAndReply{Stream: stream, replySent: replySent}
		return nil
	})
	ctx := &StreamContext{
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
			streamType := stream.Headers().Get(httpstream.StreamType)
			switch streamType {
			case httpstream.StreamTypeError:
				replyChan <- struct{}{}
				ctx.writeStatus = v4WriteStatusFunc(stream)
			case httpstream.StreamTypeStdout:
				replyChan <- struct{}{}
				ctx.stdoutStream = stream
			case httpstream.StreamTypeStdin:
				replyChan <- struct{}{}
				ctx.stdinStream = stream
			case httpstream.StreamTypeStderr:
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

	rt := spdy.NewRoundTripperWithConfig(spdy.RoundTripperConfig{
		TLS: &tls.Config{
			InsecureSkipVerify: true,
		},
	})
	uri, _ := url.Parse(server.URL)
	exec, err := NewSPDYExecutorForTransports(rt, rt, "POST", uri)
	if err != nil {
		t.Fatal(err)
	}
	streamExec := exec.(*streamExecutor)

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
