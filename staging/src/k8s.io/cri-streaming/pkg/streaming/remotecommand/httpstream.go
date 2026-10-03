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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/streaming/pkg/httpstream"
	"k8s.io/streaming/pkg/httpstream/spdy"
	"k8s.io/streaming/pkg/httpstream/wsstream"
	"k8s.io/streaming/pkg/runtime"

	"k8s.io/klog/v2"
)

// Options contains details about which streams are required for
// remote command execution.
type Options struct {
	Stdin  bool
	Stdout bool
	Stderr bool
	TTY    bool
}

// NewOptions creates a new Options from the Request.
func NewOptions(req *http.Request) (*Options, error) {
	tty := req.FormValue(ExecTTYParam) == "1"
	stdin := req.FormValue(ExecStdinParam) == "1"
	stdout := req.FormValue(ExecStdoutParam) == "1"
	stderr := req.FormValue(ExecStderrParam) == "1"
	if tty && stderr {
		// TODO: make this an error before we reach this method
		klog.V(4).InfoS("Access to exec with tty and stderr is not supported, bypassing stderr")
		stderr = false
	}

	if !stdin && !stdout && !stderr {
		return nil, fmt.Errorf("you must specify at least 1 of stdin, stdout, stderr")
	}

	return &Options{
		Stdin:  stdin,
		Stdout: stdout,
		Stderr: stderr,
		TTY:    tty,
	}, nil
}

// connection is the transport connection of a session: an SPDY
// httpstream.Connection or a WebSocket wsstream.Conn.
type connection interface {
	io.Closer
	// CloseChan returns a channel that is closed once the connection has been
	// closed by either side.
	CloseChan() <-chan bool
}

// connectionContext contains the connection and streams used when
// forwarding an attach or execute session into a container.
type connectionContext struct {
	conn         connection
	stdinStream  io.ReadCloser
	stdoutStream io.WriteCloser
	stderrStream io.WriteCloser
	errorStream  io.WriteCloser
	writeStatus  func(status *streamStatusError) error
	resizeStream io.ReadCloser
	resizeChan   chan TerminalSize
	tty          bool
	// stopInput discards any further input from the client.
	stopInput func()
	// endOutput tells the client that nothing more will be written.
	endOutput func()
}

// finishTimeout bounds how long the server waits for the client to close the
// connection once the command has finished. The output still to be delivered
// at that point is at most what the socket buffers on the way to the client
// hold, tens of megabytes at the outside, which a client reading at 100 KB/s
// drains in a few minutes. Only a client that neither reads nor closes for
// this long is cut off.
var finishTimeout = 15 * time.Minute

// finish reports the status of the command to the client and waits for the
// client to close the connection.
//
// The connection must not be closed by the server while output may still be
// in flight. The client keeps sending frames (pings, stream control) while it
// reads, and data arriving on a socket the server has already closed makes
// the kernel reset the connection and discard whatever is left in its send
// buffer. A client that reads more slowly than the command wrote therefore
// received a truncated stream, and since the status was discarded with it,
// no error either. So the server only signals the end of the session, by
// closing the streams the client reads from (SPDY) or by sending a close
// frame (WebSocket), and closes the connection once the client has: client-go
// closes it after it has read everything, and a WebSocket client's reply to
// the close frame ends the read loop as well. The wait ends early if the
// request context is done, and after finishTimeout at the latest. An idle
// timeout shorter than that, if one is configured, ends it sooner for a client
// that goes quiet; the default of four hours does not.
//
// The frames written here (status, stream close and reset) and the ones the
// deferred connection close sends afterwards go to a socket the peer may have
// stopped reading, so finish first sets a write deadline of finishTimeout on
// the connection. On SPDY that bounds those writes like the wait. On
// WebSocket it does not yet: wsstream moves the connection's deadline to the
// idle timeout on every frame it writes, so there the idle timeout bounds
// them, as it bounded the status write before. The command's own writes
// before this point are bounded by the idle timeout only, as before.
func (ctx *connectionContext) finish(reqCtx context.Context, status *streamStatusError) {
	deadline := time.Now().Add(finishTimeout)
	ctx.setWriteDeadline(deadline)
	ctx.stopInput()
	_ = ctx.writeStatus(status)
	ctx.endOutput()
	timer := time.NewTimer(time.Until(deadline))
	defer timer.Stop()
	select {
	case <-ctx.conn.CloseChan():
	case <-reqCtx.Done():
	case <-timer.C:
	}
}

// setWriteDeadline sets a write deadline on the connection, best effort: the
// WebSocket connection takes a timeout directly, a SPDY stream sets the
// deadline on the connection it belongs to.
func (ctx *connectionContext) setWriteDeadline(t time.Time) {
	type timeoutSetter interface {
		SetWriteDeadline(time.Duration)
	}
	type deadlineSetter interface {
		SetWriteDeadline(time.Time) error
	}
	if d, ok := ctx.conn.(timeoutSetter); ok {
		d.SetWriteDeadline(time.Until(t))
		return
	}
	for _, s := range []io.WriteCloser{ctx.errorStream, ctx.stdoutStream, ctx.stderrStream} {
		if sr, ok := s.(streamAndReply); ok {
			s = sr.Stream
		}
		if d, ok := s.(deadlineSetter); ok {
			_ = d.SetWriteDeadline(t)
			return
		}
	}
}

// streamAndReply holds both a Stream and a channel that is closed when the stream's reply frame is
// enqueued. Consumers can wait for replySent to be closed prior to proceeding, to ensure that the
// replyFrame is enqueued before the connection's goaway frame is sent (e.g. if a stream was
// received and right after, the connection gets closed).
type streamAndReply struct {
	httpstream.Stream
	replySent <-chan struct{}
}

// waitStreamReply waits until either replySent or stop is closed. If replySent is closed, it sends
// an empty struct to the notify channel.
func waitStreamReply(replySent <-chan struct{}, notify chan<- struct{}, stop <-chan struct{}) {
	select {
	case <-replySent:
		notify <- struct{}{}
	case <-stop:
	}
}

func createStreams(req *http.Request, w http.ResponseWriter, opts *Options, supportedStreamProtocols []string, idleTimeout, streamCreationTimeout time.Duration) (*connectionContext, bool) {
	var ctx *connectionContext
	var ok bool
	if wsstream.IsWebSocketRequest(req) {
		ctx, ok = createWebSocketStreams(req, w, opts, idleTimeout)
	} else {
		ctx, ok = createHTTPStreamStreams(req, w, opts, supportedStreamProtocols, idleTimeout, streamCreationTimeout)
	}
	if !ok {
		return nil, false
	}

	if ctx.resizeStream != nil {
		ctx.resizeChan = make(chan TerminalSize)
		go handleResizeEvents(req.Context(), ctx.resizeStream, ctx.resizeChan)
	}

	return ctx, true
}

func createHTTPStreamStreams(req *http.Request, w http.ResponseWriter, opts *Options, supportedStreamProtocols []string, idleTimeout, streamCreationTimeout time.Duration) (*connectionContext, bool) {
	protocol, err := httpstream.Handshake(req, w, supportedStreamProtocols)
	if err != nil {
		// Handshake writes the error to the client
		return nil, false
	}

	streamCh := make(chan streamAndReply)

	upgrader := spdy.NewResponseUpgrader()
	conn := upgrader.UpgradeResponse(w, req, func(stream httpstream.Stream, replySent <-chan struct{}) error {
		streamCh <- streamAndReply{Stream: stream, replySent: replySent}
		return nil
	})
	// from this point on, we can no longer call methods on response
	if conn == nil {
		// The upgrader is responsible for notifying the client of any errors that
		// occurred during upgrading. All we can do is return here at this point
		// if we weren't successful in upgrading.
		return nil, false
	}

	conn.SetIdleTimeout(idleTimeout)

	var handler protocolHandler
	switch protocol {
	case StreamProtocolV4Name:
		handler = &v4ProtocolHandler{}
	case StreamProtocolV3Name:
		handler = &v3ProtocolHandler{}
	case StreamProtocolV2Name:
		handler = &v2ProtocolHandler{}
	case "":
		klog.V(4).InfoS("Client did not request protocol negotiation. Falling back", "protocol", StreamProtocolV1Name)
		fallthrough
	case StreamProtocolV1Name:
		handler = &v1ProtocolHandler{}
	}

	// count the streams client asked for, starting with 1
	expectedStreams := 1
	if opts.Stdin {
		expectedStreams++
	}
	if opts.Stdout {
		expectedStreams++
	}
	if opts.Stderr {
		expectedStreams++
	}
	if opts.TTY && handler.supportsTerminalResizing() {
		expectedStreams++
	}

	expired := time.NewTimer(streamCreationTimeout)
	defer expired.Stop()

	ctx, err := handler.waitForStreams(streamCh, expectedStreams, expired.C)
	if err != nil {
		runtime.HandleError(err)
		return nil, false
	}

	ctx.conn = conn
	ctx.tty = opts.TTY
	ctx.stopInput = func() {
		// A reset, unlike a close, also drops what the client sends on the
		// stream from now on instead of queueing it.
		for _, s := range []io.ReadCloser{ctx.stdinStream, ctx.resizeStream} {
			if stream, ok := s.(httpstream.Stream); ok {
				_ = stream.Reset()
			}
		}
	}
	ctx.endOutput = func() {
		// Closing a stream sends its FIN after everything written to it, so the
		// client reads EOF once it has consumed all the output.
		for _, s := range []io.WriteCloser{ctx.stdoutStream, ctx.stderrStream, ctx.errorStream} {
			if s != nil {
				_ = s.Close()
			}
		}
	}

	return ctx, true
}

type protocolHandler interface {
	// waitForStreams waits for the expected streams or a timeout, returning a
	// remoteCommandContext if all the streams were received, or an error if not.
	waitForStreams(streams <-chan streamAndReply, expectedStreams int, expired <-chan time.Time) (*connectionContext, error)
	// supportsTerminalResizing returns true if the protocol handler supports terminal resizing
	supportsTerminalResizing() bool
}

// v4ProtocolHandler implements the V4 protocol version for streaming command execution. It only differs
// in from v3 in the error stream format using a json-marshaled status object which carries
// the process' exit code.
type v4ProtocolHandler struct{}

func (*v4ProtocolHandler) waitForStreams(streams <-chan streamAndReply, expectedStreams int, expired <-chan time.Time) (*connectionContext, error) {
	ctx := &connectionContext{}
	receivedStreams := 0
	replyChan := make(chan struct{})
	stop := make(chan struct{})
	defer close(stop)
WaitForStreams:
	for {
		select {
		case stream := <-streams:
			streamType := stream.Headers().Get(StreamType)
			switch streamType {
			case StreamTypeError:
				ctx.errorStream = stream
				ctx.writeStatus = v4WriteStatusFunc(stream) // write json errors
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStdin:
				ctx.stdinStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStdout:
				ctx.stdoutStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStderr:
				ctx.stderrStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeResize:
				ctx.resizeStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			default:
				runtime.HandleError(fmt.Errorf("unexpected stream type: %q", streamType))
			}
		case <-replyChan:
			receivedStreams++
			if receivedStreams == expectedStreams {
				break WaitForStreams
			}
		case <-expired:
			// TODO find a way to return the error to the user. Maybe use a separate
			// stream to report errors?
			return nil, errors.New("timed out waiting for client to create streams")
		}
	}

	return ctx, nil
}

// supportsTerminalResizing returns true because v4ProtocolHandler supports it
func (*v4ProtocolHandler) supportsTerminalResizing() bool { return true }

// v3ProtocolHandler implements the V3 protocol version for streaming command execution.
type v3ProtocolHandler struct{}

func (*v3ProtocolHandler) waitForStreams(streams <-chan streamAndReply, expectedStreams int, expired <-chan time.Time) (*connectionContext, error) {
	ctx := &connectionContext{}
	receivedStreams := 0
	replyChan := make(chan struct{})
	stop := make(chan struct{})
	defer close(stop)
WaitForStreams:
	for {
		select {
		case stream := <-streams:
			streamType := stream.Headers().Get(StreamType)
			switch streamType {
			case StreamTypeError:
				ctx.errorStream = stream
				ctx.writeStatus = v1WriteStatusFunc(stream)
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStdin:
				ctx.stdinStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStdout:
				ctx.stdoutStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStderr:
				ctx.stderrStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeResize:
				ctx.resizeStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			default:
				runtime.HandleError(fmt.Errorf("unexpected stream type: %q", streamType))
			}
		case <-replyChan:
			receivedStreams++
			if receivedStreams == expectedStreams {
				break WaitForStreams
			}
		case <-expired:
			// TODO find a way to return the error to the user. Maybe use a separate
			// stream to report errors?
			return nil, errors.New("timed out waiting for client to create streams")
		}
	}

	return ctx, nil
}

// supportsTerminalResizing returns true because v3ProtocolHandler supports it
func (*v3ProtocolHandler) supportsTerminalResizing() bool { return true }

// v2ProtocolHandler implements the V2 protocol version for streaming command execution.
type v2ProtocolHandler struct{}

func (*v2ProtocolHandler) waitForStreams(streams <-chan streamAndReply, expectedStreams int, expired <-chan time.Time) (*connectionContext, error) {
	ctx := &connectionContext{}
	receivedStreams := 0
	replyChan := make(chan struct{})
	stop := make(chan struct{})
	defer close(stop)
WaitForStreams:
	for {
		select {
		case stream := <-streams:
			streamType := stream.Headers().Get(StreamType)
			switch streamType {
			case StreamTypeError:
				ctx.errorStream = stream
				ctx.writeStatus = v1WriteStatusFunc(stream)
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStdin:
				ctx.stdinStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStdout:
				ctx.stdoutStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStderr:
				ctx.stderrStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			default:
				runtime.HandleError(fmt.Errorf("unexpected stream type: %q", streamType))
			}
		case <-replyChan:
			receivedStreams++
			if receivedStreams == expectedStreams {
				break WaitForStreams
			}
		case <-expired:
			// TODO find a way to return the error to the user. Maybe use a separate
			// stream to report errors?
			return nil, errors.New("timed out waiting for client to create streams")
		}
	}

	return ctx, nil
}

// supportsTerminalResizing returns false because v2ProtocolHandler doesn't support it.
func (*v2ProtocolHandler) supportsTerminalResizing() bool { return false }

// v1ProtocolHandler implements the V1 protocol version for streaming command execution.
type v1ProtocolHandler struct{}

func (*v1ProtocolHandler) waitForStreams(streams <-chan streamAndReply, expectedStreams int, expired <-chan time.Time) (*connectionContext, error) {
	ctx := &connectionContext{}
	receivedStreams := 0
	replyChan := make(chan struct{})
	stop := make(chan struct{})
	defer close(stop)
WaitForStreams:
	for {
		select {
		case stream := <-streams:
			streamType := stream.Headers().Get(StreamType)
			switch streamType {
			case StreamTypeError:
				ctx.errorStream = stream
				ctx.writeStatus = v1WriteStatusFunc(stream)

				// This defer statement shouldn't be here, but due to previous refactoring, it ended up in
				// here. This is what 1.0.x kubelets do, so we're retaining that behavior. This is fixed in
				// the v2ProtocolHandler.
				defer stream.Reset()

				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStdin:
				ctx.stdinStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStdout:
				ctx.stdoutStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			case StreamTypeStderr:
				ctx.stderrStream = stream
				go waitStreamReply(stream.replySent, replyChan, stop)
			default:
				runtime.HandleError(fmt.Errorf("unexpected stream type: %q", streamType))
			}
		case <-replyChan:
			receivedStreams++
			if receivedStreams == expectedStreams {
				break WaitForStreams
			}
		case <-expired:
			// TODO find a way to return the error to the user. Maybe use a separate
			// stream to report errors?
			return nil, errors.New("timed out waiting for client to create streams")
		}
	}

	if ctx.stdinStream != nil {
		ctx.stdinStream.Close()
	}

	return ctx, nil
}

// supportsTerminalResizing returns false because v1ProtocolHandler doesn't support it.
func (*v1ProtocolHandler) supportsTerminalResizing() bool { return false }

func handleResizeEvents(reqctx context.Context, stream io.Reader, channel chan<- TerminalSize) {
	defer runtime.HandleCrash()
	defer close(channel)

	decoder := json.NewDecoder(stream)
	for {
		size := TerminalSize{}
		if err := decoder.Decode(&size); err != nil {
			break
		}
		select {
		case channel <- size:
		case <-reqctx.Done():
			// To prevent go routine leak.
			return
		}
	}
}

func v1WriteStatusFunc(stream io.Writer) func(status *streamStatusError) error {
	return func(status *streamStatusError) error {
		if status.status().Status == statusSuccess {
			return nil // send error messages
		}
		_, err := stream.Write([]byte(status.Error()))
		return err
	}
}

// v4WriteStatusFunc returns a WriteStatusFunc that marshals a status object
// as json in the error channel.
func v4WriteStatusFunc(stream io.Writer) func(status *streamStatusError) error {
	return func(status *streamStatusError) error {
		bs, err := json.Marshal(status.status())
		if err != nil {
			return err
		}
		_, err = stream.Write(bs)
		return err
	}
}
