/*
 *
 * Copyright 2014 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package transport defines and implements message oriented communication
// channel to complete various transactions (e.g., an RPC).  It is meant for
// grpc-internal usage and is not intended to be imported directly by users.
package transport

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/mem"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/tap"
)

const logLevel = 2

// recvMsg represents the received msg from the transport. All transport
// protocol specific info has been removed.
type recvMsg struct {
	buffer mem.Buffer
	// nil: received some data
	// io.EOF: stream is completed. data is nil.
	// other non-nil error: transport failure. data is nil.
	err error
}

// recvBuffer is an unbounded channel of recvMsg structs.
//
// Note: recvBuffer differs from buffer.Unbounded only in the fact that it
// holds a channel of recvMsg structs instead of objects implementing "item"
// interface. recvBuffer is written to much more often and using strict recvMsg
// structs helps avoid allocation in "recvBuffer.put"
type recvBuffer struct {
	c       chan recvMsg
	mu      sync.Mutex
	backlog []recvMsg
	err     error
}

func newRecvBuffer() *recvBuffer {
	b := &recvBuffer{
		c: make(chan recvMsg, 1),
	}
	return b
}

func (b *recvBuffer) put(r recvMsg) {
	b.mu.Lock()
	if b.err != nil {
		// drop the buffer on the floor. Since b.err is not nil, any subsequent reads
		// will always return an error, making this buffer inaccessible.
		r.buffer.Free()
		b.mu.Unlock()
		// An error had occurred earlier, don't accept more
		// data or errors.
		return
	}
	b.err = r.err
	if len(b.backlog) == 0 {
		select {
		case b.c <- r:
			b.mu.Unlock()
			return
		default:
		}
	}
	b.backlog = append(b.backlog, r)
	b.mu.Unlock()
}

func (b *recvBuffer) load() {
	b.mu.Lock()
	if len(b.backlog) > 0 {
		select {
		case b.c <- b.backlog[0]:
			b.backlog[0] = recvMsg{}
			b.backlog = b.backlog[1:]
		default:
		}
	}
	b.mu.Unlock()
}

// get returns the channel that receives a recvMsg in the buffer.
//
// Upon receipt of a recvMsg, the caller should call load to send another
// recvMsg onto the channel if there is any.
func (b *recvBuffer) get() <-chan recvMsg {
	return b.c
}

// recvBufferReader implements io.Reader interface to read the data from
// recvBuffer.
type recvBufferReader struct {
	closeStream func(error) // Closes the client transport stream with the given error and nil trailer metadata.
	ctx         context.Context
	ctxDone     <-chan struct{} // cache of ctx.Done() (for performance).
	recv        *recvBuffer
	last        mem.Buffer // Stores the remaining data in the previous calls.
	err         error
}

func (r *recvBufferReader) ReadMessageHeader(header []byte) (n int, err error) {
	if r.err != nil {
		return 0, r.err
	}
	if r.last != nil {
		n, r.last = mem.ReadUnsafe(header, r.last)
		return n, nil
	}
	if r.closeStream != nil {
		n, r.err = r.readMessageHeaderClient(header)
	} else {
		n, r.err = r.readMessageHeader(header)
	}
	return n, r.err
}

// Read reads the next n bytes from last. If last is drained, it tries to read
// additional data from recv. It blocks if there no additional data available in
// recv. If Read returns any non-nil error, it will continue to return that
// error.
func (r *recvBufferReader) Read(n int) (buf mem.Buffer, err error) {
	if r.err != nil {
		return nil, r.err
	}
	if r.last != nil {
		buf = r.last
		if r.last.Len() > n {
			buf, r.last = mem.SplitUnsafe(buf, n)
		} else {
			r.last = nil
		}
		return buf, nil
	}
	if r.closeStream != nil {
		buf, r.err = r.readClient(n)
	} else {
		buf, r.err = r.read(n)
	}
	return buf, r.err
}

func (r *recvBufferReader) readMessageHeader(header []byte) (n int, err error) {
	select {
	case <-r.ctxDone:
		return 0, ContextErr(r.ctx.Err())
	case m := <-r.recv.get():
		return r.readMessageHeaderAdditional(m, header)
	}
}

func (r *recvBufferReader) read(n int) (buf mem.Buffer, err error) {
	select {
	case <-r.ctxDone:
		return nil, ContextErr(r.ctx.Err())
	case m := <-r.recv.get():
		return r.readAdditional(m, n)
	}
}

func (r *recvBufferReader) readMessageHeaderClient(header []byte) (n int, err error) {
	// If the context is canceled, then closes the stream with nil metadata.
	// closeStream writes its error parameter to r.recv as a recvMsg.
	// r.readAdditional acts on that message and returns the necessary error.
	select {
	case <-r.ctxDone:
		// Note that this adds the ctx error to the end of recv buffer, and
		// reads from the head. This will delay the error until recv buffer is
		// empty, thus will delay ctx cancellation in Recv().
		//
		// It's done this way to fix a race between ctx cancel and trailer. The
		// race was, stream.Recv() may return ctx error if ctxDone wins the
		// race, but stream.Trailer() may return a non-nil md because the stream
		// was not marked as done when trailer is received. This closeStream
		// call will mark stream as done, thus fix the race.
		//
		// TODO: delaying ctx error seems like a unnecessary side effect. What
		// we really want is to mark the stream as done, and return ctx error
		// faster.
		r.closeStream(ContextErr(r.ctx.Err()))
		m := <-r.recv.get()
		return r.readMessageHeaderAdditional(m, header)
	case m := <-r.recv.get():
		return r.readMessageHeaderAdditional(m, header)
	}
}

func (r *recvBufferReader) readClient(n int) (buf mem.Buffer, err error) {
	// If the context is canceled, then closes the stream with nil metadata.
	// closeStream writes its error parameter to r.recv as a recvMsg.
	// r.readAdditional acts on that message and returns the necessary error.
	select {
	case <-r.ctxDone:
		// Note that this adds the ctx error to the end of recv buffer, and
		// reads from the head. This will delay the error until recv buffer is
		// empty, thus will delay ctx cancellation in Recv().
		//
		// It's done this way to fix a race between ctx cancel and trailer. The
		// race was, stream.Recv() may return ctx error if ctxDone wins the
		// race, but stream.Trailer() may return a non-nil md because the stream
		// was not marked as done when trailer is received. This closeStream
		// call will mark stream as done, thus fix the race.
		//
		// TODO: delaying ctx error seems like a unnecessary side effect. What
		// we really want is to mark the stream as done, and return ctx error
		// faster.
		r.closeStream(ContextErr(r.ctx.Err()))
		m := <-r.recv.get()
		return r.readAdditional(m, n)
	case m := <-r.recv.get():
		return r.readAdditional(m, n)
	}
}

func (r *recvBufferReader) readMessageHeaderAdditional(m recvMsg, header []byte) (n int, err error) {
	r.recv.load()
	if m.err != nil {
		if m.buffer != nil {
			m.buffer.Free()
		}
		return 0, m.err
	}

	n, r.last = mem.ReadUnsafe(header, m.buffer)

	return n, nil
}

func (r *recvBufferReader) readAdditional(m recvMsg, n int) (b mem.Buffer, err error) {
	r.recv.load()
	if m.err != nil {
		if m.buffer != nil {
			m.buffer.Free()
		}
		return nil, m.err
	}

	if m.buffer.Len() > n {
		m.buffer, r.last = mem.SplitUnsafe(m.buffer, n)
	}

	return m.buffer, nil
}

type streamState uint32

const (
	streamActive    streamState = iota
	streamWriteDone             // EndStream sent
	streamReadDone              // EndStream received
	streamDone                  // the entire stream is finished.
)

// Stream represents an RPC in the transport layer.
type Stream struct {
	id           uint32
	ctx          context.Context // the associated context of the stream
	method       string          // the associated RPC method of the stream
	recvCompress string
	sendCompress string
	buf          *recvBuffer
	trReader     *transportReader
	fc           *inFlow
	wq           *writeQuota

	// Callback to state application's intentions to read data. This
	// is used to adjust flow control, if needed.
	requestRead func(int)

	state streamState

	// contentSubtype is the content-subtype for requests.
	// this must be lowercase or the behavior is undefined.
	contentSubtype string

	trailer metadata.MD // the key-value map of trailer metadata.
}

func (s *Stream) swapState(st streamState) streamState {
	return streamState(atomic.SwapUint32((*uint32)(&s.state), uint32(st)))
}

func (s *Stream) compareAndSwapState(oldState, newState streamState) bool {
	return atomic.CompareAndSwapUint32((*uint32)(&s.state), uint32(oldState), uint32(newState))
}

func (s *Stream) getState() streamState {
	return streamState(atomic.LoadUint32((*uint32)(&s.state)))
}

// Trailer returns the cached trailer metadata. Note that if it is not called
// after the entire stream is done, it could return an empty MD.
// It can be safely read only after stream has ended that is either read
// or write have returned io.EOF.
func (s *Stream) Trailer() metadata.MD {
	return s.trailer.Copy()
}

// Context returns the context of the stream.
func (s *Stream) Context() context.Context {
	return s.ctx
}

// Method returns the method for the stream.
func (s *Stream) Method() string {
	return s.method
}

func (s *Stream) write(m recvMsg) {
	s.buf.put(m)
}

// ReadMessageHeader reads data into the provided header slice from the stream.
// It first checks if there was an error during a previous read operation and
// returns it if present. It then requests a read operation for the length of
// the header. It continues to read from the stream until the entire header
// slice is filled or an error occurs. If an `io.EOF` error is encountered with
// partially read data, it is converted to `io.ErrUnexpectedEOF` to indicate an
// unexpected end of the stream. The method returns any error encountered during
// the read process or nil if the header was successfully read.
func (s *Stream) ReadMessageHeader(header []byte) (err error) {
	// Don't request a read if there was an error earlier
	if er := s.trReader.er; er != nil {
		return er
	}
	s.requestRead(len(header))
	for len(header) != 0 {
		n, err := s.trReader.ReadMessageHeader(header)
		header = header[n:]
		if len(header) == 0 {
			err = nil
		}
		if err != nil {
			if n > 0 && err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return err
		}
	}
	return nil
}

// Read reads n bytes from the wire for this stream.
func (s *Stream) read(n int) (data mem.BufferSlice, err error) {
	// Don't request a read if there was an error earlier
	if er := s.trReader.er; er != nil {
		return nil, er
	}
	s.requestRead(n)
	for n != 0 {
		buf, err := s.trReader.Read(n)
		var bufLen int
		if buf != nil {
			bufLen = buf.Len()
		}
		n -= bufLen
		if n == 0 {
			err = nil
		}
		if err != nil {
			if bufLen > 0 && err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			data.Free()
			return nil, err
		}
		data = append(data, buf)
	}
	return data, nil
}

// transportReader reads all the data available for this Stream from the transport and
// passes them into the decoder, which converts them into a gRPC message stream.
// The error is io.EOF when the stream is done or another non-nil error if
// the stream broke.
type transportReader struct {
	reader *recvBufferReader
	// The handler to control the window update procedure for both this
	// particular stream and the associated transport.
	windowHandler func(int)
	er            error
}

func (t *transportReader) ReadMessageHeader(header []byte) (int, error) {
	n, err := t.reader.ReadMessageHeader(header)
	if err != nil {
		t.er = err
		return 0, err
	}
	t.windowHandler(n)
	return n, nil
}

func (t *transportReader) Read(n int) (mem.Buffer, error) {
	buf, err := t.reader.Read(n)
	if err != nil {
		t.er = err
		return buf, err
	}
	t.windowHandler(buf.Len())
	return buf, nil
}

// GoString is implemented by Stream so context.String() won't
// race when printing %#v.
func (s *Stream) GoString() string {
	return fmt.Sprintf("<stream: %p, %v>", s, s.method)
}

// state of transport
type transportState int

const (
	reachable transportState = iota
	closing
	draining
)

// ServerConfig consists of all the configurations to establish a server transport.
type ServerConfig struct {
	MaxStreams            uint32
	ConnectionTimeout     time.Duration
	Credentials           credentials.TransportCredentials
	InTapHandle           tap.ServerInHandle
	StatsHandlers         []stats.Handler
	KeepaliveParams       keepalive.ServerParameters
	KeepalivePolicy       keepalive.EnforcementPolicy
	InitialWindowSize     int32
	InitialConnWindowSize int32
	WriteBufferSize       int
	ReadBufferSize        int
	SharedWriteBuffer     bool
	ChannelzParent        *channelz.Server
	MaxHeaderListSize     *uint32
	HeaderTableSize       *uint32
	BufferPool            mem.BufferPool
}

// ConnectOptions covers all relevant options for communicating with the server.
type ConnectOptions struct {
	// UserAgent is the application user agent.
	UserAgent string
	// Dialer specifies how to dial a network address.
	Dialer func(context.Context, string) (net.Conn, error)
	// FailOnNonTempDialError specifies if gRPC fails on non-temporary dial errors.
	FailOnNonTempDialError bool
	// PerRPCCredentials stores the PerRPCCredentials required to issue RPCs.
	PerRPCCredentials []credentials.PerRPCCredentials
	// TransportCredentials stores the Authenticator required to setup a client
	// connection. Only one of TransportCredentials and CredsBundle is non-nil.
	TransportCredentials credentials.TransportCredentials
	// CredsBundle is the credentials bundle to be used. Only one of
	// TransportCredentials and CredsBundle is non-nil.
	CredsBundle credentials.Bundle
	// KeepaliveParams stores the keepalive parameters.
	KeepaliveParams keepalive.ClientParameters
	// StatsHandlers stores the handler for stats.
	StatsHandlers []stats.Handler
	// InitialWindowSize sets the initial window size for a stream.
	InitialWindowSize int32
	// InitialConnWindowSize sets the initial window size for a connection.
	InitialConnWindowSize int32
	// WriteBufferSize sets the size of write buffer which in turn determines how much data can be batched before it's written on the wire.
	WriteBufferSize int
	// ReadBufferSize sets the size of read buffer, which in turn determines how much data can be read at most for one read syscall.
	ReadBufferSize int
	// SharedWriteBuffer indicates whether connections should reuse write buffer
	SharedWriteBuffer bool
	// ChannelzParent sets the addrConn id which initiated the creation of this client transport.
	ChannelzParent *channelz.SubChannel
	// MaxHeaderListSize sets the max (uncompressed) size of header list that is prepared to be received.
	MaxHeaderListSize *uint32
	// UseProxy specifies if a proxy should be used.
	UseProxy bool
	// The mem.BufferPool to use when reading/writing to the wire.
	BufferPool mem.BufferPool
}

// WriteOptions provides additional hints and information for message
// transmission.
type WriteOptions struct {
	// Last indicates whether this write is the last piece for
	// this stream.
	Last bool
}

// CallHdr carries the information of a particular RPC.
type CallHdr struct {
	// Host specifies the peer's host.
	Host string

	// Method specifies the operation to perform.
	Method string

	// SendCompress specifies the compression algorithm applied on
	// outbound message.
	SendCompress string

	// Creds specifies credentials.PerRPCCredentials for a call.
	Creds credentials.PerRPCCredentials

	// ContentSubtype specifies the content-subtype for a request. For example, a
	// content-subtype of "proto" will result in a content-type of
	// "application/grpc+proto". The value of ContentSubtype must be all
	// lowercase, otherwise the behavior is undefined. See
	// https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md#requests
	// for more details.
	ContentSubtype string

	PreviousAttempts int // value of grpc-previous-rpc-attempts header to set

	DoneFunc func() // called when the stream is finished
}

// ClientTransport is the common interface for all gRPC client-side transport
// implementations.
type ClientTransport interface {
	// Close tears down this transport. Once it returns, the transport
	// should not be accessed any more. The caller must make sure this
	// is called only once.
	Close(err error)

	// GracefulClose starts to tear down the transport: the transport will stop
	// accepting new RPCs and NewStream will return error. Once all streams are
	// finished, the transport will close.
	//
	// It does not block.
	GracefulClose()

	// NewStream creates a Stream for an RPC.
	NewStream(ctx context.Context, callHdr *CallHdr) (*ClientStream, error)

	// Error returns a channel that is closed when some I/O error
	// happens. Typically the caller should have a goroutine to monitor
	// this in order to take action (e.g., close the current transport
	// and create a new one) in error case. It should not return nil
	// once the transport is initiated.
	Error() <-chan struct{}

	// GoAway returns a channel that is closed when ClientTransport
	// receives the draining signal from the server (e.g., GOAWAY frame in
	// HTTP/2).
	GoAway() <-chan struct{}

	// GetGoAwayReason returns the reason why GoAway frame was received, along
	// with a human readable string with debug info.
	GetGoAwayReason() (GoAwayReason, string)

	// RemoteAddr returns the remote network address.
	RemoteAddr() net.Addr
}

// ServerTransport is the common interface for all gRPC server-side transport
// implementations.
//
// Methods may be called concurrently from multiple goroutines, but
// Write methods for a given Stream will be called serially.
type ServerTransport interface {
	// HandleStreams receives incoming streams using the given handler.
	HandleStreams(context.Context, func(*ServerStream))

	// Close tears down the transport. Once it is called, the transport
	// should not be accessed any more. All the pending streams and their
	// handlers will be terminated asynchronously.
	Close(err error)

	// Peer returns the peer of the server transport.
	Peer() *peer.Peer

	// Drain notifies the client this ServerTransport stops accepting new RPCs.
	Drain(debugData string)
}

type internalServerTransport interface {
	ServerTransport
	writeHeader(s *ServerStream, md metadata.MD) error
	write(s *ServerStream, hdr []byte, data mem.BufferSlice, opts *WriteOptions) error
	writeStatus(s *ServerStream, st *status.Status) error
	incrMsgRecv()
}

// connectionErrorf creates an ConnectionError with the specified error description.
func connectionErrorf(temp bool, e error, format string, a ...any) ConnectionError {
	return ConnectionError{
		Desc: fmt.Sprintf(format, a...),
		temp: temp,
		err:  e,
	}
}

// ConnectionError is an error that results in the termination of the
// entire connection and the retry of all the active streams.
type ConnectionError struct {
	Desc string
	temp bool
	err  error
}

func (e ConnectionError) Error() string {
	return fmt.Sprintf("connection error: desc = %q", e.Desc)
}

// Temporary indicates if this connection error is temporary or fatal.
func (e ConnectionError) Temporary() bool {
	return e.temp
}

// Origin returns the original error of this connection error.
func (e ConnectionError) Origin() error {
	// Never return nil error here.
	// If the original error is nil, return itself.
	if e.err == nil {
		return e
	}
	return e.err
}

// Unwrap returns the original error of this connection error or nil when the
// origin is nil.
func (e ConnectionError) Unwrap() error {
	return e.err
}

var (
	// ErrConnClosing indicates that the transport is closing.
	ErrConnClosing = connectionErrorf(true, nil, "transport is closing")
	// errStreamDrain indicates that the stream is rejected because the
	// connection is draining. This could be caused by goaway or balancer
	// removing the address.
	errStreamDrain = status.Error(codes.Unavailable, "the connection is draining")
	// errStreamDone is returned from write at the client side to indicate application
	// layer of an error.
	errStreamDone = errors.New("the stream is done")
	// StatusGoAway indicates that the server sent a GOAWAY that included this
	// stream's ID in unprocessed RPCs.
	statusGoAway = status.New(codes.Unavailable, "the stream is rejected because server is draining the connection")
)

// GoAwayReason contains the reason for the GoAway frame received.
type GoAwayReason uint8

const (
	// GoAwayInvalid indicates that no GoAway frame is received.
	GoAwayInvalid GoAwayReason = 0
	// GoAwayNoReason is the default value when GoAway frame is received.
	GoAwayNoReason GoAwayReason = 1
	// GoAwayTooManyPings indicates that a GoAway frame with
	// ErrCodeEnhanceYourCalm was received and that the debug data said
	// "too_many_pings".
	GoAwayTooManyPings GoAwayReason = 2
)

// ContextErr converts the error from context package into a status error.
func ContextErr(err error) error {
	switch err {
	case context.DeadlineExceeded:
		return status.Error(codes.DeadlineExceeded, err.Error())
	case context.Canceled:
		return status.Error(codes.Canceled, err.Error())
	}
	return status.Errorf(codes.Internal, "Unexpected error from context packet: %v", err)
}
