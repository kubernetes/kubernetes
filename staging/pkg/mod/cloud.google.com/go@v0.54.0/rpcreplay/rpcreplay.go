// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rpcreplay

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"sync"

	pb "cloud.google.com/go/rpcreplay/proto/rpcreplay"
	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes"
	"github.com/golang/protobuf/ptypes/any"
	spb "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// A Recorder records RPCs for later playback.
type Recorder struct {
	mu   sync.Mutex
	w    *bufio.Writer
	f    *os.File
	next int
	err  error
	// BeforeFunc defines a function that can inspect and modify requests and responses
	// written to the replay file. It does not modify messages sent to the service.
	// It is run once before a request is written to the replay file, and once before a response
	// is written to the replay file.
	// The function is called with the method name and the message that triggered the callback.
	// If the function returns an error, the error will be returned to the client.
	// This is only executed for unary RPCs; streaming RPCs are not supported.
	BeforeFunc func(string, proto.Message) error
}

// NewRecorder creates a recorder that writes to filename. The file will
// also store the initial bytes for retrieval during replay.
//
// You must call Close on the Recorder to ensure that all data is written.
func NewRecorder(filename string, initial []byte) (*Recorder, error) {
	f, err := os.Create(filename)
	if err != nil {
		return nil, err
	}
	rec, err := NewRecorderWriter(f, initial)
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	rec.f = f
	return rec, nil
}

// NewRecorderWriter creates a recorder that writes to w. The initial
// bytes will also be written to w for retrieval during replay.
//
// You must call Close on the Recorder to ensure that all data is written.
func NewRecorderWriter(w io.Writer, initial []byte) (*Recorder, error) {
	bw := bufio.NewWriter(w)
	if err := writeHeader(bw, initial); err != nil {
		return nil, err
	}
	return &Recorder{w: bw, next: 1}, nil
}

// DialOptions returns the options that must be passed to grpc.Dial
// to enable recording.
func (r *Recorder) DialOptions() []grpc.DialOption {
	return []grpc.DialOption{
		grpc.WithUnaryInterceptor(r.interceptUnary),
		grpc.WithStreamInterceptor(r.interceptStream),
	}
}

// Close saves any unwritten information.
func (r *Recorder) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return r.err
	}
	err := r.w.Flush()
	if r.f != nil {
		if err2 := r.f.Close(); err == nil {
			err = err2
		}
	}
	return err
}

// Intercepts all unary (non-stream) RPCs.
func (r *Recorder) interceptUnary(ctx context.Context, method string, req, res interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
	ereq := &entry{
		kind:   pb.Entry_REQUEST,
		method: method,
		msg:    message{msg: proto.Clone(req.(proto.Message))},
	}

	if r.BeforeFunc != nil {
		if err := r.BeforeFunc(method, ereq.msg.msg); err != nil {
			return err
		}
	}
	refIndex, err := r.writeEntry(ereq)
	if err != nil {
		return err
	}
	ierr := invoker(ctx, method, req, res, cc, opts...)
	eres := &entry{
		kind:     pb.Entry_RESPONSE,
		refIndex: refIndex,
	}
	// If the error is not a gRPC status, then something more
	// serious is wrong. More significantly, we have no way
	// of serializing an arbitrary error. So just return it
	// without recording the response.
	if _, ok := status.FromError(ierr); !ok {
		r.mu.Lock()
		r.err = fmt.Errorf("saw non-status error in %s response: %v (%T)", method, ierr, ierr)
		r.mu.Unlock()
		return ierr
	}
	eres.msg.set(proto.Clone(res.(proto.Message)), ierr)
	if r.BeforeFunc != nil {
		if err := r.BeforeFunc(method, eres.msg.msg); err != nil {
			return err
		}
	}
	if _, err := r.writeEntry(eres); err != nil {
		return err
	}
	return ierr
}

func (r *Recorder) writeEntry(e *entry) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return 0, r.err
	}
	err := writeEntry(r.w, e)
	if err != nil {
		r.err = err
		return 0, err
	}
	n := r.next
	r.next++
	return n, nil
}

func (r *Recorder) interceptStream(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
	cstream, serr := streamer(ctx, desc, cc, method, opts...)
	e := &entry{
		kind:   pb.Entry_CREATE_STREAM,
		method: method,
	}
	e.msg.set(nil, serr)
	refIndex, err := r.writeEntry(e)
	if err != nil {
		return nil, err
	}
	return &recClientStream{
		ctx:      ctx,
		rec:      r,
		cstream:  cstream,
		refIndex: refIndex,
	}, serr
}

// A recClientStream implements the gprc.ClientStream interface.
// It behaves exactly like the default ClientStream, but also
// records all messages sent and received.
type recClientStream struct {
	ctx      context.Context
	rec      *Recorder
	cstream  grpc.ClientStream
	refIndex int
}

func (rcs *recClientStream) Context() context.Context { return rcs.ctx }

func (rcs *recClientStream) SendMsg(m interface{}) error {
	serr := rcs.cstream.SendMsg(m)
	e := &entry{
		kind:     pb.Entry_SEND,
		refIndex: rcs.refIndex,
	}
	e.msg.set(m, serr)
	if _, err := rcs.rec.writeEntry(e); err != nil {
		return err
	}
	return serr
}

func (rcs *recClientStream) RecvMsg(m interface{}) error {
	serr := rcs.cstream.RecvMsg(m)
	e := &entry{
		kind:     pb.Entry_RECV,
		refIndex: rcs.refIndex,
	}
	e.msg.set(m, serr)
	if _, err := rcs.rec.writeEntry(e); err != nil {
		return err
	}
	return serr
}

func (rcs *recClientStream) Header() (metadata.MD, error) {
	// TODO(jba): record.
	return rcs.cstream.Header()
}

func (rcs *recClientStream) Trailer() metadata.MD {
	// TODO(jba): record.
	return rcs.cstream.Trailer()
}

func (rcs *recClientStream) CloseSend() error {
	// TODO(jba): record.
	return rcs.cstream.CloseSend()
}

// A Replayer replays a set of RPCs saved by a Recorder.
type Replayer struct {
	initial []byte                                // initial state
	log     func(format string, v ...interface{}) // for debugging

	mu      sync.Mutex
	calls   []*call
	streams []*stream
	// BeforeFunc defines a function that can inspect and modify requests before they
	// are matched for responses from the replay file.
	// The function is called with the method name and the message that triggered the callback.
	// If the function returns an error, the error will be returned to the client.
	// This is only executed for unary RPCs; streaming RPCs are not supported.
	BeforeFunc func(string, proto.Message) error
}

// A call represents a unary RPC, with a request and response (or error).
type call struct {
	method   string
	request  proto.Message
	response message
}

// A stream represents a gRPC stream, with an initial create-stream call, followed by
// zero or more sends and/or receives.
type stream struct {
	method      string
	createIndex int
	createErr   error // error from create call
	sends       []message
	recvs       []message
}

// NewReplayer creates a Replayer that reads from filename.
func NewReplayer(filename string) (*Replayer, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return NewReplayerReader(f)
}

// NewReplayerReader creates a Replayer that reads from r.
func NewReplayerReader(r io.Reader) (*Replayer, error) {
	rep := &Replayer{
		log: func(string, ...interface{}) {},
	}
	if err := rep.read(r); err != nil {
		return nil, err
	}
	return rep, nil
}

// read reads the stream of recorded entries.
// It matches requests with responses, with each pair grouped
// into a call struct.
func (rep *Replayer) read(r io.Reader) error {
	r = bufio.NewReader(r)
	bytes, err := readHeader(r)
	if err != nil {
		return err
	}
	rep.initial = bytes

	callsByIndex := map[int]*call{}
	streamsByIndex := map[int]*stream{}
	for i := 1; ; i++ {
		e, err := readEntry(r)
		if err != nil {
			return err
		}
		if e == nil {
			break
		}
		switch e.kind {
		case pb.Entry_REQUEST:
			callsByIndex[i] = &call{
				method:  e.method,
				request: e.msg.msg,
			}

		case pb.Entry_RESPONSE:
			call := callsByIndex[e.refIndex]
			if call == nil {
				return fmt.Errorf("replayer: no request for response #%d", i)
			}
			delete(callsByIndex, e.refIndex)
			call.response = e.msg
			rep.calls = append(rep.calls, call)

		case pb.Entry_CREATE_STREAM:
			s := &stream{method: e.method, createIndex: i}
			s.createErr = e.msg.err
			streamsByIndex[i] = s
			rep.streams = append(rep.streams, s)

		case pb.Entry_SEND:
			s := streamsByIndex[e.refIndex]
			if s == nil {
				return fmt.Errorf("replayer: no stream for send #%d", i)
			}
			s.sends = append(s.sends, e.msg)

		case pb.Entry_RECV:
			s := streamsByIndex[e.refIndex]
			if s == nil {
				return fmt.Errorf("replayer: no stream for recv #%d", i)
			}
			s.recvs = append(s.recvs, e.msg)

		default:
			return fmt.Errorf("replayer: unknown kind %s", e.kind)
		}
	}
	if len(callsByIndex) > 0 {
		return fmt.Errorf("replayer: %d unmatched requests", len(callsByIndex))
	}
	return nil
}

// DialOptions returns the options that must be passed to grpc.Dial
// to enable replaying.
func (rep *Replayer) DialOptions() []grpc.DialOption {
	return []grpc.DialOption{
		// On replay, we make no RPCs, which means the connection may be closed
		// before the normally async Dial completes. Making the Dial synchronous
		// fixes that.
		grpc.WithBlock(),
		grpc.WithUnaryInterceptor(rep.interceptUnary),
		grpc.WithStreamInterceptor(rep.interceptStream),
	}
}

// Connection returns a fake gRPC connection suitable for replaying.
func (rep *Replayer) Connection() (*grpc.ClientConn, error) {
	// We don't need an actual connection, not even a loopback one.
	// But we do need something to attach gRPC interceptors to.
	// So we start a local server and connect to it, then close it down.
	srv := grpc.NewServer()
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		return nil, err
	}
	go func() {
		if err := srv.Serve(l); err != nil {
			panic(err) // we should never get an error because we just connect and stop
		}
	}()
	conn, err := grpc.Dial(l.Addr().String(),
		append([]grpc.DialOption{grpc.WithInsecure()}, rep.DialOptions()...)...)
	if err != nil {
		return nil, err
	}
	conn.Close()
	srv.Stop()
	return conn, nil
}

// Initial returns the initial state saved by the Recorder.
func (rep *Replayer) Initial() []byte { return rep.initial }

// SetLogFunc sets a function to be used for debug logging. The function
// should be safe to be called from multiple goroutines.
func (rep *Replayer) SetLogFunc(f func(format string, v ...interface{})) {
	rep.log = f
}

// Close closes the Replayer.
func (rep *Replayer) Close() error {
	return nil
}

func (rep *Replayer) interceptUnary(_ context.Context, method string, req, res interface{}, _ *grpc.ClientConn, _ grpc.UnaryInvoker, _ ...grpc.CallOption) error {
	mreq := req.(proto.Message)
	if rep.BeforeFunc != nil {
		if err := rep.BeforeFunc(method, mreq); err != nil {
			return err
		}
	}
	rep.log("request %s (%s)", method, req)
	call := rep.extractCall(method, mreq)
	if call == nil {
		return fmt.Errorf("replayer: request not found: %s", mreq)
	}
	rep.log("returning %v", call.response)
	if call.response.err != nil {
		return call.response.err
	}
	proto.Merge(res.(proto.Message), call.response.msg) // copy msg into res
	return nil
}

func (rep *Replayer) interceptStream(ctx context.Context, _ *grpc.StreamDesc, _ *grpc.ClientConn, method string, _ grpc.Streamer, _ ...grpc.CallOption) (grpc.ClientStream, error) {
	rep.log("create-stream %s", method)
	return &repClientStream{ctx: ctx, rep: rep, method: method}, nil
}

type repClientStream struct {
	ctx    context.Context
	rep    *Replayer
	method string
	str    *stream
}

func (rcs *repClientStream) Context() context.Context { return rcs.ctx }

func (rcs *repClientStream) SendMsg(req interface{}) error {
	if rcs.str == nil {
		if err := rcs.setStream(rcs.method, req.(proto.Message)); err != nil {
			return err
		}
	}
	if len(rcs.str.sends) == 0 {
		return fmt.Errorf("replayer: no more sends for stream %s, created at index %d",
			rcs.str.method, rcs.str.createIndex)
	}
	// TODO(jba): Do not assume that the sends happen in the same order on replay.
	msg := rcs.str.sends[0]
	rcs.str.sends = rcs.str.sends[1:]
	return msg.err
}

func (rcs *repClientStream) setStream(method string, req proto.Message) error {
	str := rcs.rep.extractStream(method, req)
	if str == nil {
		return fmt.Errorf("replayer: stream not found for method %s and request %v", method, req)
	}
	if str.createErr != nil {
		return str.createErr
	}
	rcs.str = str
	return nil
}

func (rcs *repClientStream) RecvMsg(m interface{}) error {
	if rcs.str == nil {
		// Receive before send; fall back to matching stream by method only.
		if err := rcs.setStream(rcs.method, nil); err != nil {
			return err
		}
	}
	if len(rcs.str.recvs) == 0 {
		return fmt.Errorf("replayer: no more receives for stream %s, created at index %d",
			rcs.str.method, rcs.str.createIndex)
	}
	msg := rcs.str.recvs[0]
	rcs.str.recvs = rcs.str.recvs[1:]
	if msg.err != nil {
		return msg.err
	}
	proto.Merge(m.(proto.Message), msg.msg) // copy msg into m
	return nil
}

func (rcs *repClientStream) Header() (metadata.MD, error) {
	log.Printf("replay: stream metadata not supported")
	return nil, nil
}

func (rcs *repClientStream) Trailer() metadata.MD {
	log.Printf("replay: stream metadata not supported")
	return nil
}

func (rcs *repClientStream) CloseSend() error {
	return nil
}

// extractCall finds the first call in the list with the same method
// and request. It returns nil if it can't find such a call.
func (rep *Replayer) extractCall(method string, req proto.Message) *call {
	rep.mu.Lock()
	defer rep.mu.Unlock()
	for i, call := range rep.calls {
		if call == nil {
			continue
		}
		if method == call.method && proto.Equal(req, call.request) {
			rep.calls[i] = nil // nil out this call so we don't reuse it
			return call
		}
	}
	return nil
}

// extractStream find the first stream in the list with the same method and the same
// first request sent. If req is nil, that means a receive occurred before a send, so
// it matches only on method.
func (rep *Replayer) extractStream(method string, req proto.Message) *stream {
	rep.mu.Lock()
	defer rep.mu.Unlock()
	for i, stream := range rep.streams {
		// Skip stream if it is nil (already extracted) or its method doesn't match.
		if stream == nil || stream.method != method {
			continue
		}
		// If there is a first request, skip stream if it has no requests or its first
		// request doesn't match.
		if req != nil && len(stream.sends) > 0 && !proto.Equal(req, stream.sends[0].msg) {
			continue
		}
		rep.streams[i] = nil // nil out this stream so we don't reuse it
		return stream
	}
	return nil
}

// Fprint reads the entries from filename and writes them to w in human-readable form.
// It is intended for debugging.
func Fprint(w io.Writer, filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return FprintReader(w, f)
}

// FprintReader reads the entries from r and writes them to w in human-readable form.
// It is intended for debugging.
func FprintReader(w io.Writer, r io.Reader) error {
	initial, err := readHeader(r)
	if err != nil {
		return err
	}
	fmt.Fprintf(w, "initial state: %q\n", string(initial))
	for i := 1; ; i++ {
		e, err := readEntry(r)
		if err != nil {
			return err
		}
		if e == nil {
			return nil
		}

		fmt.Fprintf(w, "#%d: kind: %s, method: %s, ref index: %d", i, e.kind, e.method, e.refIndex)
		switch {
		case e.msg.msg != nil:
			fmt.Fprintf(w, ", message:\n")
			if err := proto.MarshalText(w, e.msg.msg); err != nil {
				return err
			}
		case e.msg.err != nil:
			fmt.Fprintf(w, ", error: %v\n", e.msg.err)
		default:
			fmt.Fprintln(w)
		}
	}
}

// An entry holds one gRPC action (request, response, etc.).
type entry struct {
	kind     pb.Entry_Kind
	method   string
	msg      message
	refIndex int // index of corresponding request or create-stream
}

func (e1 *entry) equal(e2 *entry) bool {
	if e1 == nil && e2 == nil {
		return true
	}
	if e1 == nil || e2 == nil {
		return false
	}
	return e1.kind == e2.kind &&
		e1.method == e2.method &&
		proto.Equal(e1.msg.msg, e2.msg.msg) &&
		errEqual(e1.msg.err, e2.msg.err) &&
		e1.refIndex == e2.refIndex
}

func errEqual(e1, e2 error) bool {
	if e1 == e2 {
		return true
	}
	s1, ok1 := status.FromError(e1)
	s2, ok2 := status.FromError(e2)
	if !ok1 || !ok2 {
		return false
	}
	return proto.Equal(s1.Proto(), s2.Proto())
}

// message holds either a single proto.Message or an error.
type message struct {
	msg proto.Message
	err error
}

func (m *message) set(msg interface{}, err error) {
	m.err = err
	if err != io.EOF && msg != nil {
		m.msg = msg.(proto.Message)
	}
}

// File format:
//   header
//   sequence of Entry protos
//
// Header format:
//   magic string
//   a record containing the bytes of the initial state

const magic = "RPCReplay"

func writeHeader(w io.Writer, initial []byte) error {
	if _, err := io.WriteString(w, magic); err != nil {
		return err
	}
	return writeRecord(w, initial)
}

func readHeader(r io.Reader) ([]byte, error) {
	var buf [len(magic)]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		if err == io.EOF {
			err = errors.New("rpcreplay: empty replay file")
		}
		return nil, err
	}
	if string(buf[:]) != magic {
		return nil, errors.New("rpcreplay: not a replay file (does not begin with magic string)")
	}
	bytes, err := readRecord(r)
	if err == io.EOF {
		err = errors.New("rpcreplay: missing initial state")
	}
	return bytes, err
}

func writeEntry(w io.Writer, e *entry) error {
	var m proto.Message
	if e.msg.err != nil && e.msg.err != io.EOF {
		s, ok := status.FromError(e.msg.err)
		if !ok {
			return fmt.Errorf("rpcreplay: error %v is not a Status", e.msg.err)
		}
		m = s.Proto()
	} else {
		m = e.msg.msg
	}
	var a *any.Any
	var err error
	if m != nil {
		a, err = ptypes.MarshalAny(m)
		if err != nil {
			return err
		}
	}
	pe := &pb.Entry{
		Kind:     e.kind,
		Method:   e.method,
		Message:  a,
		IsError:  e.msg.err != nil,
		RefIndex: int32(e.refIndex),
	}
	bytes, err := proto.Marshal(pe)
	if err != nil {
		return err
	}
	return writeRecord(w, bytes)
}

func readEntry(r io.Reader) (*entry, error) {
	buf, err := readRecord(r)
	if err == io.EOF {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var pe pb.Entry
	if err := proto.Unmarshal(buf, &pe); err != nil {
		return nil, err
	}
	var msg message
	if pe.Message != nil {
		var any ptypes.DynamicAny
		if err := ptypes.UnmarshalAny(pe.Message, &any); err != nil {
			return nil, err
		}
		if pe.IsError {
			msg.err = status.ErrorProto(any.Message.(*spb.Status))
		} else {
			msg.msg = any.Message
		}
	} else if pe.IsError {
		msg.err = io.EOF
	} else if pe.Kind != pb.Entry_CREATE_STREAM {
		return nil, errors.New("rpcreplay: entry with nil message and false is_error")
	}
	return &entry{
		kind:     pe.Kind,
		method:   pe.Method,
		msg:      msg,
		refIndex: int(pe.RefIndex),
	}, nil
}

// A record consists of an unsigned 32-bit little-endian length L followed by L
// bytes.

func writeRecord(w io.Writer, data []byte) error {
	if err := binary.Write(w, binary.LittleEndian, uint32(len(data))); err != nil {
		return err
	}
	_, err := w.Write(data)
	return err
}

func readRecord(r io.Reader) ([]byte, error) {
	var size uint32
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return nil, err
	}
	buf := make([]byte, size)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return buf, nil
}
