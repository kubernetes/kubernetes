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

package grpc

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"net/url"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/encoding"
	"google.golang.org/grpc/encoding/proto"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/transport"
)

// Compressor defines the interface gRPC uses to compress a message.
//
// Deprecated: use package encoding.
type Compressor interface {
	// Do compresses p into w.
	Do(w io.Writer, p []byte) error
	// Type returns the compression algorithm the Compressor uses.
	Type() string
}

type gzipCompressor struct {
	pool sync.Pool
}

// NewGZIPCompressor creates a Compressor based on GZIP.
//
// Deprecated: use package encoding/gzip.
func NewGZIPCompressor() Compressor {
	c, _ := NewGZIPCompressorWithLevel(gzip.DefaultCompression)
	return c
}

// NewGZIPCompressorWithLevel is like NewGZIPCompressor but specifies the gzip compression level instead
// of assuming DefaultCompression.
//
// The error returned will be nil if the level is valid.
//
// Deprecated: use package encoding/gzip.
func NewGZIPCompressorWithLevel(level int) (Compressor, error) {
	if level < gzip.DefaultCompression || level > gzip.BestCompression {
		return nil, fmt.Errorf("grpc: invalid compression level: %d", level)
	}
	return &gzipCompressor{
		pool: sync.Pool{
			New: func() interface{} {
				w, err := gzip.NewWriterLevel(ioutil.Discard, level)
				if err != nil {
					panic(err)
				}
				return w
			},
		},
	}, nil
}

func (c *gzipCompressor) Do(w io.Writer, p []byte) error {
	z := c.pool.Get().(*gzip.Writer)
	defer c.pool.Put(z)
	z.Reset(w)
	if _, err := z.Write(p); err != nil {
		return err
	}
	return z.Close()
}

func (c *gzipCompressor) Type() string {
	return "gzip"
}

// Decompressor defines the interface gRPC uses to decompress a message.
//
// Deprecated: use package encoding.
type Decompressor interface {
	// Do reads the data from r and uncompress them.
	Do(r io.Reader) ([]byte, error)
	// Type returns the compression algorithm the Decompressor uses.
	Type() string
}

type gzipDecompressor struct {
	pool sync.Pool
}

// NewGZIPDecompressor creates a Decompressor based on GZIP.
//
// Deprecated: use package encoding/gzip.
func NewGZIPDecompressor() Decompressor {
	return &gzipDecompressor{}
}

func (d *gzipDecompressor) Do(r io.Reader) ([]byte, error) {
	var z *gzip.Reader
	switch maybeZ := d.pool.Get().(type) {
	case nil:
		newZ, err := gzip.NewReader(r)
		if err != nil {
			return nil, err
		}
		z = newZ
	case *gzip.Reader:
		z = maybeZ
		if err := z.Reset(r); err != nil {
			d.pool.Put(z)
			return nil, err
		}
	}

	defer func() {
		z.Close()
		d.pool.Put(z)
	}()
	return ioutil.ReadAll(z)
}

func (d *gzipDecompressor) Type() string {
	return "gzip"
}

// callInfo contains all related configuration and information about an RPC.
type callInfo struct {
	compressorType        string
	failFast              bool
	stream                *clientStream
	traceInfo             traceInfo // in trace.go
	maxReceiveMessageSize *int
	maxSendMessageSize    *int
	creds                 credentials.PerRPCCredentials
	contentSubtype        string
	codec                 baseCodec
}

func defaultCallInfo() *callInfo {
	return &callInfo{failFast: true}
}

// CallOption configures a Call before it starts or extracts information from
// a Call after it completes.
type CallOption interface {
	// before is called before the call is sent to any server.  If before
	// returns a non-nil error, the RPC fails with that error.
	before(*callInfo) error

	// after is called after the call has completed.  after cannot return an
	// error, so any failures should be reported via output parameters.
	after(*callInfo)
}

// EmptyCallOption does not alter the Call configuration.
// It can be embedded in another structure to carry satellite data for use
// by interceptors.
type EmptyCallOption struct{}

func (EmptyCallOption) before(*callInfo) error { return nil }
func (EmptyCallOption) after(*callInfo)        {}

// Header returns a CallOptions that retrieves the header metadata
// for a unary RPC.
func Header(md *metadata.MD) CallOption {
	return HeaderCallOption{HeaderAddr: md}
}

// HeaderCallOption is a CallOption for collecting response header metadata.
// The metadata field will be populated *after* the RPC completes.
// This is an EXPERIMENTAL API.
type HeaderCallOption struct {
	HeaderAddr *metadata.MD
}

func (o HeaderCallOption) before(c *callInfo) error { return nil }
func (o HeaderCallOption) after(c *callInfo) {
	if c.stream != nil {
		*o.HeaderAddr, _ = c.stream.Header()
	}
}

// Trailer returns a CallOptions that retrieves the trailer metadata
// for a unary RPC.
func Trailer(md *metadata.MD) CallOption {
	return TrailerCallOption{TrailerAddr: md}
}

// TrailerCallOption is a CallOption for collecting response trailer metadata.
// The metadata field will be populated *after* the RPC completes.
// This is an EXPERIMENTAL API.
type TrailerCallOption struct {
	TrailerAddr *metadata.MD
}

func (o TrailerCallOption) before(c *callInfo) error { return nil }
func (o TrailerCallOption) after(c *callInfo) {
	if c.stream != nil {
		*o.TrailerAddr = c.stream.Trailer()
	}
}

// Peer returns a CallOption that retrieves peer information for a unary RPC.
// The peer field will be populated *after* the RPC completes.
func Peer(p *peer.Peer) CallOption {
	return PeerCallOption{PeerAddr: p}
}

// PeerCallOption is a CallOption for collecting the identity of the remote
// peer. The peer field will be populated *after* the RPC completes.
// This is an EXPERIMENTAL API.
type PeerCallOption struct {
	PeerAddr *peer.Peer
}

func (o PeerCallOption) before(c *callInfo) error { return nil }
func (o PeerCallOption) after(c *callInfo) {
	if c.stream != nil {
		if x, ok := peer.FromContext(c.stream.Context()); ok {
			*o.PeerAddr = *x
		}
	}
}

// FailFast configures the action to take when an RPC is attempted on broken
// connections or unreachable servers.  If failFast is true, the RPC will fail
// immediately. Otherwise, the RPC client will block the call until a
// connection is available (or the call is canceled or times out) and will
// retry the call if it fails due to a transient error.  gRPC will not retry if
// data was written to the wire unless the server indicates it did not process
// the data.  Please refer to
// https://github.com/grpc/grpc/blob/master/doc/wait-for-ready.md.
//
// By default, RPCs are "Fail Fast".
func FailFast(failFast bool) CallOption {
	return FailFastCallOption{FailFast: failFast}
}

// FailFastCallOption is a CallOption for indicating whether an RPC should fail
// fast or not.
// This is an EXPERIMENTAL API.
type FailFastCallOption struct {
	FailFast bool
}

func (o FailFastCallOption) before(c *callInfo) error {
	c.failFast = o.FailFast
	return nil
}
func (o FailFastCallOption) after(c *callInfo) {}

// MaxCallRecvMsgSize returns a CallOption which sets the maximum message size the client can receive.
func MaxCallRecvMsgSize(s int) CallOption {
	return MaxRecvMsgSizeCallOption{MaxRecvMsgSize: s}
}

// MaxRecvMsgSizeCallOption is a CallOption that indicates the maximum message
// size the client can receive.
// This is an EXPERIMENTAL API.
type MaxRecvMsgSizeCallOption struct {
	MaxRecvMsgSize int
}

func (o MaxRecvMsgSizeCallOption) before(c *callInfo) error {
	c.maxReceiveMessageSize = &o.MaxRecvMsgSize
	return nil
}
func (o MaxRecvMsgSizeCallOption) after(c *callInfo) {}

// MaxCallSendMsgSize returns a CallOption which sets the maximum message size the client can send.
func MaxCallSendMsgSize(s int) CallOption {
	return MaxSendMsgSizeCallOption{MaxSendMsgSize: s}
}

// MaxSendMsgSizeCallOption is a CallOption that indicates the maximum message
// size the client can send.
// This is an EXPERIMENTAL API.
type MaxSendMsgSizeCallOption struct {
	MaxSendMsgSize int
}

func (o MaxSendMsgSizeCallOption) before(c *callInfo) error {
	c.maxSendMessageSize = &o.MaxSendMsgSize
	return nil
}
func (o MaxSendMsgSizeCallOption) after(c *callInfo) {}

// PerRPCCredentials returns a CallOption that sets credentials.PerRPCCredentials
// for a call.
func PerRPCCredentials(creds credentials.PerRPCCredentials) CallOption {
	return PerRPCCredsCallOption{Creds: creds}
}

// PerRPCCredsCallOption is a CallOption that indicates the per-RPC
// credentials to use for the call.
// This is an EXPERIMENTAL API.
type PerRPCCredsCallOption struct {
	Creds credentials.PerRPCCredentials
}

func (o PerRPCCredsCallOption) before(c *callInfo) error {
	c.creds = o.Creds
	return nil
}
func (o PerRPCCredsCallOption) after(c *callInfo) {}

// UseCompressor returns a CallOption which sets the compressor used when
// sending the request.  If WithCompressor is also set, UseCompressor has
// higher priority.
//
// This API is EXPERIMENTAL.
func UseCompressor(name string) CallOption {
	return CompressorCallOption{CompressorType: name}
}

// CompressorCallOption is a CallOption that indicates the compressor to use.
// This is an EXPERIMENTAL API.
type CompressorCallOption struct {
	CompressorType string
}

func (o CompressorCallOption) before(c *callInfo) error {
	c.compressorType = o.CompressorType
	return nil
}
func (o CompressorCallOption) after(c *callInfo) {}

// CallContentSubtype returns a CallOption that will set the content-subtype
// for a call. For example, if content-subtype is "json", the Content-Type over
// the wire will be "application/grpc+json". The content-subtype is converted
// to lowercase before being included in Content-Type. See Content-Type on
// https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md#requests for
// more details.
//
// If CallCustomCodec is not also used, the content-subtype will be used to
// look up the Codec to use in the registry controlled by RegisterCodec. See
// the documentation on RegisterCodec for details on registration. The lookup
// of content-subtype is case-insensitive. If no such Codec is found, the call
// will result in an error with code codes.Internal.
//
// If CallCustomCodec is also used, that Codec will be used for all request and
// response messages, with the content-subtype set to the given contentSubtype
// here for requests.
func CallContentSubtype(contentSubtype string) CallOption {
	return ContentSubtypeCallOption{ContentSubtype: strings.ToLower(contentSubtype)}
}

// ContentSubtypeCallOption is a CallOption that indicates the content-subtype
// used for marshaling messages.
// This is an EXPERIMENTAL API.
type ContentSubtypeCallOption struct {
	ContentSubtype string
}

func (o ContentSubtypeCallOption) before(c *callInfo) error {
	c.contentSubtype = o.ContentSubtype
	return nil
}
func (o ContentSubtypeCallOption) after(c *callInfo) {}

// CallCustomCodec returns a CallOption that will set the given Codec to be
// used for all request and response messages for a call. The result of calling
// String() will be used as the content-subtype in a case-insensitive manner.
//
// See Content-Type on
// https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md#requests for
// more details. Also see the documentation on RegisterCodec and
// CallContentSubtype for more details on the interaction between Codec and
// content-subtype.
//
// This function is provided for advanced users; prefer to use only
// CallContentSubtype to select a registered codec instead.
func CallCustomCodec(codec Codec) CallOption {
	return CustomCodecCallOption{Codec: codec}
}

// CustomCodecCallOption is a CallOption that indicates the codec used for
// marshaling messages.
// This is an EXPERIMENTAL API.
type CustomCodecCallOption struct {
	Codec Codec
}

func (o CustomCodecCallOption) before(c *callInfo) error {
	c.codec = o.Codec
	return nil
}
func (o CustomCodecCallOption) after(c *callInfo) {}

// The format of the payload: compressed or not?
type payloadFormat uint8

const (
	compressionNone payloadFormat = iota // no compression
	compressionMade
)

// parser reads complete gRPC messages from the underlying reader.
type parser struct {
	// r is the underlying reader.
	// See the comment on recvMsg for the permissible
	// error types.
	r io.Reader

	// The header of a gRPC message. Find more detail at
	// https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md
	header [5]byte
}

// recvMsg reads a complete gRPC message from the stream.
//
// It returns the message and its payload (compression/encoding)
// format. The caller owns the returned msg memory.
//
// If there is an error, possible values are:
//   * io.EOF, when no messages remain
//   * io.ErrUnexpectedEOF
//   * of type transport.ConnectionError
//   * of type transport.StreamError
// No other error values or types must be returned, which also means
// that the underlying io.Reader must not return an incompatible
// error.
func (p *parser) recvMsg(maxReceiveMessageSize int) (pf payloadFormat, msg []byte, err error) {
	if _, err := p.r.Read(p.header[:]); err != nil {
		return 0, nil, err
	}

	pf = payloadFormat(p.header[0])
	length := binary.BigEndian.Uint32(p.header[1:])

	if length == 0 {
		return pf, nil, nil
	}
	if int64(length) > int64(maxInt) {
		return 0, nil, status.Errorf(codes.ResourceExhausted, "grpc: received message larger than max length allowed on current machine (%d vs. %d)", length, maxInt)
	}
	if int(length) > maxReceiveMessageSize {
		return 0, nil, status.Errorf(codes.ResourceExhausted, "grpc: received message larger than max (%d vs. %d)", length, maxReceiveMessageSize)
	}
	// TODO(bradfitz,zhaoq): garbage. reuse buffer after proto decoding instead
	// of making it for each message:
	msg = make([]byte, int(length))
	if _, err := p.r.Read(msg); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return 0, nil, err
	}
	return pf, msg, nil
}

// encode serializes msg and returns a buffer of message header and a buffer of msg.
// If msg is nil, it generates the message header and an empty msg buffer.
// TODO(ddyihai): eliminate extra Compressor parameter.
func encode(c baseCodec, msg interface{}, cp Compressor, outPayload *stats.OutPayload, compressor encoding.Compressor) ([]byte, []byte, error) {
	var (
		b    []byte
		cbuf *bytes.Buffer
	)
	const (
		payloadLen = 1
		sizeLen    = 4
	)
	if msg != nil {
		var err error
		b, err = c.Marshal(msg)
		if err != nil {
			return nil, nil, status.Errorf(codes.Internal, "grpc: error while marshaling: %v", err.Error())
		}
		if outPayload != nil {
			outPayload.Payload = msg
			// TODO truncate large payload.
			outPayload.Data = b
			outPayload.Length = len(b)
		}
		if compressor != nil || cp != nil {
			cbuf = new(bytes.Buffer)
			// Has compressor, check Compressor is set by UseCompressor first.
			if compressor != nil {
				z, _ := compressor.Compress(cbuf)
				if _, err := z.Write(b); err != nil {
					return nil, nil, status.Errorf(codes.Internal, "grpc: error while compressing: %v", err.Error())
				}
				z.Close()
			} else {
				// If Compressor is not set by UseCompressor, use default Compressor
				if err := cp.Do(cbuf, b); err != nil {
					return nil, nil, status.Errorf(codes.Internal, "grpc: error while compressing: %v", err.Error())
				}
			}
			b = cbuf.Bytes()
		}
	}
	if uint(len(b)) > math.MaxUint32 {
		return nil, nil, status.Errorf(codes.ResourceExhausted, "grpc: message too large (%d bytes)", len(b))
	}

	bufHeader := make([]byte, payloadLen+sizeLen)
	if compressor != nil || cp != nil {
		bufHeader[0] = byte(compressionMade)
	} else {
		bufHeader[0] = byte(compressionNone)
	}

	// Write length of b into buf
	binary.BigEndian.PutUint32(bufHeader[payloadLen:], uint32(len(b)))
	if outPayload != nil {
		outPayload.WireLength = payloadLen + sizeLen + len(b)
	}
	return bufHeader, b, nil
}

func checkRecvPayload(pf payloadFormat, recvCompress string, haveCompressor bool) *status.Status {
	switch pf {
	case compressionNone:
	case compressionMade:
		if recvCompress == "" || recvCompress == encoding.Identity {
			return status.New(codes.Internal, "grpc: compressed flag set with identity or empty encoding")
		}
		if !haveCompressor {
			return status.Newf(codes.Unimplemented, "grpc: Decompressor is not installed for grpc-encoding %q", recvCompress)
		}
	default:
		return status.Newf(codes.Internal, "grpc: received unexpected payload format %d", pf)
	}
	return nil
}

// For the two compressor parameters, both should not be set, but if they are,
// dc takes precedence over compressor.
// TODO(dfawley): wrap the old compressor/decompressor using the new API?
func recv(p *parser, c baseCodec, s *transport.Stream, dc Decompressor, m interface{}, maxReceiveMessageSize int, inPayload *stats.InPayload, compressor encoding.Compressor) error {
	pf, d, err := p.recvMsg(maxReceiveMessageSize)
	if err != nil {
		return err
	}
	if inPayload != nil {
		inPayload.WireLength = len(d)
	}

	if st := checkRecvPayload(pf, s.RecvCompress(), compressor != nil || dc != nil); st != nil {
		return st.Err()
	}

	if pf == compressionMade {
		// To match legacy behavior, if the decompressor is set by WithDecompressor or RPCDecompressor,
		// use this decompressor as the default.
		if dc != nil {
			d, err = dc.Do(bytes.NewReader(d))
			if err != nil {
				return status.Errorf(codes.Internal, "grpc: failed to decompress the received message %v", err)
			}
		} else {
			dcReader, err := compressor.Decompress(bytes.NewReader(d))
			if err != nil {
				return status.Errorf(codes.Internal, "grpc: failed to decompress the received message %v", err)
			}
			d, err = ioutil.ReadAll(dcReader)
			if err != nil {
				return status.Errorf(codes.Internal, "grpc: failed to decompress the received message %v", err)
			}
		}
	}
	if len(d) > maxReceiveMessageSize {
		// TODO: Revisit the error code. Currently keep it consistent with java
		// implementation.
		return status.Errorf(codes.ResourceExhausted, "grpc: received message larger than max (%d vs. %d)", len(d), maxReceiveMessageSize)
	}
	if err := c.Unmarshal(d, m); err != nil {
		return status.Errorf(codes.Internal, "grpc: failed to unmarshal the received message %v", err)
	}
	if inPayload != nil {
		inPayload.RecvTime = time.Now()
		inPayload.Payload = m
		// TODO truncate large payload.
		inPayload.Data = d
		inPayload.Length = len(d)
	}
	return nil
}

type rpcInfo struct {
	failfast bool
}

type rpcInfoContextKey struct{}

func newContextWithRPCInfo(ctx context.Context, failfast bool) context.Context {
	return context.WithValue(ctx, rpcInfoContextKey{}, &rpcInfo{failfast: failfast})
}

func rpcInfoFromContext(ctx context.Context) (s *rpcInfo, ok bool) {
	s, ok = ctx.Value(rpcInfoContextKey{}).(*rpcInfo)
	return
}

// Code returns the error code for err if it was produced by the rpc system.
// Otherwise, it returns codes.Unknown.
//
// Deprecated: use status.FromError and Code method instead.
func Code(err error) codes.Code {
	if s, ok := status.FromError(err); ok {
		return s.Code()
	}
	return codes.Unknown
}

// ErrorDesc returns the error description of err if it was produced by the rpc system.
// Otherwise, it returns err.Error() or empty string when err is nil.
//
// Deprecated: use status.FromError and Message method instead.
func ErrorDesc(err error) string {
	if s, ok := status.FromError(err); ok {
		return s.Message()
	}
	return err.Error()
}

// Errorf returns an error containing an error code and a description;
// Errorf returns nil if c is OK.
//
// Deprecated: use status.Errorf instead.
func Errorf(c codes.Code, format string, a ...interface{}) error {
	return status.Errorf(c, format, a...)
}

// setCallInfoCodec should only be called after CallOptions have been applied.
func setCallInfoCodec(c *callInfo) error {
	if c.codec != nil {
		// codec was already set by a CallOption; use it.
		return nil
	}

	if c.contentSubtype == "" {
		// No codec specified in CallOptions; use proto by default.
		c.codec = encoding.GetCodec(proto.Name)
		return nil
	}

	// c.contentSubtype is already lowercased in CallContentSubtype
	c.codec = encoding.GetCodec(c.contentSubtype)
	if c.codec == nil {
		return status.Errorf(codes.Internal, "no codec registered for content-subtype %s", c.contentSubtype)
	}
	return nil
}

// parseDialTarget returns the network and address to pass to dialer
func parseDialTarget(target string) (net string, addr string) {
	net = "tcp"

	m1 := strings.Index(target, ":")
	m2 := strings.Index(target, ":/")

	// handle unix:addr which will fail with url.Parse
	if m1 >= 0 && m2 < 0 {
		if n := target[0:m1]; n == "unix" {
			net = n
			addr = target[m1+1:]
			return net, addr
		}
	}
	if m2 >= 0 {
		t, err := url.Parse(target)
		if err != nil {
			return net, target
		}
		scheme := t.Scheme
		addr = t.Path
		if scheme == "unix" {
			net = scheme
			if addr == "" {
				addr = t.Host
			}
			return net, addr
		}
	}

	return net, target
}

// The SupportPackageIsVersion variables are referenced from generated protocol
// buffer files to ensure compatibility with the gRPC version used.  The latest
// support package version is 5.
//
// Older versions are kept for compatibility. They may be removed if
// compatibility cannot be maintained.
//
// These constants should not be referenced from any other code.
const (
	SupportPackageIsVersion3 = true
	SupportPackageIsVersion4 = true
	SupportPackageIsVersion5 = true
)

// Version is the current grpc version.
const Version = "1.12.2"

const grpcUA = "grpc-go/" + Version
