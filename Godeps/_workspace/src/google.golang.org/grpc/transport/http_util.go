/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package transport

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/bradfitz/http2"
	"github.com/bradfitz/http2/hpack"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/metadata"
)

const (
	// http2MaxFrameLen specifies the max length of a HTTP2 frame.
	http2MaxFrameLen = 16384 // 16KB frame
	// http://http2.github.io/http2-spec/#SettingValues
	http2InitHeaderTableSize = 4096
	// http2IOBufSize specifies the buffer size for sending frames.
	http2IOBufSize = 32 * 1024
)

var (
	clientPreface = []byte(http2.ClientPreface)
)

var http2RSTErrConvTab = map[http2.ErrCode]codes.Code{
	http2.ErrCodeNo:                 codes.Internal,
	http2.ErrCodeProtocol:           codes.Internal,
	http2.ErrCodeInternal:           codes.Internal,
	http2.ErrCodeFlowControl:        codes.Internal,
	http2.ErrCodeSettingsTimeout:    codes.Internal,
	http2.ErrCodeFrameSize:          codes.Internal,
	http2.ErrCodeRefusedStream:      codes.Unavailable,
	http2.ErrCodeCancel:             codes.Canceled,
	http2.ErrCodeCompression:        codes.Internal,
	http2.ErrCodeConnect:            codes.Internal,
	http2.ErrCodeEnhanceYourCalm:    codes.ResourceExhausted,
	http2.ErrCodeInadequateSecurity: codes.PermissionDenied,
}

var statusCodeConvTab = map[codes.Code]http2.ErrCode{
	codes.Internal:          http2.ErrCodeInternal, // pick an arbitrary one which is matched.
	codes.Canceled:          http2.ErrCodeCancel,
	codes.Unavailable:       http2.ErrCodeRefusedStream,
	codes.ResourceExhausted: http2.ErrCodeEnhanceYourCalm,
	codes.PermissionDenied:  http2.ErrCodeInadequateSecurity,
}

// Records the states during HPACK decoding. Must be reset once the
// decoding of the entire headers are finished.
type decodeState struct {
	// statusCode caches the stream status received from the trailer
	// the server sent. Client side only.
	statusCode codes.Code
	statusDesc string
	// Server side only fields.
	timeoutSet bool
	timeout    time.Duration
	method     string
	// key-value metadata map from the peer.
	mdata map[string]string
}

// An hpackDecoder decodes HTTP2 headers which may span multiple frames.
type hpackDecoder struct {
	h     *hpack.Decoder
	state decodeState
	err   error // The err when decoding
}

// A headerFrame is either a http2.HeaderFrame or http2.ContinuationFrame.
type headerFrame interface {
	Header() http2.FrameHeader
	HeaderBlockFragment() []byte
	HeadersEnded() bool
}

// isReservedHeader checks whether hdr belongs to HTTP2 headers
// reserved by gRPC protocol. Any other headers are classified as the
// user-specified metadata.
func isReservedHeader(hdr string) bool {
	if hdr[0] == ':' {
		return true
	}
	switch hdr {
	case "content-type",
		"grpc-message-type",
		"grpc-encoding",
		"grpc-message",
		"grpc-status",
		"grpc-timeout",
		"te",
		"user-agent":
		return true
	default:
		return false
	}
}

func newHPACKDecoder() *hpackDecoder {
	d := &hpackDecoder{}
	d.h = hpack.NewDecoder(http2InitHeaderTableSize, func(f hpack.HeaderField) {
		switch f.Name {
		case "grpc-status":
			code, err := strconv.Atoi(f.Value)
			if err != nil {
				d.err = StreamErrorf(codes.Internal, "transport: malformed grpc-status: %v", err)
				return
			}
			d.state.statusCode = codes.Code(code)
		case "grpc-message":
			d.state.statusDesc = f.Value
		case "grpc-timeout":
			d.state.timeoutSet = true
			var err error
			d.state.timeout, err = timeoutDecode(f.Value)
			if err != nil {
				d.err = StreamErrorf(codes.Internal, "transport: malformed time-out: %v", err)
				return
			}
		case ":path":
			d.state.method = f.Value
		default:
			if !isReservedHeader(f.Name) {
				if d.state.mdata == nil {
					d.state.mdata = make(map[string]string)
				}
				k, v, err := metadata.DecodeKeyValue(f.Name, f.Value)
				if err != nil {
					grpclog.Printf("Failed to decode (%q, %q): %v", f.Name, f.Value, err)
					return
				}
				d.state.mdata[k] = v
			}
		}
	})
	return d
}

func (d *hpackDecoder) decodeClientHTTP2Headers(frame headerFrame) (endHeaders bool, err error) {
	d.err = nil
	_, err = d.h.Write(frame.HeaderBlockFragment())
	if err != nil {
		err = StreamErrorf(codes.Internal, "transport: HPACK header decode error: %v", err)
	}

	if frame.HeadersEnded() {
		if closeErr := d.h.Close(); closeErr != nil && err == nil {
			err = StreamErrorf(codes.Internal, "transport: HPACK decoder close error: %v", closeErr)
		}
		endHeaders = true
	}

	if err == nil && d.err != nil {
		err = d.err
	}
	return
}

func (d *hpackDecoder) decodeServerHTTP2Headers(frame headerFrame) (endHeaders bool, err error) {
	d.err = nil
	_, err = d.h.Write(frame.HeaderBlockFragment())
	if err != nil {
		err = StreamErrorf(codes.Internal, "transport: HPACK header decode error: %v", err)
	}

	if frame.HeadersEnded() {
		if closeErr := d.h.Close(); closeErr != nil && err == nil {
			err = StreamErrorf(codes.Internal, "transport: HPACK decoder close error: %v", closeErr)
		}
		endHeaders = true
	}

	if err == nil && d.err != nil {
		err = d.err
	}
	return
}

type timeoutUnit uint8

const (
	hour        timeoutUnit = 'H'
	minute      timeoutUnit = 'M'
	second      timeoutUnit = 'S'
	millisecond timeoutUnit = 'm'
	microsecond timeoutUnit = 'u'
	nanosecond  timeoutUnit = 'n'
)

func timeoutUnitToDuration(u timeoutUnit) (d time.Duration, ok bool) {
	switch u {
	case hour:
		return time.Hour, true
	case minute:
		return time.Minute, true
	case second:
		return time.Second, true
	case millisecond:
		return time.Millisecond, true
	case microsecond:
		return time.Microsecond, true
	case nanosecond:
		return time.Nanosecond, true
	default:
	}
	return
}

const maxTimeoutValue int64 = 100000000 - 1

// div does integer division and round-up the result. Note that this is
// equivalent to (d+r-1)/r but has less chance to overflow.
func div(d, r time.Duration) int64 {
	if m := d % r; m > 0 {
		return int64(d/r + 1)
	}
	return int64(d / r)
}

// TODO(zhaoq): It is the simplistic and not bandwidth efficient. Improve it.
func timeoutEncode(t time.Duration) string {
	if d := div(t, time.Nanosecond); d <= maxTimeoutValue {
		return strconv.FormatInt(d, 10) + "n"
	}
	if d := div(t, time.Microsecond); d <= maxTimeoutValue {
		return strconv.FormatInt(d, 10) + "u"
	}
	if d := div(t, time.Millisecond); d <= maxTimeoutValue {
		return strconv.FormatInt(d, 10) + "m"
	}
	if d := div(t, time.Second); d <= maxTimeoutValue {
		return strconv.FormatInt(d, 10) + "S"
	}
	if d := div(t, time.Minute); d <= maxTimeoutValue {
		return strconv.FormatInt(d, 10) + "M"
	}
	// Note that maxTimeoutValue * time.Hour > MaxInt64.
	return strconv.FormatInt(div(t, time.Hour), 10) + "H"
}

func timeoutDecode(s string) (time.Duration, error) {
	size := len(s)
	if size < 2 {
		return 0, fmt.Errorf("transport: timeout string is too short: %q", s)
	}
	unit := timeoutUnit(s[size-1])
	d, ok := timeoutUnitToDuration(unit)
	if !ok {
		return 0, fmt.Errorf("transport: timeout unit is not recognized: %q", s)
	}
	t, err := strconv.ParseInt(s[:size-1], 10, 64)
	if err != nil {
		return 0, err
	}
	return d * time.Duration(t), nil
}

type framer struct {
	numWriters int32
	reader     io.Reader
	writer     *bufio.Writer
	fr         *http2.Framer
}

func newFramer(conn net.Conn) *framer {
	f := &framer{
		reader: conn,
		writer: bufio.NewWriterSize(conn, http2IOBufSize),
	}
	f.fr = http2.NewFramer(f.writer, f.reader)
	return f
}

func (f *framer) adjustNumWriters(i int32) int32 {
	return atomic.AddInt32(&f.numWriters, i)
}

// The following writeXXX functions can only be called when the caller gets
// unblocked from writableChan channel (i.e., owns the privilege to write).

func (f *framer) writeContinuation(forceFlush bool, streamID uint32, endHeaders bool, headerBlockFragment []byte) error {
	if err := f.fr.WriteContinuation(streamID, endHeaders, headerBlockFragment); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) writeData(forceFlush bool, streamID uint32, endStream bool, data []byte) error {
	if err := f.fr.WriteData(streamID, endStream, data); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) writeGoAway(forceFlush bool, maxStreamID uint32, code http2.ErrCode, debugData []byte) error {
	if err := f.fr.WriteGoAway(maxStreamID, code, debugData); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) writeHeaders(forceFlush bool, p http2.HeadersFrameParam) error {
	if err := f.fr.WriteHeaders(p); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) writePing(forceFlush, ack bool, data [8]byte) error {
	if err := f.fr.WritePing(ack, data); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) writePriority(forceFlush bool, streamID uint32, p http2.PriorityParam) error {
	if err := f.fr.WritePriority(streamID, p); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) writePushPromise(forceFlush bool, p http2.PushPromiseParam) error {
	if err := f.fr.WritePushPromise(p); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) writeRSTStream(forceFlush bool, streamID uint32, code http2.ErrCode) error {
	if err := f.fr.WriteRSTStream(streamID, code); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) writeSettings(forceFlush bool, settings ...http2.Setting) error {
	if err := f.fr.WriteSettings(settings...); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) writeSettingsAck(forceFlush bool) error {
	if err := f.fr.WriteSettingsAck(); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) writeWindowUpdate(forceFlush bool, streamID, incr uint32) error {
	if err := f.fr.WriteWindowUpdate(streamID, incr); err != nil {
		return err
	}
	if forceFlush {
		return f.writer.Flush()
	}
	return nil
}

func (f *framer) flushWrite() error {
	return f.writer.Flush()
}

func (f *framer) readFrame() (http2.Frame, error) {
	return f.fr.ReadFrame()
}
