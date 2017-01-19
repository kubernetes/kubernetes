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
	"bytes"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/metadata"
)

const (
	// The primary user agent
	primaryUA = "grpc-go/1.0"
	// http2MaxFrameLen specifies the max length of a HTTP2 frame.
	http2MaxFrameLen = 16384 // 16KB frame
	// http://http2.github.io/http2-spec/#SettingValues
	http2InitHeaderTableSize = 4096
	// http2IOBufSize specifies the buffer size for sending frames.
	http2IOBufSize = 32 * 1024
)

var (
	clientPreface   = []byte(http2.ClientPreface)
	http2ErrConvTab = map[http2.ErrCode]codes.Code{
		http2.ErrCodeNo:                 codes.Internal,
		http2.ErrCodeProtocol:           codes.Internal,
		http2.ErrCodeInternal:           codes.Internal,
		http2.ErrCodeFlowControl:        codes.ResourceExhausted,
		http2.ErrCodeSettingsTimeout:    codes.Internal,
		http2.ErrCodeStreamClosed:       codes.Internal,
		http2.ErrCodeFrameSize:          codes.Internal,
		http2.ErrCodeRefusedStream:      codes.Unavailable,
		http2.ErrCodeCancel:             codes.Canceled,
		http2.ErrCodeCompression:        codes.Internal,
		http2.ErrCodeConnect:            codes.Internal,
		http2.ErrCodeEnhanceYourCalm:    codes.ResourceExhausted,
		http2.ErrCodeInadequateSecurity: codes.PermissionDenied,
		http2.ErrCodeHTTP11Required:     codes.FailedPrecondition,
	}
	statusCodeConvTab = map[codes.Code]http2.ErrCode{
		codes.Internal:          http2.ErrCodeInternal,
		codes.Canceled:          http2.ErrCodeCancel,
		codes.Unavailable:       http2.ErrCodeRefusedStream,
		codes.ResourceExhausted: http2.ErrCodeEnhanceYourCalm,
		codes.PermissionDenied:  http2.ErrCodeInadequateSecurity,
	}
)

// Records the states during HPACK decoding. Must be reset once the
// decoding of the entire headers are finished.
type decodeState struct {
	err error // first error encountered decoding

	encoding string
	// statusCode caches the stream status received from the trailer
	// the server sent. Client side only.
	statusCode codes.Code
	statusDesc string
	// Server side only fields.
	timeoutSet bool
	timeout    time.Duration
	method     string
	// key-value metadata map from the peer.
	mdata map[string][]string
}

// isReservedHeader checks whether hdr belongs to HTTP2 headers
// reserved by gRPC protocol. Any other headers are classified as the
// user-specified metadata.
func isReservedHeader(hdr string) bool {
	if hdr != "" && hdr[0] == ':' {
		return true
	}
	switch hdr {
	case "content-type",
		"grpc-message-type",
		"grpc-encoding",
		"grpc-message",
		"grpc-status",
		"grpc-timeout",
		"te":
		return true
	default:
		return false
	}
}

// isWhitelistedPseudoHeader checks whether hdr belongs to HTTP2 pseudoheaders
// that should be propagated into metadata visible to users.
func isWhitelistedPseudoHeader(hdr string) bool {
	switch hdr {
	case ":authority":
		return true
	default:
		return false
	}
}

func (d *decodeState) setErr(err error) {
	if d.err == nil {
		d.err = err
	}
}

func validContentType(t string) bool {
	e := "application/grpc"
	if !strings.HasPrefix(t, e) {
		return false
	}
	// Support variations on the content-type
	// (e.g. "application/grpc+blah", "application/grpc;blah").
	if len(t) > len(e) && t[len(e)] != '+' && t[len(e)] != ';' {
		return false
	}
	return true
}

func (d *decodeState) processHeaderField(f hpack.HeaderField) {
	switch f.Name {
	case "content-type":
		if !validContentType(f.Value) {
			d.setErr(StreamErrorf(codes.FailedPrecondition, "transport: received the unexpected content-type %q", f.Value))
			return
		}
	case "grpc-encoding":
		d.encoding = f.Value
	case "grpc-status":
		code, err := strconv.Atoi(f.Value)
		if err != nil {
			d.setErr(StreamErrorf(codes.Internal, "transport: malformed grpc-status: %v", err))
			return
		}
		d.statusCode = codes.Code(code)
	case "grpc-message":
		d.statusDesc = decodeGrpcMessage(f.Value)
	case "grpc-timeout":
		d.timeoutSet = true
		var err error
		d.timeout, err = decodeTimeout(f.Value)
		if err != nil {
			d.setErr(StreamErrorf(codes.Internal, "transport: malformed time-out: %v", err))
			return
		}
	case ":path":
		d.method = f.Value
	default:
		if !isReservedHeader(f.Name) || isWhitelistedPseudoHeader(f.Name) {
			if f.Name == "user-agent" {
				i := strings.LastIndex(f.Value, " ")
				if i == -1 {
					// There is no application user agent string being set.
					return
				}
				// Extract the application user agent string.
				f.Value = f.Value[:i]
			}
			if d.mdata == nil {
				d.mdata = make(map[string][]string)
			}
			k, v, err := metadata.DecodeKeyValue(f.Name, f.Value)
			if err != nil {
				grpclog.Printf("Failed to decode (%q, %q): %v", f.Name, f.Value, err)
				return
			}
			d.mdata[k] = append(d.mdata[k], v)
		}
	}
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
func encodeTimeout(t time.Duration) string {
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

func decodeTimeout(s string) (time.Duration, error) {
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

const (
	spaceByte   = ' '
	tildaByte   = '~'
	percentByte = '%'
)

// encodeGrpcMessage is used to encode status code in header field
// "grpc-message".
// It checks to see if each individual byte in msg is an
// allowable byte, and then either percent encoding or passing it through.
// When percent encoding, the byte is converted into hexadecimal notation
// with a '%' prepended.
func encodeGrpcMessage(msg string) string {
	if msg == "" {
		return ""
	}
	lenMsg := len(msg)
	for i := 0; i < lenMsg; i++ {
		c := msg[i]
		if !(c >= spaceByte && c < tildaByte && c != percentByte) {
			return encodeGrpcMessageUnchecked(msg)
		}
	}
	return msg
}

func encodeGrpcMessageUnchecked(msg string) string {
	var buf bytes.Buffer
	lenMsg := len(msg)
	for i := 0; i < lenMsg; i++ {
		c := msg[i]
		if c >= spaceByte && c < tildaByte && c != percentByte {
			buf.WriteByte(c)
		} else {
			buf.WriteString(fmt.Sprintf("%%%02X", c))
		}
	}
	return buf.String()
}

// decodeGrpcMessage decodes the msg encoded by encodeGrpcMessage.
func decodeGrpcMessage(msg string) string {
	if msg == "" {
		return ""
	}
	lenMsg := len(msg)
	for i := 0; i < lenMsg; i++ {
		if msg[i] == percentByte && i+2 < lenMsg {
			return decodeGrpcMessageUnchecked(msg)
		}
	}
	return msg
}

func decodeGrpcMessageUnchecked(msg string) string {
	var buf bytes.Buffer
	lenMsg := len(msg)
	for i := 0; i < lenMsg; i++ {
		c := msg[i]
		if c == percentByte && i+2 < lenMsg {
			parsed, err := strconv.ParseInt(msg[i+1:i+3], 16, 8)
			if err != nil {
				buf.WriteByte(c)
			} else {
				buf.WriteByte(byte(parsed))
				i += 2
			}
		} else {
			buf.WriteByte(c)
		}
	}
	return buf.String()
}

type framer struct {
	numWriters int32
	reader     io.Reader
	writer     *bufio.Writer
	fr         *http2.Framer
}

func newFramer(conn net.Conn) *framer {
	f := &framer{
		reader: bufio.NewReaderSize(conn, http2IOBufSize),
		writer: bufio.NewWriterSize(conn, http2IOBufSize),
	}
	f.fr = http2.NewFramer(f.writer, f.reader)
	f.fr.ReadMetaHeaders = hpack.NewDecoder(http2InitHeaderTableSize, nil)
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

func (f *framer) errorDetail() error {
	return f.fr.ErrorDetail()
}
