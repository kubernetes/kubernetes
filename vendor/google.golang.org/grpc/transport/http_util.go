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

package transport

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
	spb "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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
	httpStatusConvTab = map[int]codes.Code{
		// 400 Bad Request - INTERNAL.
		http.StatusBadRequest: codes.Internal,
		// 401 Unauthorized  - UNAUTHENTICATED.
		http.StatusUnauthorized: codes.Unauthenticated,
		// 403 Forbidden - PERMISSION_DENIED.
		http.StatusForbidden: codes.PermissionDenied,
		// 404 Not Found - UNIMPLEMENTED.
		http.StatusNotFound: codes.Unimplemented,
		// 429 Too Many Requests - UNAVAILABLE.
		http.StatusTooManyRequests: codes.Unavailable,
		// 502 Bad Gateway - UNAVAILABLE.
		http.StatusBadGateway: codes.Unavailable,
		// 503 Service Unavailable - UNAVAILABLE.
		http.StatusServiceUnavailable: codes.Unavailable,
		// 504 Gateway timeout - UNAVAILABLE.
		http.StatusGatewayTimeout: codes.Unavailable,
	}
)

// Records the states during HPACK decoding. Must be reset once the
// decoding of the entire headers are finished.
type decodeState struct {
	encoding string
	// statusGen caches the stream status received from the trailer the server
	// sent.  Client side only.  Do not access directly.  After all trailers are
	// parsed, use the status method to retrieve the status.
	statusGen *status.Status
	// rawStatusCode and rawStatusMsg are set from the raw trailer fields and are not
	// intended for direct access outside of parsing.
	rawStatusCode *int
	rawStatusMsg  string
	httpStatus    *int
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
		"grpc-status-details-bin",
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

func (d *decodeState) status() *status.Status {
	if d.statusGen == nil {
		// No status-details were provided; generate status using code/msg.
		d.statusGen = status.New(codes.Code(int32(*(d.rawStatusCode))), d.rawStatusMsg)
	}
	return d.statusGen
}

const binHdrSuffix = "-bin"

func encodeBinHeader(v []byte) string {
	return base64.RawStdEncoding.EncodeToString(v)
}

func decodeBinHeader(v string) ([]byte, error) {
	if len(v)%4 == 0 {
		// Input was padded, or padding was not necessary.
		return base64.StdEncoding.DecodeString(v)
	}
	return base64.RawStdEncoding.DecodeString(v)
}

func encodeMetadataHeader(k, v string) string {
	if strings.HasSuffix(k, binHdrSuffix) {
		return encodeBinHeader(([]byte)(v))
	}
	return v
}

func decodeMetadataHeader(k, v string) (string, error) {
	if strings.HasSuffix(k, binHdrSuffix) {
		b, err := decodeBinHeader(v)
		return string(b), err
	}
	return v, nil
}

func (d *decodeState) decodeResponseHeader(frame *http2.MetaHeadersFrame) error {
	for _, hf := range frame.Fields {
		if err := d.processHeaderField(hf); err != nil {
			return err
		}
	}

	// If grpc status exists, no need to check further.
	if d.rawStatusCode != nil || d.statusGen != nil {
		return nil
	}

	// If grpc status doesn't exist and http status doesn't exist,
	// then it's a malformed header.
	if d.httpStatus == nil {
		return streamErrorf(codes.Internal, "malformed header: doesn't contain status(gRPC or HTTP)")
	}

	if *(d.httpStatus) != http.StatusOK {
		code, ok := httpStatusConvTab[*(d.httpStatus)]
		if !ok {
			code = codes.Unknown
		}
		return streamErrorf(code, http.StatusText(*(d.httpStatus)))
	}

	// gRPC status doesn't exist and http status is OK.
	// Set rawStatusCode to be unknown and return nil error.
	// So that, if the stream has ended this Unknown status
	// will be propogated to the user.
	// Otherwise, it will be ignored. In which case, status from
	// a later trailer, that has StreamEnded flag set, is propogated.
	code := int(codes.Unknown)
	d.rawStatusCode = &code
	return nil

}

func (d *decodeState) processHeaderField(f hpack.HeaderField) error {
	switch f.Name {
	case "content-type":
		if !validContentType(f.Value) {
			return streamErrorf(codes.FailedPrecondition, "transport: received the unexpected content-type %q", f.Value)
		}
	case "grpc-encoding":
		d.encoding = f.Value
	case "grpc-status":
		code, err := strconv.Atoi(f.Value)
		if err != nil {
			return streamErrorf(codes.Internal, "transport: malformed grpc-status: %v", err)
		}
		d.rawStatusCode = &code
	case "grpc-message":
		d.rawStatusMsg = decodeGrpcMessage(f.Value)
	case "grpc-status-details-bin":
		v, err := decodeBinHeader(f.Value)
		if err != nil {
			return streamErrorf(codes.Internal, "transport: malformed grpc-status-details-bin: %v", err)
		}
		s := &spb.Status{}
		if err := proto.Unmarshal(v, s); err != nil {
			return streamErrorf(codes.Internal, "transport: malformed grpc-status-details-bin: %v", err)
		}
		d.statusGen = status.FromProto(s)
	case "grpc-timeout":
		d.timeoutSet = true
		var err error
		if d.timeout, err = decodeTimeout(f.Value); err != nil {
			return streamErrorf(codes.Internal, "transport: malformed time-out: %v", err)
		}
	case ":path":
		d.method = f.Value
	case ":status":
		code, err := strconv.Atoi(f.Value)
		if err != nil {
			return streamErrorf(codes.Internal, "transport: malformed http-status: %v", err)
		}
		d.httpStatus = &code
	default:
		if !isReservedHeader(f.Name) || isWhitelistedPseudoHeader(f.Name) {
			if d.mdata == nil {
				d.mdata = make(map[string][]string)
			}
			v, err := decodeMetadataHeader(f.Name, f.Value)
			if err != nil {
				errorf("Failed to decode metadata header (%q, %q): %v", f.Name, f.Value, err)
				return nil
			}
			d.mdata[f.Name] = append(d.mdata[f.Name], v)
		}
	}
	return nil
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
	if t <= 0 {
		return "0n"
	}
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
			parsed, err := strconv.ParseUint(msg[i+1:i+3], 16, 8)
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
	// Opt-in to Frame reuse API on framer to reduce garbage.
	// Frames aren't safe to read from after a subsequent call to ReadFrame.
	f.fr.SetReuseFrames()
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
