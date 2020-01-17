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
	"math"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

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
	// baseContentType is the base content-type for gRPC.  This is a valid
	// content-type on it's own, but can also include a content-subtype such as
	// "proto" as a suffix after "+" or ";".  See
	// https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md#requests
	// for more details.
	baseContentType = "application/grpc"
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
		http2.ErrCodeHTTP11Required:     codes.Internal,
	}
	statusCodeConvTab = map[codes.Code]http2.ErrCode{
		codes.Internal:          http2.ErrCodeInternal,
		codes.Canceled:          http2.ErrCodeCancel,
		codes.Unavailable:       http2.ErrCodeRefusedStream,
		codes.ResourceExhausted: http2.ErrCodeEnhanceYourCalm,
		codes.PermissionDenied:  http2.ErrCodeInadequateSecurity,
	}
	// HTTPStatusConvTab is the HTTP status code to gRPC error code conversion table.
	HTTPStatusConvTab = map[int]codes.Code{
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

type parsedHeaderData struct {
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
	mdata          map[string][]string
	statsTags      []byte
	statsTrace     []byte
	contentSubtype string

	// isGRPC field indicates whether the peer is speaking gRPC (otherwise HTTP).
	//
	// We are in gRPC mode (peer speaking gRPC) if:
	// 	* We are client side and have already received a HEADER frame that indicates gRPC peer.
	//  * The header contains valid  a content-type, i.e. a string starts with "application/grpc"
	// And we should handle error specific to gRPC.
	//
	// Otherwise (i.e. a content-type string starts without "application/grpc", or does not exist), we
	// are in HTTP fallback mode, and should handle error specific to HTTP.
	isGRPC         bool
	grpcErr        error
	httpErr        error
	contentTypeErr string
}

// decodeState configures decoding criteria and records the decoded data.
type decodeState struct {
	// whether decoding on server side or not
	serverSide bool

	// Records the states during HPACK decoding. It will be filled with info parsed from HTTP HEADERS
	// frame once decodeHeader function has been invoked and returned.
	data parsedHeaderData
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
		"user-agent",
		"grpc-message-type",
		"grpc-encoding",
		"grpc-message",
		"grpc-status",
		"grpc-timeout",
		"grpc-status-details-bin",
		// Intentionally exclude grpc-previous-rpc-attempts and
		// grpc-retry-pushback-ms, which are "reserved", but their API
		// intentionally works via metadata.
		"te":
		return true
	default:
		return false
	}
}

// isWhitelistedHeader checks whether hdr should be propagated into metadata
// visible to users, even though it is classified as "reserved", above.
func isWhitelistedHeader(hdr string) bool {
	switch hdr {
	case ":authority", "user-agent":
		return true
	default:
		return false
	}
}

// contentSubtype returns the content-subtype for the given content-type.  The
// given content-type must be a valid content-type that starts with
// "application/grpc". A content-subtype will follow "application/grpc" after a
// "+" or ";". See
// https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md#requests for
// more details.
//
// If contentType is not a valid content-type for gRPC, the boolean
// will be false, otherwise true. If content-type == "application/grpc",
// "application/grpc+", or "application/grpc;", the boolean will be true,
// but no content-subtype will be returned.
//
// contentType is assumed to be lowercase already.
func contentSubtype(contentType string) (string, bool) {
	if contentType == baseContentType {
		return "", true
	}
	if !strings.HasPrefix(contentType, baseContentType) {
		return "", false
	}
	// guaranteed since != baseContentType and has baseContentType prefix
	switch contentType[len(baseContentType)] {
	case '+', ';':
		// this will return true for "application/grpc+" or "application/grpc;"
		// which the previous validContentType function tested to be valid, so we
		// just say that no content-subtype is specified in this case
		return contentType[len(baseContentType)+1:], true
	default:
		return "", false
	}
}

// contentSubtype is assumed to be lowercase
func contentType(contentSubtype string) string {
	if contentSubtype == "" {
		return baseContentType
	}
	return baseContentType + "+" + contentSubtype
}

func (d *decodeState) status() *status.Status {
	if d.data.statusGen == nil {
		// No status-details were provided; generate status using code/msg.
		d.data.statusGen = status.New(codes.Code(int32(*(d.data.rawStatusCode))), d.data.rawStatusMsg)
	}
	return d.data.statusGen
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

func (d *decodeState) decodeHeader(frame *http2.MetaHeadersFrame) error {
	// frame.Truncated is set to true when framer detects that the current header
	// list size hits MaxHeaderListSize limit.
	if frame.Truncated {
		return status.Error(codes.Internal, "peer header list size exceeded limit")
	}

	for _, hf := range frame.Fields {
		d.processHeaderField(hf)
	}

	if d.data.isGRPC {
		if d.data.grpcErr != nil {
			return d.data.grpcErr
		}
		if d.serverSide {
			return nil
		}
		if d.data.rawStatusCode == nil && d.data.statusGen == nil {
			// gRPC status doesn't exist.
			// Set rawStatusCode to be unknown and return nil error.
			// So that, if the stream has ended this Unknown status
			// will be propagated to the user.
			// Otherwise, it will be ignored. In which case, status from
			// a later trailer, that has StreamEnded flag set, is propagated.
			code := int(codes.Unknown)
			d.data.rawStatusCode = &code
		}
		return nil
	}

	// HTTP fallback mode
	if d.data.httpErr != nil {
		return d.data.httpErr
	}

	var (
		code = codes.Internal // when header does not include HTTP status, return INTERNAL
		ok   bool
	)

	if d.data.httpStatus != nil {
		code, ok = HTTPStatusConvTab[*(d.data.httpStatus)]
		if !ok {
			code = codes.Unknown
		}
	}

	return status.Error(code, d.constructHTTPErrMsg())
}

// constructErrMsg constructs error message to be returned in HTTP fallback mode.
// Format: HTTP status code and its corresponding message + content-type error message.
func (d *decodeState) constructHTTPErrMsg() string {
	var errMsgs []string

	if d.data.httpStatus == nil {
		errMsgs = append(errMsgs, "malformed header: missing HTTP status")
	} else {
		errMsgs = append(errMsgs, fmt.Sprintf("%s: HTTP status code %d", http.StatusText(*(d.data.httpStatus)), *d.data.httpStatus))
	}

	if d.data.contentTypeErr == "" {
		errMsgs = append(errMsgs, "transport: missing content-type field")
	} else {
		errMsgs = append(errMsgs, d.data.contentTypeErr)
	}

	return strings.Join(errMsgs, "; ")
}

func (d *decodeState) addMetadata(k, v string) {
	if d.data.mdata == nil {
		d.data.mdata = make(map[string][]string)
	}
	d.data.mdata[k] = append(d.data.mdata[k], v)
}

func (d *decodeState) processHeaderField(f hpack.HeaderField) {
	switch f.Name {
	case "content-type":
		contentSubtype, validContentType := contentSubtype(f.Value)
		if !validContentType {
			d.data.contentTypeErr = fmt.Sprintf("transport: received the unexpected content-type %q", f.Value)
			return
		}
		d.data.contentSubtype = contentSubtype
		// TODO: do we want to propagate the whole content-type in the metadata,
		// or come up with a way to just propagate the content-subtype if it was set?
		// ie {"content-type": "application/grpc+proto"} or {"content-subtype": "proto"}
		// in the metadata?
		d.addMetadata(f.Name, f.Value)
		d.data.isGRPC = true
	case "grpc-encoding":
		d.data.encoding = f.Value
	case "grpc-status":
		code, err := strconv.Atoi(f.Value)
		if err != nil {
			d.data.grpcErr = status.Errorf(codes.Internal, "transport: malformed grpc-status: %v", err)
			return
		}
		d.data.rawStatusCode = &code
	case "grpc-message":
		d.data.rawStatusMsg = decodeGrpcMessage(f.Value)
	case "grpc-status-details-bin":
		v, err := decodeBinHeader(f.Value)
		if err != nil {
			d.data.grpcErr = status.Errorf(codes.Internal, "transport: malformed grpc-status-details-bin: %v", err)
			return
		}
		s := &spb.Status{}
		if err := proto.Unmarshal(v, s); err != nil {
			d.data.grpcErr = status.Errorf(codes.Internal, "transport: malformed grpc-status-details-bin: %v", err)
			return
		}
		d.data.statusGen = status.FromProto(s)
	case "grpc-timeout":
		d.data.timeoutSet = true
		var err error
		if d.data.timeout, err = decodeTimeout(f.Value); err != nil {
			d.data.grpcErr = status.Errorf(codes.Internal, "transport: malformed time-out: %v", err)
		}
	case ":path":
		d.data.method = f.Value
	case ":status":
		code, err := strconv.Atoi(f.Value)
		if err != nil {
			d.data.httpErr = status.Errorf(codes.Internal, "transport: malformed http-status: %v", err)
			return
		}
		d.data.httpStatus = &code
	case "grpc-tags-bin":
		v, err := decodeBinHeader(f.Value)
		if err != nil {
			d.data.grpcErr = status.Errorf(codes.Internal, "transport: malformed grpc-tags-bin: %v", err)
			return
		}
		d.data.statsTags = v
		d.addMetadata(f.Name, string(v))
	case "grpc-trace-bin":
		v, err := decodeBinHeader(f.Value)
		if err != nil {
			d.data.grpcErr = status.Errorf(codes.Internal, "transport: malformed grpc-trace-bin: %v", err)
			return
		}
		d.data.statsTrace = v
		d.addMetadata(f.Name, string(v))
	default:
		if isReservedHeader(f.Name) && !isWhitelistedHeader(f.Name) {
			break
		}
		v, err := decodeMetadataHeader(f.Name, f.Value)
		if err != nil {
			errorf("Failed to decode metadata header (%q, %q): %v", f.Name, f.Value, err)
			return
		}
		d.addMetadata(f.Name, v)
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
	if size > 9 {
		// Spec allows for 8 digits plus the unit.
		return 0, fmt.Errorf("transport: timeout string is too long: %q", s)
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
	const maxHours = math.MaxInt64 / int64(time.Hour)
	if d == time.Hour && t > maxHours {
		// This timeout would overflow math.MaxInt64; clamp it.
		return time.Duration(math.MaxInt64), nil
	}
	return d * time.Duration(t), nil
}

const (
	spaceByte   = ' '
	tildeByte   = '~'
	percentByte = '%'
)

// encodeGrpcMessage is used to encode status code in header field
// "grpc-message". It does percent encoding and also replaces invalid utf-8
// characters with Unicode replacement character.
//
// It checks to see if each individual byte in msg is an allowable byte, and
// then either percent encoding or passing it through. When percent encoding,
// the byte is converted into hexadecimal notation with a '%' prepended.
func encodeGrpcMessage(msg string) string {
	if msg == "" {
		return ""
	}
	lenMsg := len(msg)
	for i := 0; i < lenMsg; i++ {
		c := msg[i]
		if !(c >= spaceByte && c <= tildeByte && c != percentByte) {
			return encodeGrpcMessageUnchecked(msg)
		}
	}
	return msg
}

func encodeGrpcMessageUnchecked(msg string) string {
	var buf bytes.Buffer
	for len(msg) > 0 {
		r, size := utf8.DecodeRuneInString(msg)
		for _, b := range []byte(string(r)) {
			if size > 1 {
				// If size > 1, r is not ascii. Always do percent encoding.
				buf.WriteString(fmt.Sprintf("%%%02X", b))
				continue
			}

			// The for loop is necessary even if size == 1. r could be
			// utf8.RuneError.
			//
			// fmt.Sprintf("%%%02X", utf8.RuneError) gives "%FFFD".
			if b >= spaceByte && b <= tildeByte && b != percentByte {
				buf.WriteByte(b)
			} else {
				buf.WriteString(fmt.Sprintf("%%%02X", b))
			}
		}
		msg = msg[size:]
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

type bufWriter struct {
	buf       []byte
	offset    int
	batchSize int
	conn      net.Conn
	err       error

	onFlush func()
}

func newBufWriter(conn net.Conn, batchSize int) *bufWriter {
	return &bufWriter{
		buf:       make([]byte, batchSize*2),
		batchSize: batchSize,
		conn:      conn,
	}
}

func (w *bufWriter) Write(b []byte) (n int, err error) {
	if w.err != nil {
		return 0, w.err
	}
	if w.batchSize == 0 { // Buffer has been disabled.
		return w.conn.Write(b)
	}
	for len(b) > 0 {
		nn := copy(w.buf[w.offset:], b)
		b = b[nn:]
		w.offset += nn
		n += nn
		if w.offset >= w.batchSize {
			err = w.Flush()
		}
	}
	return n, err
}

func (w *bufWriter) Flush() error {
	if w.err != nil {
		return w.err
	}
	if w.offset == 0 {
		return nil
	}
	if w.onFlush != nil {
		w.onFlush()
	}
	_, w.err = w.conn.Write(w.buf[:w.offset])
	w.offset = 0
	return w.err
}

type framer struct {
	writer *bufWriter
	fr     *http2.Framer
}

func newFramer(conn net.Conn, writeBufferSize, readBufferSize int, maxHeaderListSize uint32) *framer {
	if writeBufferSize < 0 {
		writeBufferSize = 0
	}
	var r io.Reader = conn
	if readBufferSize > 0 {
		r = bufio.NewReaderSize(r, readBufferSize)
	}
	w := newBufWriter(conn, writeBufferSize)
	f := &framer{
		writer: w,
		fr:     http2.NewFramer(w, r),
	}
	// Opt-in to Frame reuse API on framer to reduce garbage.
	// Frames aren't safe to read from after a subsequent call to ReadFrame.
	f.fr.SetReuseFrames()
	f.fr.MaxHeaderListSize = maxHeaderListSize
	f.fr.ReadMetaHeaders = hpack.NewDecoder(http2InitHeaderTableSize, nil)
	return f
}
