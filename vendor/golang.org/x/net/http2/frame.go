// Copyright 2014 The Go Authors.
// See https://code.google.com/p/go/source/browse/CONTRIBUTORS
// Licensed under the same terms as Go itself:
// https://code.google.com/p/go/source/browse/LICENSE

package http2

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync"
)

const frameHeaderLen = 9

var padZeros = make([]byte, 255) // zeros for padding

// A FrameType is a registered frame type as defined in
// http://http2.github.io/http2-spec/#rfc.section.11.2
type FrameType uint8

const (
	FrameData         FrameType = 0x0
	FrameHeaders      FrameType = 0x1
	FramePriority     FrameType = 0x2
	FrameRSTStream    FrameType = 0x3
	FrameSettings     FrameType = 0x4
	FramePushPromise  FrameType = 0x5
	FramePing         FrameType = 0x6
	FrameGoAway       FrameType = 0x7
	FrameWindowUpdate FrameType = 0x8
	FrameContinuation FrameType = 0x9
)

var frameName = map[FrameType]string{
	FrameData:         "DATA",
	FrameHeaders:      "HEADERS",
	FramePriority:     "PRIORITY",
	FrameRSTStream:    "RST_STREAM",
	FrameSettings:     "SETTINGS",
	FramePushPromise:  "PUSH_PROMISE",
	FramePing:         "PING",
	FrameGoAway:       "GOAWAY",
	FrameWindowUpdate: "WINDOW_UPDATE",
	FrameContinuation: "CONTINUATION",
}

func (t FrameType) String() string {
	if s, ok := frameName[t]; ok {
		return s
	}
	return fmt.Sprintf("UNKNOWN_FRAME_TYPE_%d", uint8(t))
}

// Flags is a bitmask of HTTP/2 flags.
// The meaning of flags varies depending on the frame type.
type Flags uint8

// Has reports whether f contains all (0 or more) flags in v.
func (f Flags) Has(v Flags) bool {
	return (f & v) == v
}

// Frame-specific FrameHeader flag bits.
const (
	// Data Frame
	FlagDataEndStream Flags = 0x1
	FlagDataPadded    Flags = 0x8

	// Headers Frame
	FlagHeadersEndStream  Flags = 0x1
	FlagHeadersEndHeaders Flags = 0x4
	FlagHeadersPadded     Flags = 0x8
	FlagHeadersPriority   Flags = 0x20

	// Settings Frame
	FlagSettingsAck Flags = 0x1

	// Ping Frame
	FlagPingAck Flags = 0x1

	// Continuation Frame
	FlagContinuationEndHeaders Flags = 0x4

	FlagPushPromiseEndHeaders Flags = 0x4
	FlagPushPromisePadded     Flags = 0x8
)

var flagName = map[FrameType]map[Flags]string{
	FrameData: {
		FlagDataEndStream: "END_STREAM",
		FlagDataPadded:    "PADDED",
	},
	FrameHeaders: {
		FlagHeadersEndStream:  "END_STREAM",
		FlagHeadersEndHeaders: "END_HEADERS",
		FlagHeadersPadded:     "PADDED",
		FlagHeadersPriority:   "PRIORITY",
	},
	FrameSettings: {
		FlagSettingsAck: "ACK",
	},
	FramePing: {
		FlagPingAck: "ACK",
	},
	FrameContinuation: {
		FlagContinuationEndHeaders: "END_HEADERS",
	},
	FramePushPromise: {
		FlagPushPromiseEndHeaders: "END_HEADERS",
		FlagPushPromisePadded:     "PADDED",
	},
}

// a frameParser parses a frame given its FrameHeader and payload
// bytes. The length of payload will always equal fh.Length (which
// might be 0).
type frameParser func(fh FrameHeader, payload []byte) (Frame, error)

var frameParsers = map[FrameType]frameParser{
	FrameData:         parseDataFrame,
	FrameHeaders:      parseHeadersFrame,
	FramePriority:     parsePriorityFrame,
	FrameRSTStream:    parseRSTStreamFrame,
	FrameSettings:     parseSettingsFrame,
	FramePushPromise:  parsePushPromise,
	FramePing:         parsePingFrame,
	FrameGoAway:       parseGoAwayFrame,
	FrameWindowUpdate: parseWindowUpdateFrame,
	FrameContinuation: parseContinuationFrame,
}

func typeFrameParser(t FrameType) frameParser {
	if f := frameParsers[t]; f != nil {
		return f
	}
	return parseUnknownFrame
}

// A FrameHeader is the 9 byte header of all HTTP/2 frames.
//
// See http://http2.github.io/http2-spec/#FrameHeader
type FrameHeader struct {
	valid bool // caller can access []byte fields in the Frame

	// Type is the 1 byte frame type. There are ten standard frame
	// types, but extension frame types may be written by WriteRawFrame
	// and will be returned by ReadFrame (as UnknownFrame).
	Type FrameType

	// Flags are the 1 byte of 8 potential bit flags per frame.
	// They are specific to the frame type.
	Flags Flags

	// Length is the length of the frame, not including the 9 byte header.
	// The maximum size is one byte less than 16MB (uint24), but only
	// frames up to 16KB are allowed without peer agreement.
	Length uint32

	// StreamID is which stream this frame is for. Certain frames
	// are not stream-specific, in which case this field is 0.
	StreamID uint32
}

// Header returns h. It exists so FrameHeaders can be embedded in other
// specific frame types and implement the Frame interface.
func (h FrameHeader) Header() FrameHeader { return h }

func (h FrameHeader) String() string {
	var buf bytes.Buffer
	buf.WriteString("[FrameHeader ")
	buf.WriteString(h.Type.String())
	if h.Flags != 0 {
		buf.WriteString(" flags=")
		set := 0
		for i := uint8(0); i < 8; i++ {
			if h.Flags&(1<<i) == 0 {
				continue
			}
			set++
			if set > 1 {
				buf.WriteByte('|')
			}
			name := flagName[h.Type][Flags(1<<i)]
			if name != "" {
				buf.WriteString(name)
			} else {
				fmt.Fprintf(&buf, "0x%x", 1<<i)
			}
		}
	}
	if h.StreamID != 0 {
		fmt.Fprintf(&buf, " stream=%d", h.StreamID)
	}
	fmt.Fprintf(&buf, " len=%d]", h.Length)
	return buf.String()
}

func (h *FrameHeader) checkValid() {
	if !h.valid {
		panic("Frame accessor called on non-owned Frame")
	}
}

func (h *FrameHeader) invalidate() { h.valid = false }

// frame header bytes.
// Used only by ReadFrameHeader.
var fhBytes = sync.Pool{
	New: func() interface{} {
		buf := make([]byte, frameHeaderLen)
		return &buf
	},
}

// ReadFrameHeader reads 9 bytes from r and returns a FrameHeader.
// Most users should use Framer.ReadFrame instead.
func ReadFrameHeader(r io.Reader) (FrameHeader, error) {
	bufp := fhBytes.Get().(*[]byte)
	defer fhBytes.Put(bufp)
	return readFrameHeader(*bufp, r)
}

func readFrameHeader(buf []byte, r io.Reader) (FrameHeader, error) {
	_, err := io.ReadFull(r, buf[:frameHeaderLen])
	if err != nil {
		return FrameHeader{}, err
	}
	return FrameHeader{
		Length:   (uint32(buf[0])<<16 | uint32(buf[1])<<8 | uint32(buf[2])),
		Type:     FrameType(buf[3]),
		Flags:    Flags(buf[4]),
		StreamID: binary.BigEndian.Uint32(buf[5:]) & (1<<31 - 1),
		valid:    true,
	}, nil
}

// A Frame is the base interface implemented by all frame types.
// Callers will generally type-assert the specific frame type:
// *HeadersFrame, *SettingsFrame, *WindowUpdateFrame, etc.
//
// Frames are only valid until the next call to Framer.ReadFrame.
type Frame interface {
	Header() FrameHeader

	// invalidate is called by Framer.ReadFrame to make this
	// frame's buffers as being invalid, since the subsequent
	// frame will reuse them.
	invalidate()
}

// A Framer reads and writes Frames.
type Framer struct {
	r         io.Reader
	lastFrame Frame

	maxReadSize uint32
	headerBuf   [frameHeaderLen]byte

	// TODO: let getReadBuf be configurable, and use a less memory-pinning
	// allocator in server.go to minimize memory pinned for many idle conns.
	// Will probably also need to make frame invalidation have a hook too.
	getReadBuf func(size uint32) []byte
	readBuf    []byte // cache for default getReadBuf

	maxWriteSize uint32 // zero means unlimited; TODO: implement

	w    io.Writer
	wbuf []byte

	// AllowIllegalWrites permits the Framer's Write methods to
	// write frames that do not conform to the HTTP/2 spec.  This
	// permits using the Framer to test other HTTP/2
	// implementations' conformance to the spec.
	// If false, the Write methods will prefer to return an error
	// rather than comply.
	AllowIllegalWrites bool

	// TODO: track which type of frame & with which flags was sent
	// last.  Then return an error (unless AllowIllegalWrites) if
	// we're in the middle of a header block and a
	// non-Continuation or Continuation on a different stream is
	// attempted to be written.
}

func (f *Framer) startWrite(ftype FrameType, flags Flags, streamID uint32) {
	// Write the FrameHeader.
	f.wbuf = append(f.wbuf[:0],
		0, // 3 bytes of length, filled in in endWrite
		0,
		0,
		byte(ftype),
		byte(flags),
		byte(streamID>>24),
		byte(streamID>>16),
		byte(streamID>>8),
		byte(streamID))
}

func (f *Framer) endWrite() error {
	// Now that we know the final size, fill in the FrameHeader in
	// the space previously reserved for it. Abuse append.
	length := len(f.wbuf) - frameHeaderLen
	if length >= (1 << 24) {
		return ErrFrameTooLarge
	}
	_ = append(f.wbuf[:0],
		byte(length>>16),
		byte(length>>8),
		byte(length))
	n, err := f.w.Write(f.wbuf)
	if err == nil && n != len(f.wbuf) {
		err = io.ErrShortWrite
	}
	return err
}

func (f *Framer) writeByte(v byte)     { f.wbuf = append(f.wbuf, v) }
func (f *Framer) writeBytes(v []byte)  { f.wbuf = append(f.wbuf, v...) }
func (f *Framer) writeUint16(v uint16) { f.wbuf = append(f.wbuf, byte(v>>8), byte(v)) }
func (f *Framer) writeUint32(v uint32) {
	f.wbuf = append(f.wbuf, byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}

const (
	minMaxFrameSize = 1 << 14
	maxFrameSize    = 1<<24 - 1
)

// NewFramer returns a Framer that writes frames to w and reads them from r.
func NewFramer(w io.Writer, r io.Reader) *Framer {
	fr := &Framer{
		w: w,
		r: r,
	}
	fr.getReadBuf = func(size uint32) []byte {
		if cap(fr.readBuf) >= int(size) {
			return fr.readBuf[:size]
		}
		fr.readBuf = make([]byte, size)
		return fr.readBuf
	}
	fr.SetMaxReadFrameSize(maxFrameSize)
	return fr
}

// SetMaxReadFrameSize sets the maximum size of a frame
// that will be read by a subsequent call to ReadFrame.
// It is the caller's responsibility to advertise this
// limit with a SETTINGS frame.
func (fr *Framer) SetMaxReadFrameSize(v uint32) {
	if v > maxFrameSize {
		v = maxFrameSize
	}
	fr.maxReadSize = v
}

// ErrFrameTooLarge is returned from Framer.ReadFrame when the peer
// sends a frame that is larger than declared with SetMaxReadFrameSize.
var ErrFrameTooLarge = errors.New("http2: frame too large")

// ReadFrame reads a single frame. The returned Frame is only valid
// until the next call to ReadFrame.
// If the frame is larger than previously set with SetMaxReadFrameSize,
// the returned error is ErrFrameTooLarge.
func (fr *Framer) ReadFrame() (Frame, error) {
	if fr.lastFrame != nil {
		fr.lastFrame.invalidate()
	}
	fh, err := readFrameHeader(fr.headerBuf[:], fr.r)
	if err != nil {
		return nil, err
	}
	if fh.Length > fr.maxReadSize {
		return nil, ErrFrameTooLarge
	}
	payload := fr.getReadBuf(fh.Length)
	if _, err := io.ReadFull(fr.r, payload); err != nil {
		return nil, err
	}
	f, err := typeFrameParser(fh.Type)(fh, payload)
	if err != nil {
		return nil, err
	}
	fr.lastFrame = f
	return f, nil
}

// A DataFrame conveys arbitrary, variable-length sequences of octets
// associated with a stream.
// See http://http2.github.io/http2-spec/#rfc.section.6.1
type DataFrame struct {
	FrameHeader
	data []byte
}

func (f *DataFrame) StreamEnded() bool {
	return f.FrameHeader.Flags.Has(FlagDataEndStream)
}

// Data returns the frame's data octets, not including any padding
// size byte or padding suffix bytes.
// The caller must not retain the returned memory past the next
// call to ReadFrame.
func (f *DataFrame) Data() []byte {
	f.checkValid()
	return f.data
}

func parseDataFrame(fh FrameHeader, payload []byte) (Frame, error) {
	if fh.StreamID == 0 {
		// DATA frames MUST be associated with a stream. If a
		// DATA frame is received whose stream identifier
		// field is 0x0, the recipient MUST respond with a
		// connection error (Section 5.4.1) of type
		// PROTOCOL_ERROR.
		return nil, ConnectionError(ErrCodeProtocol)
	}
	f := &DataFrame{
		FrameHeader: fh,
	}
	var padSize byte
	if fh.Flags.Has(FlagDataPadded) {
		var err error
		payload, padSize, err = readByte(payload)
		if err != nil {
			return nil, err
		}
	}
	if int(padSize) > len(payload) {
		// If the length of the padding is greater than the
		// length of the frame payload, the recipient MUST
		// treat this as a connection error.
		// Filed: https://github.com/http2/http2-spec/issues/610
		return nil, ConnectionError(ErrCodeProtocol)
	}
	f.data = payload[:len(payload)-int(padSize)]
	return f, nil
}

var errStreamID = errors.New("invalid streamid")

func validStreamID(streamID uint32) bool {
	return streamID != 0 && streamID&(1<<31) == 0
}

// WriteData writes a DATA frame.
//
// It will perform exactly one Write to the underlying Writer.
// It is the caller's responsibility to not call other Write methods concurrently.
func (f *Framer) WriteData(streamID uint32, endStream bool, data []byte) error {
	// TODO: ignoring padding for now. will add when somebody cares.
	if !validStreamID(streamID) && !f.AllowIllegalWrites {
		return errStreamID
	}
	var flags Flags
	if endStream {
		flags |= FlagDataEndStream
	}
	f.startWrite(FrameData, flags, streamID)
	f.wbuf = append(f.wbuf, data...)
	return f.endWrite()
}

// A SettingsFrame conveys configuration parameters that affect how
// endpoints communicate, such as preferences and constraints on peer
// behavior.
//
// See http://http2.github.io/http2-spec/#SETTINGS
type SettingsFrame struct {
	FrameHeader
	p []byte
}

func parseSettingsFrame(fh FrameHeader, p []byte) (Frame, error) {
	if fh.Flags.Has(FlagSettingsAck) && fh.Length > 0 {
		// When this (ACK 0x1) bit is set, the payload of the
		// SETTINGS frame MUST be empty.  Receipt of a
		// SETTINGS frame with the ACK flag set and a length
		// field value other than 0 MUST be treated as a
		// connection error (Section 5.4.1) of type
		// FRAME_SIZE_ERROR.
		return nil, ConnectionError(ErrCodeFrameSize)
	}
	if fh.StreamID != 0 {
		// SETTINGS frames always apply to a connection,
		// never a single stream.  The stream identifier for a
		// SETTINGS frame MUST be zero (0x0).  If an endpoint
		// receives a SETTINGS frame whose stream identifier
		// field is anything other than 0x0, the endpoint MUST
		// respond with a connection error (Section 5.4.1) of
		// type PROTOCOL_ERROR.
		return nil, ConnectionError(ErrCodeProtocol)
	}
	if len(p)%6 != 0 {
		// Expecting even number of 6 byte settings.
		return nil, ConnectionError(ErrCodeFrameSize)
	}
	f := &SettingsFrame{FrameHeader: fh, p: p}
	if v, ok := f.Value(SettingInitialWindowSize); ok && v > (1<<31)-1 {
		// Values above the maximum flow control window size of 2^31 - 1 MUST
		// be treated as a connection error (Section 5.4.1) of type
		// FLOW_CONTROL_ERROR.
		return nil, ConnectionError(ErrCodeFlowControl)
	}
	return f, nil
}

func (f *SettingsFrame) IsAck() bool {
	return f.FrameHeader.Flags.Has(FlagSettingsAck)
}

func (f *SettingsFrame) Value(s SettingID) (v uint32, ok bool) {
	f.checkValid()
	buf := f.p
	for len(buf) > 0 {
		settingID := SettingID(binary.BigEndian.Uint16(buf[:2]))
		if settingID == s {
			return binary.BigEndian.Uint32(buf[2:6]), true
		}
		buf = buf[6:]
	}
	return 0, false
}

// ForeachSetting runs fn for each setting.
// It stops and returns the first error.
func (f *SettingsFrame) ForeachSetting(fn func(Setting) error) error {
	f.checkValid()
	buf := f.p
	for len(buf) > 0 {
		if err := fn(Setting{
			SettingID(binary.BigEndian.Uint16(buf[:2])),
			binary.BigEndian.Uint32(buf[2:6]),
		}); err != nil {
			return err
		}
		buf = buf[6:]
	}
	return nil
}

// WriteSettings writes a SETTINGS frame with zero or more settings
// specified and the ACK bit not set.
//
// It will perform exactly one Write to the underlying Writer.
// It is the caller's responsibility to not call other Write methods concurrently.
func (f *Framer) WriteSettings(settings ...Setting) error {
	f.startWrite(FrameSettings, 0, 0)
	for _, s := range settings {
		f.writeUint16(uint16(s.ID))
		f.writeUint32(s.Val)
	}
	return f.endWrite()
}

// WriteSettings writes an empty SETTINGS frame with the ACK bit set.
//
// It will perform exactly one Write to the underlying Writer.
// It is the caller's responsibility to not call other Write methods concurrently.
func (f *Framer) WriteSettingsAck() error {
	f.startWrite(FrameSettings, FlagSettingsAck, 0)
	return f.endWrite()
}

// A PingFrame is a mechanism for measuring a minimal round trip time
// from the sender, as well as determining whether an idle connection
// is still functional.
// See http://http2.github.io/http2-spec/#rfc.section.6.7
type PingFrame struct {
	FrameHeader
	Data [8]byte
}

func parsePingFrame(fh FrameHeader, payload []byte) (Frame, error) {
	if len(payload) != 8 {
		return nil, ConnectionError(ErrCodeFrameSize)
	}
	if fh.StreamID != 0 {
		return nil, ConnectionError(ErrCodeProtocol)
	}
	f := &PingFrame{FrameHeader: fh}
	copy(f.Data[:], payload)
	return f, nil
}

func (f *Framer) WritePing(ack bool, data [8]byte) error {
	var flags Flags
	if ack {
		flags = FlagPingAck
	}
	f.startWrite(FramePing, flags, 0)
	f.writeBytes(data[:])
	return f.endWrite()
}

// A GoAwayFrame informs the remote peer to stop creating streams on this connection.
// See http://http2.github.io/http2-spec/#rfc.section.6.8
type GoAwayFrame struct {
	FrameHeader
	LastStreamID uint32
	ErrCode      ErrCode
	debugData    []byte
}

// DebugData returns any debug data in the GOAWAY frame. Its contents
// are not defined.
// The caller must not retain the returned memory past the next
// call to ReadFrame.
func (f *GoAwayFrame) DebugData() []byte {
	f.checkValid()
	return f.debugData
}

func parseGoAwayFrame(fh FrameHeader, p []byte) (Frame, error) {
	if fh.StreamID != 0 {
		return nil, ConnectionError(ErrCodeProtocol)
	}
	if len(p) < 8 {
		return nil, ConnectionError(ErrCodeFrameSize)
	}
	return &GoAwayFrame{
		FrameHeader:  fh,
		LastStreamID: binary.BigEndian.Uint32(p[:4]) & (1<<31 - 1),
		ErrCode:      ErrCode(binary.BigEndian.Uint32(p[4:8])),
		debugData:    p[8:],
	}, nil
}

func (f *Framer) WriteGoAway(maxStreamID uint32, code ErrCode, debugData []byte) error {
	f.startWrite(FrameGoAway, 0, 0)
	f.writeUint32(maxStreamID & (1<<31 - 1))
	f.writeUint32(uint32(code))
	f.writeBytes(debugData)
	return f.endWrite()
}

// An UnknownFrame is the frame type returned when the frame type is unknown
// or no specific frame type parser exists.
type UnknownFrame struct {
	FrameHeader
	p []byte
}

// Payload returns the frame's payload (after the header).  It is not
// valid to call this method after a subsequent call to
// Framer.ReadFrame, nor is it valid to retain the returned slice.
// The memory is owned by the Framer and is invalidated when the next
// frame is read.
func (f *UnknownFrame) Payload() []byte {
	f.checkValid()
	return f.p
}

func parseUnknownFrame(fh FrameHeader, p []byte) (Frame, error) {
	return &UnknownFrame{fh, p}, nil
}

// A WindowUpdateFrame is used to implement flow control.
// See http://http2.github.io/http2-spec/#rfc.section.6.9
type WindowUpdateFrame struct {
	FrameHeader
	Increment uint32
}

func parseWindowUpdateFrame(fh FrameHeader, p []byte) (Frame, error) {
	if len(p) != 4 {
		return nil, ConnectionError(ErrCodeFrameSize)
	}
	inc := binary.BigEndian.Uint32(p[:4]) & 0x7fffffff // mask off high reserved bit
	if inc == 0 {
		// A receiver MUST treat the receipt of a
		// WINDOW_UPDATE frame with an flow control window
		// increment of 0 as a stream error (Section 5.4.2) of
		// type PROTOCOL_ERROR; errors on the connection flow
		// control window MUST be treated as a connection
		// error (Section 5.4.1).
		if fh.StreamID == 0 {
			return nil, ConnectionError(ErrCodeProtocol)
		}
		return nil, StreamError{fh.StreamID, ErrCodeProtocol}
	}
	return &WindowUpdateFrame{
		FrameHeader: fh,
		Increment:   inc,
	}, nil
}

// WriteWindowUpdate writes a WINDOW_UPDATE frame.
// The increment value must be between 1 and 2,147,483,647, inclusive.
// If the Stream ID is zero, the window update applies to the
// connection as a whole.
func (f *Framer) WriteWindowUpdate(streamID, incr uint32) error {
	// "The legal range for the increment to the flow control window is 1 to 2^31-1 (2,147,483,647) octets."
	if (incr < 1 || incr > 2147483647) && !f.AllowIllegalWrites {
		return errors.New("illegal window increment value")
	}
	f.startWrite(FrameWindowUpdate, 0, streamID)
	f.writeUint32(incr)
	return f.endWrite()
}

// A HeadersFrame is used to open a stream and additionally carries a
// header block fragment.
type HeadersFrame struct {
	FrameHeader

	// Priority is set if FlagHeadersPriority is set in the FrameHeader.
	Priority PriorityParam

	headerFragBuf []byte // not owned
}

func (f *HeadersFrame) HeaderBlockFragment() []byte {
	f.checkValid()
	return f.headerFragBuf
}

func (f *HeadersFrame) HeadersEnded() bool {
	return f.FrameHeader.Flags.Has(FlagHeadersEndHeaders)
}

func (f *HeadersFrame) StreamEnded() bool {
	return f.FrameHeader.Flags.Has(FlagHeadersEndStream)
}

func (f *HeadersFrame) HasPriority() bool {
	return f.FrameHeader.Flags.Has(FlagHeadersPriority)
}

func parseHeadersFrame(fh FrameHeader, p []byte) (_ Frame, err error) {
	hf := &HeadersFrame{
		FrameHeader: fh,
	}
	if fh.StreamID == 0 {
		// HEADERS frames MUST be associated with a stream.  If a HEADERS frame
		// is received whose stream identifier field is 0x0, the recipient MUST
		// respond with a connection error (Section 5.4.1) of type
		// PROTOCOL_ERROR.
		return nil, ConnectionError(ErrCodeProtocol)
	}
	var padLength uint8
	if fh.Flags.Has(FlagHeadersPadded) {
		if p, padLength, err = readByte(p); err != nil {
			return
		}
	}
	if fh.Flags.Has(FlagHeadersPriority) {
		var v uint32
		p, v, err = readUint32(p)
		if err != nil {
			return nil, err
		}
		hf.Priority.StreamDep = v & 0x7fffffff
		hf.Priority.Exclusive = (v != hf.Priority.StreamDep) // high bit was set
		p, hf.Priority.Weight, err = readByte(p)
		if err != nil {
			return nil, err
		}
	}
	if len(p)-int(padLength) <= 0 {
		return nil, StreamError{fh.StreamID, ErrCodeProtocol}
	}
	hf.headerFragBuf = p[:len(p)-int(padLength)]
	return hf, nil
}

// HeadersFrameParam are the parameters for writing a HEADERS frame.
type HeadersFrameParam struct {
	// StreamID is the required Stream ID to initiate.
	StreamID uint32
	// BlockFragment is part (or all) of a Header Block.
	BlockFragment []byte

	// EndStream indicates that the header block is the last that
	// the endpoint will send for the identified stream. Setting
	// this flag causes the stream to enter one of "half closed"
	// states.
	EndStream bool

	// EndHeaders indicates that this frame contains an entire
	// header block and is not followed by any
	// CONTINUATION frames.
	EndHeaders bool

	// PadLength is the optional number of bytes of zeros to add
	// to this frame.
	PadLength uint8

	// Priority, if non-zero, includes stream priority information
	// in the HEADER frame.
	Priority PriorityParam
}

// WriteHeaders writes a single HEADERS frame.
//
// This is a low-level header writing method. Encoding headers and
// splitting them into any necessary CONTINUATION frames is handled
// elsewhere.
//
// It will perform exactly one Write to the underlying Writer.
// It is the caller's responsibility to not call other Write methods concurrently.
func (f *Framer) WriteHeaders(p HeadersFrameParam) error {
	if !validStreamID(p.StreamID) && !f.AllowIllegalWrites {
		return errStreamID
	}
	var flags Flags
	if p.PadLength != 0 {
		flags |= FlagHeadersPadded
	}
	if p.EndStream {
		flags |= FlagHeadersEndStream
	}
	if p.EndHeaders {
		flags |= FlagHeadersEndHeaders
	}
	if !p.Priority.IsZero() {
		flags |= FlagHeadersPriority
	}
	f.startWrite(FrameHeaders, flags, p.StreamID)
	if p.PadLength != 0 {
		f.writeByte(p.PadLength)
	}
	if !p.Priority.IsZero() {
		v := p.Priority.StreamDep
		if !validStreamID(v) && !f.AllowIllegalWrites {
			return errors.New("invalid dependent stream id")
		}
		if p.Priority.Exclusive {
			v |= 1 << 31
		}
		f.writeUint32(v)
		f.writeByte(p.Priority.Weight)
	}
	f.wbuf = append(f.wbuf, p.BlockFragment...)
	f.wbuf = append(f.wbuf, padZeros[:p.PadLength]...)
	return f.endWrite()
}

// A PriorityFrame specifies the sender-advised priority of a stream.
// See http://http2.github.io/http2-spec/#rfc.section.6.3
type PriorityFrame struct {
	FrameHeader
	PriorityParam
}

// PriorityParam are the stream prioritzation parameters.
type PriorityParam struct {
	// StreamDep is a 31-bit stream identifier for the
	// stream that this stream depends on. Zero means no
	// dependency.
	StreamDep uint32

	// Exclusive is whether the dependency is exclusive.
	Exclusive bool

	// Weight is the stream's zero-indexed weight. It should be
	// set together with StreamDep, or neither should be set.  Per
	// the spec, "Add one to the value to obtain a weight between
	// 1 and 256."
	Weight uint8
}

func (p PriorityParam) IsZero() bool {
	return p == PriorityParam{}
}

func parsePriorityFrame(fh FrameHeader, payload []byte) (Frame, error) {
	if fh.StreamID == 0 {
		return nil, ConnectionError(ErrCodeProtocol)
	}
	if len(payload) != 5 {
		return nil, ConnectionError(ErrCodeFrameSize)
	}
	v := binary.BigEndian.Uint32(payload[:4])
	streamID := v & 0x7fffffff // mask off high bit
	return &PriorityFrame{
		FrameHeader: fh,
		PriorityParam: PriorityParam{
			Weight:    payload[4],
			StreamDep: streamID,
			Exclusive: streamID != v, // was high bit set?
		},
	}, nil
}

// WritePriority writes a PRIORITY frame.
//
// It will perform exactly one Write to the underlying Writer.
// It is the caller's responsibility to not call other Write methods concurrently.
func (f *Framer) WritePriority(streamID uint32, p PriorityParam) error {
	if !validStreamID(streamID) && !f.AllowIllegalWrites {
		return errStreamID
	}
	f.startWrite(FramePriority, 0, streamID)
	v := p.StreamDep
	if p.Exclusive {
		v |= 1 << 31
	}
	f.writeUint32(v)
	f.writeByte(p.Weight)
	return f.endWrite()
}

// A RSTStreamFrame allows for abnormal termination of a stream.
// See http://http2.github.io/http2-spec/#rfc.section.6.4
type RSTStreamFrame struct {
	FrameHeader
	ErrCode ErrCode
}

func parseRSTStreamFrame(fh FrameHeader, p []byte) (Frame, error) {
	if len(p) != 4 {
		return nil, ConnectionError(ErrCodeFrameSize)
	}
	if fh.StreamID == 0 {
		return nil, ConnectionError(ErrCodeProtocol)
	}
	return &RSTStreamFrame{fh, ErrCode(binary.BigEndian.Uint32(p[:4]))}, nil
}

// WriteRSTStream writes a RST_STREAM frame.
//
// It will perform exactly one Write to the underlying Writer.
// It is the caller's responsibility to not call other Write methods concurrently.
func (f *Framer) WriteRSTStream(streamID uint32, code ErrCode) error {
	if !validStreamID(streamID) && !f.AllowIllegalWrites {
		return errStreamID
	}
	f.startWrite(FrameRSTStream, 0, streamID)
	f.writeUint32(uint32(code))
	return f.endWrite()
}

// A ContinuationFrame is used to continue a sequence of header block fragments.
// See http://http2.github.io/http2-spec/#rfc.section.6.10
type ContinuationFrame struct {
	FrameHeader
	headerFragBuf []byte
}

func parseContinuationFrame(fh FrameHeader, p []byte) (Frame, error) {
	return &ContinuationFrame{fh, p}, nil
}

func (f *ContinuationFrame) StreamEnded() bool {
	return f.FrameHeader.Flags.Has(FlagDataEndStream)
}

func (f *ContinuationFrame) HeaderBlockFragment() []byte {
	f.checkValid()
	return f.headerFragBuf
}

func (f *ContinuationFrame) HeadersEnded() bool {
	return f.FrameHeader.Flags.Has(FlagContinuationEndHeaders)
}

// WriteContinuation writes a CONTINUATION frame.
//
// It will perform exactly one Write to the underlying Writer.
// It is the caller's responsibility to not call other Write methods concurrently.
func (f *Framer) WriteContinuation(streamID uint32, endHeaders bool, headerBlockFragment []byte) error {
	if !validStreamID(streamID) && !f.AllowIllegalWrites {
		return errStreamID
	}
	var flags Flags
	if endHeaders {
		flags |= FlagContinuationEndHeaders
	}
	f.startWrite(FrameContinuation, flags, streamID)
	f.wbuf = append(f.wbuf, headerBlockFragment...)
	return f.endWrite()
}

// A PushPromiseFrame is used to initiate a server stream.
// See http://http2.github.io/http2-spec/#rfc.section.6.6
type PushPromiseFrame struct {
	FrameHeader
	PromiseID     uint32
	headerFragBuf []byte // not owned
}

func (f *PushPromiseFrame) HeaderBlockFragment() []byte {
	f.checkValid()
	return f.headerFragBuf
}

func (f *PushPromiseFrame) HeadersEnded() bool {
	return f.FrameHeader.Flags.Has(FlagPushPromiseEndHeaders)
}

func parsePushPromise(fh FrameHeader, p []byte) (_ Frame, err error) {
	pp := &PushPromiseFrame{
		FrameHeader: fh,
	}
	if pp.StreamID == 0 {
		// PUSH_PROMISE frames MUST be associated with an existing,
		// peer-initiated stream. The stream identifier of a
		// PUSH_PROMISE frame indicates the stream it is associated
		// with. If the stream identifier field specifies the value
		// 0x0, a recipient MUST respond with a connection error
		// (Section 5.4.1) of type PROTOCOL_ERROR.
		return nil, ConnectionError(ErrCodeProtocol)
	}
	// The PUSH_PROMISE frame includes optional padding.
	// Padding fields and flags are identical to those defined for DATA frames
	var padLength uint8
	if fh.Flags.Has(FlagPushPromisePadded) {
		if p, padLength, err = readByte(p); err != nil {
			return
		}
	}

	p, pp.PromiseID, err = readUint32(p)
	if err != nil {
		return
	}
	pp.PromiseID = pp.PromiseID & (1<<31 - 1)

	if int(padLength) > len(p) {
		// like the DATA frame, error out if padding is longer than the body.
		return nil, ConnectionError(ErrCodeProtocol)
	}
	pp.headerFragBuf = p[:len(p)-int(padLength)]
	return pp, nil
}

// PushPromiseParam are the parameters for writing a PUSH_PROMISE frame.
type PushPromiseParam struct {
	// StreamID is the required Stream ID to initiate.
	StreamID uint32

	// PromiseID is the required Stream ID which this
	// Push Promises
	PromiseID uint32

	// BlockFragment is part (or all) of a Header Block.
	BlockFragment []byte

	// EndHeaders indicates that this frame contains an entire
	// header block and is not followed by any
	// CONTINUATION frames.
	EndHeaders bool

	// PadLength is the optional number of bytes of zeros to add
	// to this frame.
	PadLength uint8
}

// WritePushPromise writes a single PushPromise Frame.
//
// As with Header Frames, This is the low level call for writing
// individual frames. Continuation frames are handled elsewhere.
//
// It will perform exactly one Write to the underlying Writer.
// It is the caller's responsibility to not call other Write methods concurrently.
func (f *Framer) WritePushPromise(p PushPromiseParam) error {
	if !validStreamID(p.StreamID) && !f.AllowIllegalWrites {
		return errStreamID
	}
	var flags Flags
	if p.PadLength != 0 {
		flags |= FlagPushPromisePadded
	}
	if p.EndHeaders {
		flags |= FlagPushPromiseEndHeaders
	}
	f.startWrite(FramePushPromise, flags, p.StreamID)
	if p.PadLength != 0 {
		f.writeByte(p.PadLength)
	}
	if !validStreamID(p.PromiseID) && !f.AllowIllegalWrites {
		return errStreamID
	}
	f.writeUint32(p.PromiseID)
	f.wbuf = append(f.wbuf, p.BlockFragment...)
	f.wbuf = append(f.wbuf, padZeros[:p.PadLength]...)
	return f.endWrite()
}

// WriteRawFrame writes a raw frame. This can be used to write
// extension frames unknown to this package.
func (f *Framer) WriteRawFrame(t FrameType, flags Flags, streamID uint32, payload []byte) error {
	f.startWrite(t, flags, streamID)
	f.writeBytes(payload)
	return f.endWrite()
}

func readByte(p []byte) (remain []byte, b byte, err error) {
	if len(p) == 0 {
		return nil, 0, io.ErrUnexpectedEOF
	}
	return p[1:], p[0], nil
}

func readUint32(p []byte) (remain []byte, v uint32, err error) {
	if len(p) < 4 {
		return nil, 0, io.ErrUnexpectedEOF
	}
	return p[4:], binary.BigEndian.Uint32(p[:4]), nil
}

type streamEnder interface {
	StreamEnded() bool
}

type headersEnder interface {
	HeadersEnded() bool
}
