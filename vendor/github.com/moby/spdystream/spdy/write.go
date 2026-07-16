// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spdy

import (
	"encoding/binary"
	"io"
	"math"
	"net/http"
	"strings"
)

func (frame *SynStreamFrame) write(f *Framer) error {
	return f.writeSynStreamFrame(frame)
}

func (frame *SynReplyFrame) write(f *Framer) error {
	return f.writeSynReplyFrame(frame)
}

func (frame *RstStreamFrame) write(f *Framer) (err error) {
	if frame.StreamId == 0 {
		return &Error{ZeroStreamId, 0}
	}
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeRstStream
	frame.CFHeader.Flags = 0
	frame.CFHeader.length = 8

	// Serialize frame to Writer.
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return
	}
	if frame.Status == 0 {
		return &Error{InvalidControlFrame, frame.StreamId}
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.Status); err != nil {
		return
	}
	return
}

func (frame *SettingsFrame) write(f *Framer) (err error) {
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeSettings
	payloadLen := len(frame.FlagIdValues)*8 + 4
	if payloadLen > MaxDataLength {
		return &Error{InvalidControlFrame, 0}
	}
	frame.CFHeader.length = uint32(payloadLen)

	// Serialize frame to Writer.
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	n := len(frame.FlagIdValues)
	if uint64(n) > math.MaxUint32 {
		return &Error{InvalidControlFrame, 0}
	}
	if err = binary.Write(f.w, binary.BigEndian, uint32(n)); err != nil {
		return
	}
	for _, flagIdValue := range frame.FlagIdValues {
		flagId := uint32(flagIdValue.Flag)<<24 | uint32(flagIdValue.Id)
		if err = binary.Write(f.w, binary.BigEndian, flagId); err != nil {
			return
		}
		if err = binary.Write(f.w, binary.BigEndian, flagIdValue.Value); err != nil {
			return
		}
	}
	return
}

func (frame *PingFrame) write(f *Framer) (err error) {
	if frame.Id == 0 {
		return &Error{ZeroStreamId, 0}
	}
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypePing
	frame.CFHeader.Flags = 0
	frame.CFHeader.length = 4

	// Serialize frame to Writer.
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.Id); err != nil {
		return
	}
	return
}

func (frame *GoAwayFrame) write(f *Framer) (err error) {
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeGoAway
	frame.CFHeader.Flags = 0
	frame.CFHeader.length = 8

	// Serialize frame to Writer.
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.LastGoodStreamId); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.Status); err != nil {
		return
	}
	return nil
}

func (frame *HeadersFrame) write(f *Framer) error {
	return f.writeHeadersFrame(frame)
}

func (frame *WindowUpdateFrame) write(f *Framer) (err error) {
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeWindowUpdate
	frame.CFHeader.Flags = 0
	frame.CFHeader.length = 8

	// Serialize frame to Writer.
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.DeltaWindowSize); err != nil {
		return
	}
	return nil
}

func (frame *DataFrame) write(f *Framer) error {
	return f.writeDataFrame(frame)
}

// WriteFrame writes a frame.
func (f *Framer) WriteFrame(frame Frame) error {
	return frame.write(f)
}

func writeControlFrameHeader(w io.Writer, h ControlFrameHeader) error {
	if err := binary.Write(w, binary.BigEndian, 0x8000|h.version); err != nil {
		return err
	}
	if err := binary.Write(w, binary.BigEndian, h.frameType); err != nil {
		return err
	}
	flagsAndLength := uint32(h.Flags)<<24 | h.length
	if err := binary.Write(w, binary.BigEndian, flagsAndLength); err != nil {
		return err
	}
	return nil
}

func writeHeaderValueBlock(w io.Writer, h http.Header) (n int, err error) {
	n = 0
	numHeaders := len(h)
	if numHeaders > math.MaxInt32 {
		return n, &Error{InvalidControlFrame, 0}
	}
	if err = binary.Write(w, binary.BigEndian, uint32(numHeaders)); err != nil {
		return
	}
	n += 4
	for name, values := range h {
		nameLen := len(name)
		if nameLen > math.MaxInt32 {
			return n, &Error{InvalidControlFrame, 0}
		}
		if err = binary.Write(w, binary.BigEndian, uint32(nameLen)); err != nil {
			return
		}
		n += 4
		name = strings.ToLower(name)
		if _, err = io.WriteString(w, name); err != nil {
			return
		}
		n += nameLen
		v := strings.Join(values, headerValueSeparator)
		vLen := len(v)
		if vLen > math.MaxInt32 {
			return n, &Error{InvalidControlFrame, 0}
		}
		if err = binary.Write(w, binary.BigEndian, uint32(vLen)); err != nil {
			return
		}
		n += 4
		if _, err = io.WriteString(w, v); err != nil {
			return
		}
		n += vLen
	}
	return
}

func (f *Framer) writeSynStreamFrame(frame *SynStreamFrame) (err error) {
	if frame.StreamId == 0 {
		return &Error{ZeroStreamId, 0}
	}
	// Marshal the headers.
	var writer io.Writer = f.headerBuf
	if !f.headerCompressionDisabled {
		writer = f.headerCompressor
	}
	if _, err = writeHeaderValueBlock(writer, frame.Headers); err != nil {
		return
	}
	if !f.headerCompressionDisabled {
		f.headerCompressor.Flush()
	}

	// Set ControlFrameHeader.
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeSynStream
	hLen := len(f.headerBuf.Bytes()) + 10
	if hLen > MaxDataLength {
		return &Error{InvalidControlFrame, 0}
	}
	frame.CFHeader.length = uint32(hLen)

	// Serialize frame to Writer.
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return err
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return err
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.AssociatedToStreamId); err != nil {
		return err
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.Priority<<5); err != nil {
		return err
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.Slot); err != nil {
		return err
	}
	if _, err = f.w.Write(f.headerBuf.Bytes()); err != nil {
		return err
	}
	f.headerBuf.Reset()
	return nil
}

func (f *Framer) writeSynReplyFrame(frame *SynReplyFrame) (err error) {
	if frame.StreamId == 0 {
		return &Error{ZeroStreamId, 0}
	}
	// Marshal the headers.
	var writer io.Writer = f.headerBuf
	if !f.headerCompressionDisabled {
		writer = f.headerCompressor
	}
	if _, err = writeHeaderValueBlock(writer, frame.Headers); err != nil {
		return
	}
	if !f.headerCompressionDisabled {
		f.headerCompressor.Flush()
	}

	// Set ControlFrameHeader.
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeSynReply
	hLen := len(f.headerBuf.Bytes()) + 4
	if hLen > MaxDataLength {
		return &Error{InvalidControlFrame, 0}
	}
	frame.CFHeader.length = uint32(hLen)

	// Serialize frame to Writer.
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return
	}
	if _, err = f.w.Write(f.headerBuf.Bytes()); err != nil {
		return
	}
	f.headerBuf.Reset()
	return
}

func (f *Framer) writeHeadersFrame(frame *HeadersFrame) (err error) {
	if frame.StreamId == 0 {
		return &Error{ZeroStreamId, 0}
	}
	// Marshal the headers.
	var writer io.Writer = f.headerBuf
	if !f.headerCompressionDisabled {
		writer = f.headerCompressor
	}
	if _, err = writeHeaderValueBlock(writer, frame.Headers); err != nil {
		return
	}
	if !f.headerCompressionDisabled {
		f.headerCompressor.Flush()
	}

	// Set ControlFrameHeader.
	frame.CFHeader.version = Version
	frame.CFHeader.frameType = TypeHeaders
	hLen := len(f.headerBuf.Bytes()) + 4
	if hLen > MaxDataLength {
		return &Error{InvalidControlFrame, 0}
	}
	frame.CFHeader.length = uint32(hLen)

	// Serialize frame to Writer.
	if err = writeControlFrameHeader(f.w, frame.CFHeader); err != nil {
		return
	}
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return
	}
	if _, err = f.w.Write(f.headerBuf.Bytes()); err != nil {
		return
	}
	f.headerBuf.Reset()
	return
}

func (f *Framer) writeDataFrame(frame *DataFrame) (err error) {
	if frame.StreamId == 0 {
		return &Error{ZeroStreamId, 0}
	}
	if frame.StreamId&0x80000000 != 0 || len(frame.Data) > MaxDataLength {
		return &Error{InvalidDataFrame, frame.StreamId}
	}

	// Serialize frame to Writer.
	if err = binary.Write(f.w, binary.BigEndian, frame.StreamId); err != nil {
		return
	}
	dLen := len(frame.Data)
	if dLen > MaxDataLength {
		return &Error{InvalidDataFrame, frame.StreamId}
	}
	flagsAndLength := uint32(frame.Flags)<<24 | uint32(dLen)
	if err = binary.Write(f.w, binary.BigEndian, flagsAndLength); err != nil {
		return
	}
	if _, err = f.w.Write(frame.Data); err != nil {
		return
	}
	return nil
}
