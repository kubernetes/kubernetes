// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spdy

import (
	"bytes"
	"compress/zlib"
	"encoding/base64"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"
)

var HeadersFixture = http.Header{
	"Url":     []string{"http://www.google.com/"},
	"Method":  []string{"get"},
	"Version": []string{"http/1.1"},
}

func TestHeaderParsing(t *testing.T) {
	var headerValueBlockBuf bytes.Buffer
	writeHeaderValueBlock(&headerValueBlockBuf, HeadersFixture)
	const bogusStreamId = 1
	newHeaders, err := parseHeaderValueBlock(&headerValueBlockBuf, bogusStreamId)
	if err != nil {
		t.Fatal("parseHeaderValueBlock:", err)
	}
	if !reflect.DeepEqual(HeadersFixture, newHeaders) {
		t.Fatal("got: ", newHeaders, "\nwant: ", HeadersFixture)
	}
}

func TestCreateParseSynStreamFrameCompressionDisable(t *testing.T) {
	buffer := new(bytes.Buffer)
	// Fixture framer for no compression test.
	framer := &Framer{
		headerCompressionDisabled: true,
		w:         buffer,
		headerBuf: new(bytes.Buffer),
		r:         buffer,
	}
	synStreamFrame := SynStreamFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeSynStream,
		},
		StreamId: 2,
		Headers:  HeadersFixture,
	}
	if err := framer.WriteFrame(&synStreamFrame); err != nil {
		t.Fatal("WriteFrame without compression:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame without compression:", err)
	}
	parsedSynStreamFrame, ok := frame.(*SynStreamFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(synStreamFrame, *parsedSynStreamFrame) {
		t.Fatal("got: ", *parsedSynStreamFrame, "\nwant: ", synStreamFrame)
	}
}

func TestCreateParseSynStreamFrameCompressionEnable(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	synStreamFrame := SynStreamFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeSynStream,
		},
		StreamId: 2,
		Headers:  HeadersFixture,
	}
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	if err := framer.WriteFrame(&synStreamFrame); err != nil {
		t.Fatal("WriteFrame with compression:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame with compression:", err)
	}
	parsedSynStreamFrame, ok := frame.(*SynStreamFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(synStreamFrame, *parsedSynStreamFrame) {
		t.Fatal("got: ", *parsedSynStreamFrame, "\nwant: ", synStreamFrame)
	}
}

func TestCreateParseSynReplyFrameCompressionDisable(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer := &Framer{
		headerCompressionDisabled: true,
		w:         buffer,
		headerBuf: new(bytes.Buffer),
		r:         buffer,
	}
	synReplyFrame := SynReplyFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeSynReply,
		},
		StreamId: 2,
		Headers:  HeadersFixture,
	}
	if err := framer.WriteFrame(&synReplyFrame); err != nil {
		t.Fatal("WriteFrame without compression:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame without compression:", err)
	}
	parsedSynReplyFrame, ok := frame.(*SynReplyFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(synReplyFrame, *parsedSynReplyFrame) {
		t.Fatal("got: ", *parsedSynReplyFrame, "\nwant: ", synReplyFrame)
	}
}

func TestCreateParseSynReplyFrameCompressionEnable(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	synReplyFrame := SynReplyFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeSynReply,
		},
		StreamId: 2,
		Headers:  HeadersFixture,
	}
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	if err := framer.WriteFrame(&synReplyFrame); err != nil {
		t.Fatal("WriteFrame with compression:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame with compression:", err)
	}
	parsedSynReplyFrame, ok := frame.(*SynReplyFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(synReplyFrame, *parsedSynReplyFrame) {
		t.Fatal("got: ", *parsedSynReplyFrame, "\nwant: ", synReplyFrame)
	}
}

func TestCreateParseRstStream(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	rstStreamFrame := RstStreamFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeRstStream,
		},
		StreamId: 1,
		Status:   InvalidStream,
	}
	if err := framer.WriteFrame(&rstStreamFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedRstStreamFrame, ok := frame.(*RstStreamFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(rstStreamFrame, *parsedRstStreamFrame) {
		t.Fatal("got: ", *parsedRstStreamFrame, "\nwant: ", rstStreamFrame)
	}
}

func TestCreateParseSettings(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	settingsFrame := SettingsFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeSettings,
		},
		FlagIdValues: []SettingsFlagIdValue{
			{FlagSettingsPersistValue, SettingsCurrentCwnd, 10},
			{FlagSettingsPersisted, SettingsUploadBandwidth, 1},
		},
	}
	if err := framer.WriteFrame(&settingsFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedSettingsFrame, ok := frame.(*SettingsFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(settingsFrame, *parsedSettingsFrame) {
		t.Fatal("got: ", *parsedSettingsFrame, "\nwant: ", settingsFrame)
	}
}

func TestCreateParsePing(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	pingFrame := PingFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypePing,
		},
		Id: 31337,
	}
	if err := framer.WriteFrame(&pingFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	if pingFrame.CFHeader.Flags != 0 {
		t.Fatal("Incorrect frame type:", pingFrame)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedPingFrame, ok := frame.(*PingFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if parsedPingFrame.CFHeader.Flags != 0 {
		t.Fatal("Parsed incorrect frame type:", parsedPingFrame)
	}
	if !reflect.DeepEqual(pingFrame, *parsedPingFrame) {
		t.Fatal("got: ", *parsedPingFrame, "\nwant: ", pingFrame)
	}
}

func TestCreateParseGoAway(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	goAwayFrame := GoAwayFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeGoAway,
		},
		LastGoodStreamId: 31337,
		Status:           1,
	}
	if err := framer.WriteFrame(&goAwayFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	if goAwayFrame.CFHeader.Flags != 0 {
		t.Fatal("Incorrect frame type:", goAwayFrame)
	}
	if goAwayFrame.CFHeader.length != 8 {
		t.Fatal("Incorrect frame type:", goAwayFrame)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedGoAwayFrame, ok := frame.(*GoAwayFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if parsedGoAwayFrame.CFHeader.Flags != 0 {
		t.Fatal("Incorrect frame type:", parsedGoAwayFrame)
	}
	if parsedGoAwayFrame.CFHeader.length != 8 {
		t.Fatal("Incorrect frame type:", parsedGoAwayFrame)
	}
	if !reflect.DeepEqual(goAwayFrame, *parsedGoAwayFrame) {
		t.Fatal("got: ", *parsedGoAwayFrame, "\nwant: ", goAwayFrame)
	}
}

func TestCreateParseHeadersFrame(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer := &Framer{
		headerCompressionDisabled: true,
		w:         buffer,
		headerBuf: new(bytes.Buffer),
		r:         buffer,
	}
	headersFrame := HeadersFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeHeaders,
		},
		StreamId: 2,
	}
	headersFrame.Headers = HeadersFixture
	if err := framer.WriteFrame(&headersFrame); err != nil {
		t.Fatal("WriteFrame without compression:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame without compression:", err)
	}
	parsedHeadersFrame, ok := frame.(*HeadersFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(headersFrame, *parsedHeadersFrame) {
		t.Fatal("got: ", *parsedHeadersFrame, "\nwant: ", headersFrame)
	}
}

func TestCreateParseHeadersFrameCompressionEnable(t *testing.T) {
	buffer := new(bytes.Buffer)
	headersFrame := HeadersFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeHeaders,
		},
		StreamId: 2,
	}
	headersFrame.Headers = HeadersFixture

	framer, err := NewFramer(buffer, buffer)
	if err := framer.WriteFrame(&headersFrame); err != nil {
		t.Fatal("WriteFrame with compression:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame with compression:", err)
	}
	parsedHeadersFrame, ok := frame.(*HeadersFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(headersFrame, *parsedHeadersFrame) {
		t.Fatal("got: ", *parsedHeadersFrame, "\nwant: ", headersFrame)
	}
}

func TestCreateParseWindowUpdateFrame(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	windowUpdateFrame := WindowUpdateFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeWindowUpdate,
		},
		StreamId:        31337,
		DeltaWindowSize: 1,
	}
	if err := framer.WriteFrame(&windowUpdateFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	if windowUpdateFrame.CFHeader.Flags != 0 {
		t.Fatal("Incorrect frame type:", windowUpdateFrame)
	}
	if windowUpdateFrame.CFHeader.length != 8 {
		t.Fatal("Incorrect frame type:", windowUpdateFrame)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedWindowUpdateFrame, ok := frame.(*WindowUpdateFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if parsedWindowUpdateFrame.CFHeader.Flags != 0 {
		t.Fatal("Incorrect frame type:", parsedWindowUpdateFrame)
	}
	if parsedWindowUpdateFrame.CFHeader.length != 8 {
		t.Fatal("Incorrect frame type:", parsedWindowUpdateFrame)
	}
	if !reflect.DeepEqual(windowUpdateFrame, *parsedWindowUpdateFrame) {
		t.Fatal("got: ", *parsedWindowUpdateFrame, "\nwant: ", windowUpdateFrame)
	}
}

func TestCreateParseDataFrame(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	dataFrame := DataFrame{
		StreamId: 1,
		Data:     []byte{'h', 'e', 'l', 'l', 'o'},
	}
	if err := framer.WriteFrame(&dataFrame); err != nil {
		t.Fatal("WriteFrame:", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame:", err)
	}
	parsedDataFrame, ok := frame.(*DataFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(dataFrame, *parsedDataFrame) {
		t.Fatal("got: ", *parsedDataFrame, "\nwant: ", dataFrame)
	}
}

func TestCompressionContextAcrossFrames(t *testing.T) {
	buffer := new(bytes.Buffer)
	framer, err := NewFramer(buffer, buffer)
	if err != nil {
		t.Fatal("Failed to create new framer:", err)
	}
	headersFrame := HeadersFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeHeaders,
		},
		StreamId: 2,
		Headers:  HeadersFixture,
	}
	if err := framer.WriteFrame(&headersFrame); err != nil {
		t.Fatal("WriteFrame (HEADERS):", err)
	}
	synStreamFrame := SynStreamFrame{
		ControlFrameHeader{
			Version,
			TypeSynStream,
			0, // Flags
			0, // length
		},
		2,   // StreamId
		0,   // AssociatedTOStreamID
		0,   // Priority
		1,   // Slot
		nil, // Headers
	}
	synStreamFrame.Headers = HeadersFixture

	if err := framer.WriteFrame(&synStreamFrame); err != nil {
		t.Fatal("WriteFrame (SYN_STREAM):", err)
	}
	frame, err := framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame (HEADERS):", err, buffer.Bytes())
	}
	parsedHeadersFrame, ok := frame.(*HeadersFrame)
	if !ok {
		t.Fatalf("expected HeadersFrame; got %T %v", frame, frame)
	}
	if !reflect.DeepEqual(headersFrame, *parsedHeadersFrame) {
		t.Fatal("got: ", *parsedHeadersFrame, "\nwant: ", headersFrame)
	}
	frame, err = framer.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame (SYN_STREAM):", err, buffer.Bytes())
	}
	parsedSynStreamFrame, ok := frame.(*SynStreamFrame)
	if !ok {
		t.Fatalf("expected SynStreamFrame; got %T %v", frame, frame)
	}
	if !reflect.DeepEqual(synStreamFrame, *parsedSynStreamFrame) {
		t.Fatal("got: ", *parsedSynStreamFrame, "\nwant: ", synStreamFrame)
	}
}

func TestMultipleSPDYFrames(t *testing.T) {
	// Initialize the framers.
	pr1, pw1 := io.Pipe()
	pr2, pw2 := io.Pipe()
	writer, err := NewFramer(pw1, pr2)
	if err != nil {
		t.Fatal("Failed to create writer:", err)
	}
	reader, err := NewFramer(pw2, pr1)
	if err != nil {
		t.Fatal("Failed to create reader:", err)
	}

	// Set up the frames we're actually transferring.
	headersFrame := HeadersFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeHeaders,
		},
		StreamId: 2,
		Headers:  HeadersFixture,
	}
	synStreamFrame := SynStreamFrame{
		CFHeader: ControlFrameHeader{
			version:   Version,
			frameType: TypeSynStream,
		},
		StreamId: 2,
		Headers:  HeadersFixture,
	}

	// Start the goroutines to write the frames.
	go func() {
		if err := writer.WriteFrame(&headersFrame); err != nil {
			t.Fatal("WriteFrame (HEADERS): ", err)
		}
		if err := writer.WriteFrame(&synStreamFrame); err != nil {
			t.Fatal("WriteFrame (SYN_STREAM): ", err)
		}
	}()

	// Read the frames and verify they look as expected.
	frame, err := reader.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame (HEADERS): ", err)
	}
	parsedHeadersFrame, ok := frame.(*HeadersFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type:", frame)
	}
	if !reflect.DeepEqual(headersFrame, *parsedHeadersFrame) {
		t.Fatal("got: ", *parsedHeadersFrame, "\nwant: ", headersFrame)
	}
	frame, err = reader.ReadFrame()
	if err != nil {
		t.Fatal("ReadFrame (SYN_STREAM):", err)
	}
	parsedSynStreamFrame, ok := frame.(*SynStreamFrame)
	if !ok {
		t.Fatal("Parsed incorrect frame type.")
	}
	if !reflect.DeepEqual(synStreamFrame, *parsedSynStreamFrame) {
		t.Fatal("got: ", *parsedSynStreamFrame, "\nwant: ", synStreamFrame)
	}
}

func TestReadMalformedZlibHeader(t *testing.T) {
	// These were constructed by corrupting the first byte of the zlib
	// header after writing.
	malformedStructs := map[string]string{
		"SynStreamFrame": "gAIAAQAAABgAAAACAAAAAAAAF/nfolGyYmAAAAAA//8=",
		"SynReplyFrame":  "gAIAAgAAABQAAAACAAAX+d+iUbJiYAAAAAD//w==",
		"HeadersFrame":   "gAIACAAAABQAAAACAAAX+d+iUbJiYAAAAAD//w==",
	}
	for name, bad := range malformedStructs {
		b, err := base64.StdEncoding.DecodeString(bad)
		if err != nil {
			t.Errorf("Unable to decode base64 encoded frame %s: %v", name, err)
		}
		buf := bytes.NewBuffer(b)
		reader, err := NewFramer(buf, buf)
		if err != nil {
			t.Fatalf("NewFramer: %v", err)
		}
		_, err = reader.ReadFrame()
		if err != zlib.ErrHeader {
			t.Errorf("Frame %s, expected: %#v, actual: %#v", name, zlib.ErrHeader, err)
		}
	}
}

// TODO: these tests are too weak for updating SPDY spec. Fix me.

type zeroStream struct {
	frame   Frame
	encoded string
}

var streamIdZeroFrames = map[string]zeroStream{
	"SynStreamFrame": {
		&SynStreamFrame{StreamId: 0},
		"gAIAAQAAABgAAAAAAAAAAAAAePnfolGyYmAAAAAA//8=",
	},
	"SynReplyFrame": {
		&SynReplyFrame{StreamId: 0},
		"gAIAAgAAABQAAAAAAAB4+d+iUbJiYAAAAAD//w==",
	},
	"RstStreamFrame": {
		&RstStreamFrame{StreamId: 0},
		"gAIAAwAAAAgAAAAAAAAAAA==",
	},
	"HeadersFrame": {
		&HeadersFrame{StreamId: 0},
		"gAIACAAAABQAAAAAAAB4+d+iUbJiYAAAAAD//w==",
	},
	"DataFrame": {
		&DataFrame{StreamId: 0},
		"AAAAAAAAAAA=",
	},
	"PingFrame": {
		&PingFrame{Id: 0},
		"gAIABgAAAAQAAAAA",
	},
}

func TestNoZeroStreamId(t *testing.T) {
	t.Log("skipping") // TODO: update to work with SPDY3
	return

	for name, f := range streamIdZeroFrames {
		b, err := base64.StdEncoding.DecodeString(f.encoded)
		if err != nil {
			t.Errorf("Unable to decode base64 encoded frame %s: %v", f, err)
			continue
		}
		framer, err := NewFramer(ioutil.Discard, bytes.NewReader(b))
		if err != nil {
			t.Fatalf("NewFramer: %v", err)
		}
		err = framer.WriteFrame(f.frame)
		checkZeroStreamId(t, name, "WriteFrame", err)

		_, err = framer.ReadFrame()
		checkZeroStreamId(t, name, "ReadFrame", err)
	}
}

func checkZeroStreamId(t *testing.T, frame string, method string, err error) {
	if err == nil {
		t.Errorf("%s ZeroStreamId, no error on %s", method, frame)
		return
	}
	eerr, ok := err.(*Error)
	if !ok || eerr.Err != ZeroStreamId {
		t.Errorf("%s ZeroStreamId, incorrect error %#v, frame %s", method, eerr, frame)
	}
}
