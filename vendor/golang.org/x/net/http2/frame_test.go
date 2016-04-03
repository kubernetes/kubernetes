// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

func testFramer() (*Framer, *bytes.Buffer) {
	buf := new(bytes.Buffer)
	return NewFramer(buf, buf), buf
}

func TestFrameSizes(t *testing.T) {
	// Catch people rearranging the FrameHeader fields.
	if got, want := int(unsafe.Sizeof(FrameHeader{})), 12; got != want {
		t.Errorf("FrameHeader size = %d; want %d", got, want)
	}
}

func TestFrameTypeString(t *testing.T) {
	tests := []struct {
		ft   FrameType
		want string
	}{
		{FrameData, "DATA"},
		{FramePing, "PING"},
		{FrameGoAway, "GOAWAY"},
		{0xf, "UNKNOWN_FRAME_TYPE_15"},
	}

	for i, tt := range tests {
		got := tt.ft.String()
		if got != tt.want {
			t.Errorf("%d. String(FrameType %d) = %q; want %q", i, int(tt.ft), got, tt.want)
		}
	}
}

func TestWriteRST(t *testing.T) {
	fr, buf := testFramer()
	var streamID uint32 = 1<<24 + 2<<16 + 3<<8 + 4
	var errCode uint32 = 7<<24 + 6<<16 + 5<<8 + 4
	fr.WriteRSTStream(streamID, ErrCode(errCode))
	const wantEnc = "\x00\x00\x04\x03\x00\x01\x02\x03\x04\x07\x06\x05\x04"
	if buf.String() != wantEnc {
		t.Errorf("encoded as %q; want %q", buf.Bytes(), wantEnc)
	}
	f, err := fr.ReadFrame()
	if err != nil {
		t.Fatal(err)
	}
	want := &RSTStreamFrame{
		FrameHeader: FrameHeader{
			valid:    true,
			Type:     0x3,
			Flags:    0x0,
			Length:   0x4,
			StreamID: 0x1020304,
		},
		ErrCode: 0x7060504,
	}
	if !reflect.DeepEqual(f, want) {
		t.Errorf("parsed back %#v; want %#v", f, want)
	}
}

func TestWriteData(t *testing.T) {
	fr, buf := testFramer()
	var streamID uint32 = 1<<24 + 2<<16 + 3<<8 + 4
	data := []byte("ABC")
	fr.WriteData(streamID, true, data)
	const wantEnc = "\x00\x00\x03\x00\x01\x01\x02\x03\x04ABC"
	if buf.String() != wantEnc {
		t.Errorf("encoded as %q; want %q", buf.Bytes(), wantEnc)
	}
	f, err := fr.ReadFrame()
	if err != nil {
		t.Fatal(err)
	}
	df, ok := f.(*DataFrame)
	if !ok {
		t.Fatalf("got %T; want *DataFrame", f)
	}
	if !bytes.Equal(df.Data(), data) {
		t.Errorf("got %q; want %q", df.Data(), data)
	}
	if f.Header().Flags&1 == 0 {
		t.Errorf("didn't see END_STREAM flag")
	}
}

func TestWriteHeaders(t *testing.T) {
	tests := []struct {
		name      string
		p         HeadersFrameParam
		wantEnc   string
		wantFrame *HeadersFrame
	}{
		{
			"basic",
			HeadersFrameParam{
				StreamID:      42,
				BlockFragment: []byte("abc"),
				Priority:      PriorityParam{},
			},
			"\x00\x00\x03\x01\x00\x00\x00\x00*abc",
			&HeadersFrame{
				FrameHeader: FrameHeader{
					valid:    true,
					StreamID: 42,
					Type:     FrameHeaders,
					Length:   uint32(len("abc")),
				},
				Priority:      PriorityParam{},
				headerFragBuf: []byte("abc"),
			},
		},
		{
			"basic + end flags",
			HeadersFrameParam{
				StreamID:      42,
				BlockFragment: []byte("abc"),
				EndStream:     true,
				EndHeaders:    true,
				Priority:      PriorityParam{},
			},
			"\x00\x00\x03\x01\x05\x00\x00\x00*abc",
			&HeadersFrame{
				FrameHeader: FrameHeader{
					valid:    true,
					StreamID: 42,
					Type:     FrameHeaders,
					Flags:    FlagHeadersEndStream | FlagHeadersEndHeaders,
					Length:   uint32(len("abc")),
				},
				Priority:      PriorityParam{},
				headerFragBuf: []byte("abc"),
			},
		},
		{
			"with padding",
			HeadersFrameParam{
				StreamID:      42,
				BlockFragment: []byte("abc"),
				EndStream:     true,
				EndHeaders:    true,
				PadLength:     5,
				Priority:      PriorityParam{},
			},
			"\x00\x00\t\x01\r\x00\x00\x00*\x05abc\x00\x00\x00\x00\x00",
			&HeadersFrame{
				FrameHeader: FrameHeader{
					valid:    true,
					StreamID: 42,
					Type:     FrameHeaders,
					Flags:    FlagHeadersEndStream | FlagHeadersEndHeaders | FlagHeadersPadded,
					Length:   uint32(1 + len("abc") + 5), // pad length + contents + padding
				},
				Priority:      PriorityParam{},
				headerFragBuf: []byte("abc"),
			},
		},
		{
			"with priority",
			HeadersFrameParam{
				StreamID:      42,
				BlockFragment: []byte("abc"),
				EndStream:     true,
				EndHeaders:    true,
				PadLength:     2,
				Priority: PriorityParam{
					StreamDep: 15,
					Exclusive: true,
					Weight:    127,
				},
			},
			"\x00\x00\v\x01-\x00\x00\x00*\x02\x80\x00\x00\x0f\u007fabc\x00\x00",
			&HeadersFrame{
				FrameHeader: FrameHeader{
					valid:    true,
					StreamID: 42,
					Type:     FrameHeaders,
					Flags:    FlagHeadersEndStream | FlagHeadersEndHeaders | FlagHeadersPadded | FlagHeadersPriority,
					Length:   uint32(1 + 5 + len("abc") + 2), // pad length + priority + contents + padding
				},
				Priority: PriorityParam{
					StreamDep: 15,
					Exclusive: true,
					Weight:    127,
				},
				headerFragBuf: []byte("abc"),
			},
		},
	}
	for _, tt := range tests {
		fr, buf := testFramer()
		if err := fr.WriteHeaders(tt.p); err != nil {
			t.Errorf("test %q: %v", tt.name, err)
			continue
		}
		if buf.String() != tt.wantEnc {
			t.Errorf("test %q: encoded %q; want %q", tt.name, buf.Bytes(), tt.wantEnc)
		}
		f, err := fr.ReadFrame()
		if err != nil {
			t.Errorf("test %q: failed to read the frame back: %v", tt.name, err)
			continue
		}
		if !reflect.DeepEqual(f, tt.wantFrame) {
			t.Errorf("test %q: mismatch.\n got: %#v\nwant: %#v\n", tt.name, f, tt.wantFrame)
		}
	}
}

func TestWriteContinuation(t *testing.T) {
	const streamID = 42
	tests := []struct {
		name string
		end  bool
		frag []byte

		wantFrame *ContinuationFrame
	}{
		{
			"not end",
			false,
			[]byte("abc"),
			&ContinuationFrame{
				FrameHeader: FrameHeader{
					valid:    true,
					StreamID: streamID,
					Type:     FrameContinuation,
					Length:   uint32(len("abc")),
				},
				headerFragBuf: []byte("abc"),
			},
		},
		{
			"end",
			true,
			[]byte("def"),
			&ContinuationFrame{
				FrameHeader: FrameHeader{
					valid:    true,
					StreamID: streamID,
					Type:     FrameContinuation,
					Flags:    FlagContinuationEndHeaders,
					Length:   uint32(len("def")),
				},
				headerFragBuf: []byte("def"),
			},
		},
	}
	for _, tt := range tests {
		fr, _ := testFramer()
		if err := fr.WriteContinuation(streamID, tt.end, tt.frag); err != nil {
			t.Errorf("test %q: %v", tt.name, err)
			continue
		}
		f, err := fr.ReadFrame()
		if err != nil {
			t.Errorf("test %q: failed to read the frame back: %v", tt.name, err)
			continue
		}
		if !reflect.DeepEqual(f, tt.wantFrame) {
			t.Errorf("test %q: mismatch.\n got: %#v\nwant: %#v\n", tt.name, f, tt.wantFrame)
		}
	}
}

func TestWritePriority(t *testing.T) {
	const streamID = 42
	tests := []struct {
		name      string
		priority  PriorityParam
		wantFrame *PriorityFrame
	}{
		{
			"not exclusive",
			PriorityParam{
				StreamDep: 2,
				Exclusive: false,
				Weight:    127,
			},
			&PriorityFrame{
				FrameHeader{
					valid:    true,
					StreamID: streamID,
					Type:     FramePriority,
					Length:   5,
				},
				PriorityParam{
					StreamDep: 2,
					Exclusive: false,
					Weight:    127,
				},
			},
		},

		{
			"exclusive",
			PriorityParam{
				StreamDep: 3,
				Exclusive: true,
				Weight:    77,
			},
			&PriorityFrame{
				FrameHeader{
					valid:    true,
					StreamID: streamID,
					Type:     FramePriority,
					Length:   5,
				},
				PriorityParam{
					StreamDep: 3,
					Exclusive: true,
					Weight:    77,
				},
			},
		},
	}
	for _, tt := range tests {
		fr, _ := testFramer()
		if err := fr.WritePriority(streamID, tt.priority); err != nil {
			t.Errorf("test %q: %v", tt.name, err)
			continue
		}
		f, err := fr.ReadFrame()
		if err != nil {
			t.Errorf("test %q: failed to read the frame back: %v", tt.name, err)
			continue
		}
		if !reflect.DeepEqual(f, tt.wantFrame) {
			t.Errorf("test %q: mismatch.\n got: %#v\nwant: %#v\n", tt.name, f, tt.wantFrame)
		}
	}
}

func TestWriteSettings(t *testing.T) {
	fr, buf := testFramer()
	settings := []Setting{{1, 2}, {3, 4}}
	fr.WriteSettings(settings...)
	const wantEnc = "\x00\x00\f\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x03\x00\x00\x00\x04"
	if buf.String() != wantEnc {
		t.Errorf("encoded as %q; want %q", buf.Bytes(), wantEnc)
	}
	f, err := fr.ReadFrame()
	if err != nil {
		t.Fatal(err)
	}
	sf, ok := f.(*SettingsFrame)
	if !ok {
		t.Fatalf("Got a %T; want a SettingsFrame", f)
	}
	var got []Setting
	sf.ForeachSetting(func(s Setting) error {
		got = append(got, s)
		valBack, ok := sf.Value(s.ID)
		if !ok || valBack != s.Val {
			t.Errorf("Value(%d) = %v, %v; want %v, true", s.ID, valBack, ok, s.Val)
		}
		return nil
	})
	if !reflect.DeepEqual(settings, got) {
		t.Errorf("Read settings %+v != written settings %+v", got, settings)
	}
}

func TestWriteSettingsAck(t *testing.T) {
	fr, buf := testFramer()
	fr.WriteSettingsAck()
	const wantEnc = "\x00\x00\x00\x04\x01\x00\x00\x00\x00"
	if buf.String() != wantEnc {
		t.Errorf("encoded as %q; want %q", buf.Bytes(), wantEnc)
	}
}

func TestWriteWindowUpdate(t *testing.T) {
	fr, buf := testFramer()
	const streamID = 1<<24 + 2<<16 + 3<<8 + 4
	const incr = 7<<24 + 6<<16 + 5<<8 + 4
	if err := fr.WriteWindowUpdate(streamID, incr); err != nil {
		t.Fatal(err)
	}
	const wantEnc = "\x00\x00\x04\x08\x00\x01\x02\x03\x04\x07\x06\x05\x04"
	if buf.String() != wantEnc {
		t.Errorf("encoded as %q; want %q", buf.Bytes(), wantEnc)
	}
	f, err := fr.ReadFrame()
	if err != nil {
		t.Fatal(err)
	}
	want := &WindowUpdateFrame{
		FrameHeader: FrameHeader{
			valid:    true,
			Type:     0x8,
			Flags:    0x0,
			Length:   0x4,
			StreamID: 0x1020304,
		},
		Increment: 0x7060504,
	}
	if !reflect.DeepEqual(f, want) {
		t.Errorf("parsed back %#v; want %#v", f, want)
	}
}

func TestWritePing(t *testing.T)    { testWritePing(t, false) }
func TestWritePingAck(t *testing.T) { testWritePing(t, true) }

func testWritePing(t *testing.T, ack bool) {
	fr, buf := testFramer()
	if err := fr.WritePing(ack, [8]byte{1, 2, 3, 4, 5, 6, 7, 8}); err != nil {
		t.Fatal(err)
	}
	var wantFlags Flags
	if ack {
		wantFlags = FlagPingAck
	}
	var wantEnc = "\x00\x00\x08\x06" + string(wantFlags) + "\x00\x00\x00\x00" + "\x01\x02\x03\x04\x05\x06\x07\x08"
	if buf.String() != wantEnc {
		t.Errorf("encoded as %q; want %q", buf.Bytes(), wantEnc)
	}

	f, err := fr.ReadFrame()
	if err != nil {
		t.Fatal(err)
	}
	want := &PingFrame{
		FrameHeader: FrameHeader{
			valid:    true,
			Type:     0x6,
			Flags:    wantFlags,
			Length:   0x8,
			StreamID: 0,
		},
		Data: [8]byte{1, 2, 3, 4, 5, 6, 7, 8},
	}
	if !reflect.DeepEqual(f, want) {
		t.Errorf("parsed back %#v; want %#v", f, want)
	}
}

func TestReadFrameHeader(t *testing.T) {
	tests := []struct {
		in   string
		want FrameHeader
	}{
		{in: "\x00\x00\x00" + "\x00" + "\x00" + "\x00\x00\x00\x00", want: FrameHeader{}},
		{in: "\x01\x02\x03" + "\x04" + "\x05" + "\x06\x07\x08\x09", want: FrameHeader{
			Length: 66051, Type: 4, Flags: 5, StreamID: 101124105,
		}},
		// Ignore high bit:
		{in: "\xff\xff\xff" + "\xff" + "\xff" + "\xff\xff\xff\xff", want: FrameHeader{
			Length: 16777215, Type: 255, Flags: 255, StreamID: 2147483647}},
		{in: "\xff\xff\xff" + "\xff" + "\xff" + "\x7f\xff\xff\xff", want: FrameHeader{
			Length: 16777215, Type: 255, Flags: 255, StreamID: 2147483647}},
	}
	for i, tt := range tests {
		got, err := readFrameHeader(make([]byte, 9), strings.NewReader(tt.in))
		if err != nil {
			t.Errorf("%d. readFrameHeader(%q) = %v", i, tt.in, err)
			continue
		}
		tt.want.valid = true
		if got != tt.want {
			t.Errorf("%d. readFrameHeader(%q) = %+v; want %+v", i, tt.in, got, tt.want)
		}
	}
}

func TestReadWriteFrameHeader(t *testing.T) {
	tests := []struct {
		len      uint32
		typ      FrameType
		flags    Flags
		streamID uint32
	}{
		{len: 0, typ: 255, flags: 1, streamID: 0},
		{len: 0, typ: 255, flags: 1, streamID: 1},
		{len: 0, typ: 255, flags: 1, streamID: 255},
		{len: 0, typ: 255, flags: 1, streamID: 256},
		{len: 0, typ: 255, flags: 1, streamID: 65535},
		{len: 0, typ: 255, flags: 1, streamID: 65536},

		{len: 0, typ: 1, flags: 255, streamID: 1},
		{len: 255, typ: 1, flags: 255, streamID: 1},
		{len: 256, typ: 1, flags: 255, streamID: 1},
		{len: 65535, typ: 1, flags: 255, streamID: 1},
		{len: 65536, typ: 1, flags: 255, streamID: 1},
		{len: 16777215, typ: 1, flags: 255, streamID: 1},
	}
	for _, tt := range tests {
		fr, buf := testFramer()
		fr.startWrite(tt.typ, tt.flags, tt.streamID)
		fr.writeBytes(make([]byte, tt.len))
		fr.endWrite()
		fh, err := ReadFrameHeader(buf)
		if err != nil {
			t.Errorf("ReadFrameHeader(%+v) = %v", tt, err)
			continue
		}
		if fh.Type != tt.typ || fh.Flags != tt.flags || fh.Length != tt.len || fh.StreamID != tt.streamID {
			t.Errorf("ReadFrameHeader(%+v) = %+v; mismatch", tt, fh)
		}
	}

}

func TestWriteTooLargeFrame(t *testing.T) {
	fr, _ := testFramer()
	fr.startWrite(0, 1, 1)
	fr.writeBytes(make([]byte, 1<<24))
	err := fr.endWrite()
	if err != ErrFrameTooLarge {
		t.Errorf("endWrite = %v; want errFrameTooLarge", err)
	}
}

func TestWriteGoAway(t *testing.T) {
	const debug = "foo"
	fr, buf := testFramer()
	if err := fr.WriteGoAway(0x01020304, 0x05060708, []byte(debug)); err != nil {
		t.Fatal(err)
	}
	const wantEnc = "\x00\x00\v\a\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08" + debug
	if buf.String() != wantEnc {
		t.Errorf("encoded as %q; want %q", buf.Bytes(), wantEnc)
	}
	f, err := fr.ReadFrame()
	if err != nil {
		t.Fatal(err)
	}
	want := &GoAwayFrame{
		FrameHeader: FrameHeader{
			valid:    true,
			Type:     0x7,
			Flags:    0,
			Length:   uint32(4 + 4 + len(debug)),
			StreamID: 0,
		},
		LastStreamID: 0x01020304,
		ErrCode:      0x05060708,
		debugData:    []byte(debug),
	}
	if !reflect.DeepEqual(f, want) {
		t.Fatalf("parsed back:\n%#v\nwant:\n%#v", f, want)
	}
	if got := string(f.(*GoAwayFrame).DebugData()); got != debug {
		t.Errorf("debug data = %q; want %q", got, debug)
	}
}

func TestWritePushPromise(t *testing.T) {
	pp := PushPromiseParam{
		StreamID:      42,
		PromiseID:     42,
		BlockFragment: []byte("abc"),
	}
	fr, buf := testFramer()
	if err := fr.WritePushPromise(pp); err != nil {
		t.Fatal(err)
	}
	const wantEnc = "\x00\x00\x07\x05\x00\x00\x00\x00*\x00\x00\x00*abc"
	if buf.String() != wantEnc {
		t.Errorf("encoded as %q; want %q", buf.Bytes(), wantEnc)
	}
	f, err := fr.ReadFrame()
	if err != nil {
		t.Fatal(err)
	}
	_, ok := f.(*PushPromiseFrame)
	if !ok {
		t.Fatalf("got %T; want *PushPromiseFrame", f)
	}
	want := &PushPromiseFrame{
		FrameHeader: FrameHeader{
			valid:    true,
			Type:     0x5,
			Flags:    0x0,
			Length:   0x7,
			StreamID: 42,
		},
		PromiseID:     42,
		headerFragBuf: []byte("abc"),
	}
	if !reflect.DeepEqual(f, want) {
		t.Fatalf("parsed back:\n%#v\nwant:\n%#v", f, want)
	}
}
