// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"bytes"
	"errors"
	"math"
	"strings"
	"sync"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/google/go-cmp/cmp"

	pb2 "github.com/golang/protobuf/internal/testprotos/proto2_proto"
	pb3 "github.com/golang/protobuf/internal/testprotos/proto3_proto"
	anypb "github.com/golang/protobuf/ptypes/any"
)

var (
	expandedMarshaler        = proto.TextMarshaler{ExpandAny: true}
	expandedCompactMarshaler = proto.TextMarshaler{Compact: true, ExpandAny: true}
)

// anyEqual reports whether two messages which may be google.protobuf.Any or may
// contain google.protobuf.Any fields are equal. We can't use proto.Equal for
// comparison, because semantically equivalent messages may be marshaled to
// binary in different tag order. Instead, trust that TextMarshaler with
// ExpandAny option works and compare the text marshaling results.
func anyEqual(got, want proto.Message) bool {
	// if messages are proto.Equal, no need to marshal.
	if proto.Equal(got, want) {
		return true
	}
	g := expandedMarshaler.Text(got)
	w := expandedMarshaler.Text(want)
	return g == w
}

type golden struct {
	m    proto.Message
	t, c string
}

var goldenMessages = makeGolden()

func makeGolden() []golden {
	nested := &pb3.Nested{Bunny: "Monty"}
	nb, err := proto.Marshal(nested)
	if err != nil {
		panic(err)
	}
	m1 := &pb3.Message{
		Name:        "David",
		ResultCount: 47,
		Anything:    &anypb.Any{TypeUrl: "type.googleapis.com/" + proto.MessageName(nested), Value: nb},
	}
	m2 := &pb3.Message{
		Name:        "David",
		ResultCount: 47,
		Anything:    &anypb.Any{TypeUrl: "http://[::1]/type.googleapis.com/" + proto.MessageName(nested), Value: nb},
	}
	m3 := &pb3.Message{
		Name:        "David",
		ResultCount: 47,
		Anything:    &anypb.Any{TypeUrl: `type.googleapis.com/"/` + proto.MessageName(nested), Value: nb},
	}
	m4 := &pb3.Message{
		Name:        "David",
		ResultCount: 47,
		Anything:    &anypb.Any{TypeUrl: "type.googleapis.com/a/path/" + proto.MessageName(nested), Value: nb},
	}
	m5 := &anypb.Any{TypeUrl: "type.googleapis.com/" + proto.MessageName(nested), Value: nb}

	any1 := &pb2.MyMessage{Count: proto.Int32(47), Name: proto.String("David")}
	proto.SetExtension(any1, pb2.E_Ext_More, &pb2.Ext{Data: proto.String("foo")})
	proto.SetExtension(any1, pb2.E_Ext_Text, proto.String("bar"))
	any1b, err := proto.Marshal(any1)
	if err != nil {
		panic(err)
	}
	any2 := &pb2.MyMessage{Count: proto.Int32(42), Bikeshed: pb2.MyMessage_GREEN.Enum(), RepBytes: [][]byte{[]byte("roboto")}}
	proto.SetExtension(any2, pb2.E_Ext_More, &pb2.Ext{Data: proto.String("baz")})
	any2b, err := proto.Marshal(any2)
	if err != nil {
		panic(err)
	}
	m6 := &pb3.Message{
		Name:        "David",
		ResultCount: 47,
		Anything:    &anypb.Any{TypeUrl: "type.googleapis.com/" + proto.MessageName(any1), Value: any1b},
		ManyThings: []*anypb.Any{
			&anypb.Any{TypeUrl: "type.googleapis.com/" + proto.MessageName(any2), Value: any2b},
			&anypb.Any{TypeUrl: "type.googleapis.com/" + proto.MessageName(any1), Value: any1b},
		},
	}

	const (
		m1Golden = `
name: "David"
result_count: 47
anything: <
  [type.googleapis.com/proto3_test.Nested]: <
    bunny: "Monty"
  >
>
`
		m2Golden = `
name: "David"
result_count: 47
anything: <
  ["http://[::1]/type.googleapis.com/proto3_test.Nested"]: <
    bunny: "Monty"
  >
>
`
		m3Golden = `
name: "David"
result_count: 47
anything: <
  ["type.googleapis.com/\"/proto3_test.Nested"]: <
    bunny: "Monty"
  >
>
`
		m4Golden = `
name: "David"
result_count: 47
anything: <
  [type.googleapis.com/a/path/proto3_test.Nested]: <
    bunny: "Monty"
  >
>
`
		m5Golden = `
[type.googleapis.com/proto3_test.Nested]: <
  bunny: "Monty"
>
`
		m6Golden = `
name: "David"
result_count: 47
anything: <
  [type.googleapis.com/proto2_test.MyMessage]: <
    count: 47
    name: "David"
    [proto2_test.Ext.more]: <
      data: "foo"
    >
    [proto2_test.Ext.text]: "bar"
  >
>
many_things: <
  [type.googleapis.com/proto2_test.MyMessage]: <
    count: 42
    bikeshed: GREEN
    rep_bytes: "roboto"
    [proto2_test.Ext.more]: <
      data: "baz"
    >
  >
>
many_things: <
  [type.googleapis.com/proto2_test.MyMessage]: <
    count: 47
    name: "David"
    [proto2_test.Ext.more]: <
      data: "foo"
    >
    [proto2_test.Ext.text]: "bar"
  >
>
`
	)
	return []golden{
		{m1, strings.TrimSpace(m1Golden) + "\n", strings.TrimSpace(compact(m1Golden)) + " "},
		{m2, strings.TrimSpace(m2Golden) + "\n", strings.TrimSpace(compact(m2Golden)) + " "},
		{m3, strings.TrimSpace(m3Golden) + "\n", strings.TrimSpace(compact(m3Golden)) + " "},
		{m4, strings.TrimSpace(m4Golden) + "\n", strings.TrimSpace(compact(m4Golden)) + " "},
		{m5, strings.TrimSpace(m5Golden) + "\n", strings.TrimSpace(compact(m5Golden)) + " "},
		{m6, strings.TrimSpace(m6Golden) + "\n", strings.TrimSpace(compact(m6Golden)) + " "},
	}
}

func TestMarshalGolden(t *testing.T) {
	for _, tt := range goldenMessages {
		t.Run("", func(t *testing.T) {
			if got, want := expandedMarshaler.Text(tt.m), tt.t; got != want {
				t.Errorf("message %v: got:\n%s\nwant:\n%s", tt.m, got, want)
			}
			if got, want := expandedCompactMarshaler.Text(tt.m), tt.c; got != want {
				t.Errorf("message %v: got:\n`%s`\nwant:\n`%s`", tt.m, got, want)
			}
		})
	}
}

func TestUnmarshalGolden(t *testing.T) {
	for _, tt := range goldenMessages {
		t.Run("", func(t *testing.T) {
			want := tt.m
			got := proto.Clone(tt.m)
			got.Reset()
			if err := proto.UnmarshalText(tt.t, got); err != nil {
				t.Errorf("failed to unmarshal\n%s\nerror: %v", tt.t, err)
			}
			if !anyEqual(got, want) {
				t.Errorf("message:\n%s\ngot:\n%s\nwant:\n%s", tt.t, got, want)
			}
			got.Reset()
			if err := proto.UnmarshalText(tt.c, got); err != nil {
				t.Errorf("failed to unmarshal\n%s\nerror: %v", tt.c, err)
			}
			if !anyEqual(got, want) {
				t.Errorf("message:\n%s\ngot:\n%s\nwant:\n%s", tt.c, got, want)
			}
		})
	}
}

func TestMarshalUnknownAny(t *testing.T) {
	m := &pb3.Message{
		Anything: &anypb.Any{
			TypeUrl: "foo",
			Value:   []byte("bar"),
		},
	}
	want := `anything: <
  type_url: "foo"
  value: "bar"
>
`
	got := expandedMarshaler.Text(m)
	if got != want {
		t.Errorf("got:\n%s\nwant:\n%s", got, want)
	}
}

func TestAmbiguousAny(t *testing.T) {
	pb := &anypb.Any{}
	err := proto.UnmarshalText(`
	type_url: "ttt/proto3_test.Nested"
	value: "\n\x05Monty"
	`, pb)
	if err != nil {
		t.Errorf("unexpected proto.UnmarshalText error: %v", err)
	}
}

func TestUnmarshalOverwriteAny(t *testing.T) {
	pb := &anypb.Any{}
	err := proto.UnmarshalText(`
  [type.googleapis.com/a/path/proto3_test.Nested]: <
    bunny: "Monty"
  >
  [type.googleapis.com/a/path/proto3_test.Nested]: <
    bunny: "Rabbit of Caerbannog"
  >
	`, pb)
	want := `line 7: Any message unpacked multiple times, or "type_url" already set`
	if err.Error() != want {
		t.Errorf("incorrect error:\ngot:  %v\nwant: %v", err.Error(), want)
	}
}

func TestUnmarshalAnyMixAndMatch(t *testing.T) {
	pb := &anypb.Any{}
	err := proto.UnmarshalText(`
	value: "\n\x05Monty"
  [type.googleapis.com/a/path/proto3_test.Nested]: <
    bunny: "Rabbit of Caerbannog"
  >
	`, pb)
	want := `line 5: Any message unpacked multiple times, or "value" already set`
	if err.Error() != want {
		t.Errorf("incorrect error:\ngot:  %v\nwant: %v", err.Error(), want)
	}
}

// textMessage implements the methods that allow it to marshal and unmarshal
// itself as text.
type textMessage struct {
}

func (*textMessage) MarshalText() ([]byte, error) {
	return []byte("custom"), nil
}

func (*textMessage) UnmarshalText(bytes []byte) error {
	if string(bytes) != "custom" {
		return errors.New("expected 'custom'")
	}
	return nil
}

func (*textMessage) Reset()         {}
func (*textMessage) String() string { return "" }
func (*textMessage) ProtoMessage()  {}

func newTestMessage() *pb2.MyMessage {
	msg := &pb2.MyMessage{
		Count: proto.Int32(42),
		Name:  proto.String("Dave"),
		Quote: proto.String(`"I didn't want to go."`),
		Pet:   []string{"bunny", "kitty", "horsey"},
		Inner: &pb2.InnerMessage{
			Host:      proto.String("footrest.syd"),
			Port:      proto.Int32(7001),
			Connected: proto.Bool(true),
		},
		Others: []*pb2.OtherMessage{
			{
				Key:   proto.Int64(0xdeadbeef),
				Value: []byte{1, 65, 7, 12},
			},
			{
				Weight: proto.Float32(6.022),
				Inner: &pb2.InnerMessage{
					Host: proto.String("lesha.mtv"),
					Port: proto.Int32(8002),
				},
			},
		},
		Bikeshed: pb2.MyMessage_BLUE.Enum(),
		Somegroup: &pb2.MyMessage_SomeGroup{
			GroupField: proto.Int32(8),
		},
		// One normally wouldn't do this.
		// This is an undeclared tag 13, as a varint (wire type 0) with value 4.
		XXX_unrecognized: []byte{13<<3 | 0, 4},
	}
	ext := &pb2.Ext{
		Data: proto.String("Big gobs for big rats"),
	}
	if err := proto.SetExtension(msg, pb2.E_Ext_More, ext); err != nil {
		panic(err)
	}
	greetings := []string{"adg", "easy", "cow"}
	if err := proto.SetExtension(msg, pb2.E_Greeting, greetings); err != nil {
		panic(err)
	}

	// Add an unknown extension. We marshal a pb2.Ext, and fake the ID.
	b, err := proto.Marshal(&pb2.Ext{Data: proto.String("3G skiing")})
	if err != nil {
		panic(err)
	}
	b = append(proto.EncodeVarint(201<<3|proto.WireBytes), b...)
	proto.SetRawExtension(msg, 201, b)

	// Extensions can be plain fields, too, so let's test that.
	b = append(proto.EncodeVarint(202<<3|proto.WireVarint), 19)
	proto.SetRawExtension(msg, 202, b)

	return msg
}

const text = `count: 42
name: "Dave"
quote: "\"I didn't want to go.\""
pet: "bunny"
pet: "kitty"
pet: "horsey"
inner: <
  host: "footrest.syd"
  port: 7001
  connected: true
>
others: <
  key: 3735928559
  value: "\001A\007\014"
>
others: <
  weight: 6.022
  inner: <
    host: "lesha.mtv"
    port: 8002
  >
>
bikeshed: BLUE
SomeGroup {
  group_field: 8
}
/* 18 unknown bytes */
13: 4
201: "\t3G skiing"
202: 19
[proto2_test.Ext.more]: <
  data: "Big gobs for big rats"
>
[proto2_test.greeting]: "adg"
[proto2_test.greeting]: "easy"
[proto2_test.greeting]: "cow"
`

func TestMarshalText(t *testing.T) {
	buf := new(bytes.Buffer)
	if err := proto.MarshalText(buf, newTestMessage()); err != nil {
		t.Fatalf("proto.MarshalText: %v", err)
	}
	got := buf.String()
	if diff := cmp.Diff(text, got); got != text {
		t.Errorf("diff (-want +got):\n%v\n\ngot:\n%v\n\nwant:\n%v", diff, got, text)
	}
}

func TestMarshalTextCustomMessage(t *testing.T) {
	buf := new(bytes.Buffer)
	if err := proto.MarshalText(buf, &textMessage{}); err != nil {
		t.Fatalf("proto.MarshalText: %v", err)
	}
	got := buf.String()
	if got != "custom" {
		t.Errorf("got:\n%v\n\nwant:\n%v", got, "custom")
	}
}
func TestMarshalTextNil(t *testing.T) {
	want := "<nil>"
	tests := []proto.Message{nil, (*pb2.MyMessage)(nil)}
	for i, test := range tests {
		buf := new(bytes.Buffer)
		if err := proto.MarshalText(buf, test); err != nil {
			t.Fatal(err)
		}
		if got := buf.String(); got != want {
			t.Errorf("%d: got %q want %q", i, got, want)
		}
	}
}

func TestMarshalTextUnknownEnum(t *testing.T) {
	// The Color enum only specifies values 0-2.
	m := &pb2.MyMessage{Bikeshed: pb2.MyMessage_Color(3).Enum()}
	got := m.String()
	const want = `bikeshed:3 `
	if got != want {
		t.Errorf("\n got %q\nwant %q", got, want)
	}
}

func TestTextOneof(t *testing.T) {
	tests := []struct {
		m    proto.Message
		want string
	}{
		// zero message
		{&pb2.Communique{}, ``},
		// scalar field
		{&pb2.Communique{Union: &pb2.Communique_Number{4}}, `number:4`},
		// message field
		{&pb2.Communique{Union: &pb2.Communique_Msg{
			&pb2.Strings{StringField: proto.String("why hello!")},
		}}, `msg:<string_field:"why hello!" >`},
		// bad oneof (should not panic)
		{&pb2.Communique{Union: &pb2.Communique_Msg{nil}}, `msg:<>`},
	}
	for _, test := range tests {
		got := strings.TrimSpace(test.m.String())
		if got != test.want {
			t.Errorf("got:\n%s\n\nwant:\n%s", got, test.want)
		}
	}
}

func compact(src string) string {
	// s/[ \n]+/ /g; s/ $//;
	dst := make([]byte, len(src))
	space, comment := false, false
	j := 0
	for i := 0; i < len(src); i++ {
		if strings.HasPrefix(src[i:], "/*") {
			comment = true
			i++
			continue
		}
		if comment && strings.HasPrefix(src[i:], "*/") {
			comment = false
			i++
			continue
		}
		if comment {
			continue
		}
		c := src[i]
		if c == ' ' || c == '\n' {
			space = true
			continue
		}
		if j > 0 && (dst[j-1] == ':' || dst[j-1] == '<' || dst[j-1] == '{') {
			space = false
		}
		if c == '{' {
			space = false
		}
		if space {
			dst[j] = ' '
			j++
			space = false
		}
		dst[j] = c
		j++
	}
	if space {
		dst[j] = ' '
		j++
	}
	return string(dst[0:j])
}

func TestCompactText(t *testing.T) {
	got := proto.CompactTextString(newTestMessage())
	if got != compact(text) {
		t.Errorf("got:\n%v\n\nwant:\n%v", got, compact(text))
	}
}

func TestStringEscaping(t *testing.T) {
	testCases := []struct {
		in  *pb2.Strings
		out string
	}{
		{
			// Test data from C++ test (TextFormatTest.StringEscape).
			// Single divergence: we don't escape apostrophes.
			&pb2.Strings{StringField: proto.String("\"A string with ' characters \n and \r newlines and \t tabs and \001 slashes \\ and  multiple   spaces")},
			"string_field: \"\\\"A string with ' characters \\n and \\r newlines and \\t tabs and \\001 slashes \\\\ and  multiple   spaces\"\n",
		},
		{
			// Test data from the same C++ test.
			&pb2.Strings{StringField: proto.String("\350\260\267\346\255\214")},
			"string_field: \"\\350\\260\\267\\346\\255\\214\"\n",
		},
		{
			// Some UTF-8.
			&pb2.Strings{StringField: proto.String("\x00\x01\xff\x81")},
			`string_field: "\000\001\377\201"` + "\n",
		},
	}

	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			var buf bytes.Buffer
			if err := proto.MarshalText(&buf, tc.in); err != nil {
				t.Fatalf("proto.MarsalText error: %v", err)
			}
			got := buf.String()
			if got != tc.out {
				t.Fatalf("want:\n%s\n\nwant:\n%s", got, tc.out)
			}

			// Check round-trip.
			pb := new(pb2.Strings)
			if err := proto.UnmarshalText(got, pb); err != nil {
				t.Fatalf("proto.UnmarshalText error: %v", err)
			}
			if !proto.Equal(pb, tc.in) {
				t.Fatalf("proto.Equal mismatch:\ngot:\n%v\n\nwant:\n%v", pb, tc.in)
			}
		})
	}
}

// A limitedWriter accepts some output before it fails.
// This is a proxy for something like a nearly-full or imminently-failing disk,
// or a network connection that is about to die.
type limitedWriter struct {
	b     bytes.Buffer
	limit int
}

var outOfSpace = errors.New("proto: insufficient space")

func (w *limitedWriter) Write(p []byte) (n int, err error) {
	var avail = w.limit - w.b.Len()
	if avail <= 0 {
		return 0, outOfSpace
	}
	if len(p) <= avail {
		return w.b.Write(p)
	}
	n, _ = w.b.Write(p[:avail])
	return n, outOfSpace
}

func TestMarshalTextFailing(t *testing.T) {
	// Try lots of different sizes to exercise more error code-paths.
	for lim := 0; lim < len(text); lim++ {
		buf := new(limitedWriter)
		buf.limit = lim
		err := proto.MarshalText(buf, newTestMessage())
		// We expect a certain error, but also some partial results in the buffer.
		if err != outOfSpace {
			t.Errorf("error mismatch: got %v, want %v", err, outOfSpace)
		}
		got := buf.b.String()
		want := text[:buf.limit]
		if got != want {
			t.Errorf("text mismatch:\n\ngot:\n%v\n\nwant:\n%v", got, want)
		}
	}
}

func TestFloats(t *testing.T) {
	tests := []struct {
		f    float64
		want string
	}{
		{0, "0"},
		{4.7, "4.7"},
		{math.Inf(1), "inf"},
		{math.Inf(-1), "-inf"},
		{math.NaN(), "nan"},
	}
	for _, test := range tests {
		msg := &pb2.FloatingPoint{F: &test.f}
		got := strings.TrimSpace(msg.String())
		want := `f:` + test.want
		if got != want {
			t.Errorf("f=%f: got %q, want %q", test.f, got, want)
		}
	}
}

func TestRepeatedNilText(t *testing.T) {
	m := &pb2.MessageList{
		Message: []*pb2.MessageList_Message{
			nil,
			&pb2.MessageList_Message{
				Name: proto.String("Horse"),
			},
			nil,
		},
	}
	want := `Message {
}
Message {
  name: "Horse"
}
Message {
}
`
	if got := proto.MarshalTextString(m); got != want {
		t.Errorf("got:\n%s\n\nwant:\n%s", got, want)
	}
}

func TestProto3Text(t *testing.T) {
	tests := []struct {
		m    proto.Message
		want string
	}{
		// zero message
		{&pb3.Message{}, ``},
		// zero message except for an empty byte slice
		{&pb3.Message{Data: []byte{}}, ``},
		// trivial case
		{&pb3.Message{Name: "Rob", HeightInCm: 175}, `name:"Rob" height_in_cm:175`},
		// empty map
		{&pb2.MessageWithMap{}, ``},
		// non-empty map; map format is the same as a repeated struct,
		// and they are sorted by key (numerically for numeric keys).
		{
			&pb2.MessageWithMap{NameMapping: map[int32]string{
				-1:      "Negatory",
				7:       "Lucky",
				1234:    "Feist",
				6345789: "Otis",
			}},
			`name_mapping:<key:-1 value:"Negatory" > ` +
				`name_mapping:<key:7 value:"Lucky" > ` +
				`name_mapping:<key:1234 value:"Feist" > ` +
				`name_mapping:<key:6345789 value:"Otis" >`,
		},
		// map with nil value; not well-defined, but we shouldn't crash
		{
			&pb2.MessageWithMap{MsgMapping: map[int64]*pb2.FloatingPoint{7: nil}},
			`msg_mapping:<key:7 value:<> >`,
		},
	}
	for _, test := range tests {
		got := strings.TrimSpace(test.m.String())
		if got != test.want {
			t.Errorf("got:\n%s\n\nwant:\n%s", got, test.want)
		}
	}
}

func TestRacyMarshal(t *testing.T) {
	// This test should be run with the race detector.

	any := &pb2.MyMessage{Count: proto.Int32(47), Name: proto.String("David")}
	proto.SetExtension(any, pb2.E_Ext_Text, proto.String("bar"))
	b, err := proto.Marshal(any)
	if err != nil {
		panic(err)
	}
	m := &pb3.Message{
		Name:        "David",
		ResultCount: 47,
		Anything:    &anypb.Any{TypeUrl: "type.googleapis.com/" + proto.MessageName(any), Value: b},
	}

	wantText := proto.MarshalTextString(m)
	wantBytes, err := proto.Marshal(m)
	if err != nil {
		t.Fatalf("proto.Marshal error: %v", err)
	}

	var wg sync.WaitGroup
	defer wg.Wait()
	wg.Add(20)
	for i := 0; i < 10; i++ {
		go func() {
			defer wg.Done()
			got := proto.MarshalTextString(m)
			if got != wantText {
				t.Errorf("proto.MarshalTextString = %q, want %q", got, wantText)
			}
		}()
		go func() {
			defer wg.Done()
			got, err := proto.Marshal(m)
			if !bytes.Equal(got, wantBytes) || err != nil {
				t.Errorf("proto.Marshal = (%x, %v), want (%x, nil)", got, err, wantBytes)
			}
		}()
	}
}

type UnmarshalTextTest struct {
	in  string
	err string // if "", no error expected
	out *pb2.MyMessage
}

func buildExtStructTest(text string) UnmarshalTextTest {
	msg := &pb2.MyMessage{
		Count: proto.Int32(42),
	}
	proto.SetExtension(msg, pb2.E_Ext_More, &pb2.Ext{
		Data: proto.String("Hello, world!"),
	})
	return UnmarshalTextTest{in: text, out: msg}
}

func buildExtDataTest(text string) UnmarshalTextTest {
	msg := &pb2.MyMessage{
		Count: proto.Int32(42),
	}
	proto.SetExtension(msg, pb2.E_Ext_Text, proto.String("Hello, world!"))
	proto.SetExtension(msg, pb2.E_Ext_Number, proto.Int32(1729))
	return UnmarshalTextTest{in: text, out: msg}
}

func buildExtRepStringTest(text string) UnmarshalTextTest {
	msg := &pb2.MyMessage{
		Count: proto.Int32(42),
	}
	if err := proto.SetExtension(msg, pb2.E_Greeting, []string{"bula", "hola"}); err != nil {
		panic(err)
	}
	return UnmarshalTextTest{in: text, out: msg}
}

var unmarshalTextTests = []UnmarshalTextTest{
	// Basic
	{
		in: " count:42\n  name:\"Dave\" ",
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("Dave"),
		},
	},

	// Empty quoted string
	{
		in: `count:42 name:""`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String(""),
		},
	},

	// Quoted string concatenation with double quotes
	{
		in: `count:42 name: "My name is "` + "\n" + `"elsewhere"`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("My name is elsewhere"),
		},
	},

	// Quoted string concatenation with single quotes
	{
		in: "count:42 name: 'My name is '\n'elsewhere'",
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("My name is elsewhere"),
		},
	},

	// Quoted string concatenations with mixed quotes
	{
		in: "count:42 name: 'My name is '\n\"elsewhere\"",
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("My name is elsewhere"),
		},
	},
	{
		in: "count:42 name: \"My name is \"\n'elsewhere'",
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("My name is elsewhere"),
		},
	},

	// Quoted string with escaped apostrophe
	{
		in: `count:42 name: "HOLIDAY - New Year\'s Day"`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("HOLIDAY - New Year's Day"),
		},
	},

	// Quoted string with single quote
	{
		in: `count:42 name: 'Roger "The Ramster" Ramjet'`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String(`Roger "The Ramster" Ramjet`),
		},
	},

	// Quoted string with all the accepted special characters from the C++ test
	{
		in: `count:42 name: ` + "\"\\\"A string with \\' characters \\n and \\r newlines and \\t tabs and \\001 slashes \\\\ and  multiple   spaces\"",
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("\"A string with ' characters \n and \r newlines and \t tabs and \001 slashes \\ and  multiple   spaces"),
		},
	},

	// Quoted string with quoted backslash
	{
		in: `count:42 name: "\\'xyz"`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String(`\'xyz`),
		},
	},

	// Quoted string with UTF-8 bytes.
	{
		in: "count:42 name: '\303\277\302\201\x00\xAB\xCD\xEF'",
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("\303\277\302\201\x00\xAB\xCD\xEF"),
		},
	},

	// Quoted string with unicode escapes.
	{
		in: `count: 42 name: "\u0047\U00000047\uffff\U0010ffff"`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("GG\uffff\U0010ffff"),
		},
	},

	// Bad quoted string
	{
		in:  `inner: < host: "\0" >` + "\n",
		err: `line 1.15: invalid quoted string "\0": \0 requires 2 following digits`,
	},

	// Bad \u escape
	{
		in:  `count: 42 name: "\u000"`,
		err: `line 1.16: invalid quoted string "\u000": \u requires 4 following digits`,
	},

	// Bad \U escape
	{
		in:  `count: 42 name: "\U0000000"`,
		err: `line 1.16: invalid quoted string "\U0000000": \U requires 8 following digits`,
	},

	// Bad \U escape
	{
		in:  `count: 42 name: "\xxx"`,
		err: `line 1.16: invalid quoted string "\xxx": \xxx contains non-hexadecimal digits`,
	},

	// Number too large for int64
	{
		in:  "count: 1 others { key: 123456789012345678901 }",
		err: "line 1.23: invalid int64: 123456789012345678901",
	},

	// Number too large for int32
	{
		in:  "count: 1234567890123",
		err: "line 1.7: invalid int32: 1234567890123",
	},

	// Number in hexadecimal
	{
		in: "count: 0x2beef",
		out: &pb2.MyMessage{
			Count: proto.Int32(0x2beef),
		},
	},

	// Number in octal
	{
		in: "count: 024601",
		out: &pb2.MyMessage{
			Count: proto.Int32(024601),
		},
	},

	// Floating point number with "f" suffix
	{
		in: "count: 4 others:< weight: 17.0f >",
		out: &pb2.MyMessage{
			Count: proto.Int32(4),
			Others: []*pb2.OtherMessage{
				{
					Weight: proto.Float32(17),
				},
			},
		},
	},

	// Floating point positive infinity
	{
		in: "count: 4 bigfloat: inf",
		out: &pb2.MyMessage{
			Count:    proto.Int32(4),
			Bigfloat: proto.Float64(math.Inf(1)),
		},
	},

	// Floating point negative infinity
	{
		in: "count: 4 bigfloat: -inf",
		out: &pb2.MyMessage{
			Count:    proto.Int32(4),
			Bigfloat: proto.Float64(math.Inf(-1)),
		},
	},

	// Number too large for float32
	{
		in:  "others:< weight: 12345678901234567890123456789012345678901234567890 >",
		err: "line 1.17: invalid float: 12345678901234567890123456789012345678901234567890",
	},

	// Number posing as a quoted string
	{
		in:  `inner: < host: 12 >` + "\n",
		err: `line 1.15: invalid string: 12`,
	},

	// Quoted string posing as int32
	{
		in:  `count: "12"`,
		err: `line 1.7: invalid int32: "12"`,
	},

	// Quoted string posing a float32
	{
		in:  `others:< weight: "17.4" >`,
		err: `line 1.17: invalid float: "17.4"`,
	},

	// unclosed bracket doesn't cause infinite loop
	{
		in:  `[`,
		err: `line 1.0: unclosed type_url or extension name`,
	},

	// Enum
	{
		in: `count:42 bikeshed: BLUE`,
		out: &pb2.MyMessage{
			Count:    proto.Int32(42),
			Bikeshed: pb2.MyMessage_BLUE.Enum(),
		},
	},

	// Repeated field
	{
		in: `count:42 pet: "horsey" pet:"bunny"`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Pet:   []string{"horsey", "bunny"},
		},
	},

	// Repeated field with list notation
	{
		in: `count:42 pet: ["horsey", "bunny"]`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Pet:   []string{"horsey", "bunny"},
		},
	},

	// Repeated message with/without colon and <>/{}
	{
		in: `count:42 others:{} others{} others:<> others:{}`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Others: []*pb2.OtherMessage{
				{},
				{},
				{},
				{},
			},
		},
	},

	// Missing colon for inner message
	{
		in: `count:42 inner < host: "cauchy.syd" >`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Inner: &pb2.InnerMessage{
				Host: proto.String("cauchy.syd"),
			},
		},
	},

	// Missing colon for string field
	{
		in:  `name "Dave"`,
		err: `line 1.5: expected ':', found "\"Dave\""`,
	},

	// Missing colon for int32 field
	{
		in:  `count 42`,
		err: `line 1.6: expected ':', found "42"`,
	},

	// Missing required field
	{
		in:  `name: "Pawel"`,
		err: `required field proto2_test.MyMessage.count not set`,
		out: &pb2.MyMessage{
			Name: proto.String("Pawel"),
		},
	},

	// Missing required field in a required submessage
	{
		in:  `count: 42 we_must_go_deeper < leo_finally_won_an_oscar <> >`,
		err: `required field proto2_test.InnerMessage.host not set`,
		out: &pb2.MyMessage{
			Count:          proto.Int32(42),
			WeMustGoDeeper: &pb2.RequiredInnerMessage{LeoFinallyWonAnOscar: &pb2.InnerMessage{}},
		},
	},

	// Repeated non-repeated field
	{
		in:  `name: "Rob" name: "Russ"`,
		err: `line 1.12: non-repeated field "name" was repeated`,
	},

	// Group
	{
		in: `count: 17 SomeGroup { group_field: 12 }`,
		out: &pb2.MyMessage{
			Count: proto.Int32(17),
			Somegroup: &pb2.MyMessage_SomeGroup{
				GroupField: proto.Int32(12),
			},
		},
	},

	// Semicolon between fields
	{
		in: `count:3;name:"Calvin"`,
		out: &pb2.MyMessage{
			Count: proto.Int32(3),
			Name:  proto.String("Calvin"),
		},
	},
	// Comma between fields
	{
		in: `count:4,name:"Ezekiel"`,
		out: &pb2.MyMessage{
			Count: proto.Int32(4),
			Name:  proto.String("Ezekiel"),
		},
	},

	// Boolean false
	{
		in: `count:42 inner { host: "example.com" connected: false }`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Inner: &pb2.InnerMessage{
				Host:      proto.String("example.com"),
				Connected: proto.Bool(false),
			},
		},
	},
	// Boolean true
	{
		in: `count:42 inner { host: "example.com" connected: true }`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Inner: &pb2.InnerMessage{
				Host:      proto.String("example.com"),
				Connected: proto.Bool(true),
			},
		},
	},
	// Boolean 0
	{
		in: `count:42 inner { host: "example.com" connected: 0 }`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Inner: &pb2.InnerMessage{
				Host:      proto.String("example.com"),
				Connected: proto.Bool(false),
			},
		},
	},
	// Boolean 1
	{
		in: `count:42 inner { host: "example.com" connected: 1 }`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Inner: &pb2.InnerMessage{
				Host:      proto.String("example.com"),
				Connected: proto.Bool(true),
			},
		},
	},
	// Boolean f
	{
		in: `count:42 inner { host: "example.com" connected: f }`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Inner: &pb2.InnerMessage{
				Host:      proto.String("example.com"),
				Connected: proto.Bool(false),
			},
		},
	},
	// Boolean t
	{
		in: `count:42 inner { host: "example.com" connected: t }`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Inner: &pb2.InnerMessage{
				Host:      proto.String("example.com"),
				Connected: proto.Bool(true),
			},
		},
	},
	// Boolean False
	{
		in: `count:42 inner { host: "example.com" connected: False }`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Inner: &pb2.InnerMessage{
				Host:      proto.String("example.com"),
				Connected: proto.Bool(false),
			},
		},
	},
	// Boolean True
	{
		in: `count:42 inner { host: "example.com" connected: True }`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Inner: &pb2.InnerMessage{
				Host:      proto.String("example.com"),
				Connected: proto.Bool(true),
			},
		},
	},

	// Extension
	buildExtStructTest(`count: 42 [proto2_test.Ext.more]:<data:"Hello, world!" >`),
	buildExtStructTest(`count: 42 [proto2_test.Ext.more] {data:"Hello, world!"}`),
	buildExtDataTest(`count: 42 [proto2_test.Ext.text]:"Hello, world!" [proto2_test.Ext.number]:1729`),
	buildExtRepStringTest(`count: 42 [proto2_test.greeting]:"bula" [proto2_test.greeting]:"hola"`),
	{
		in:  `[proto2_test.complex]:<>`,
		err: `line 1.20: extension field "proto2_test.complex" does not extend message "proto2_test.MyMessage"`,
	},

	// Big all-in-one
	{
		in: "count:42  # Meaning\n" +
			`name:"Dave" ` +
			`quote:"\"I didn't want to go.\"" ` +
			`pet:"bunny" ` +
			`pet:"kitty" ` +
			`pet:"horsey" ` +
			`inner:<` +
			`  host:"footrest.syd" ` +
			`  port:7001 ` +
			`  connected:true ` +
			`> ` +
			`others:<` +
			`  key:3735928559 ` +
			`  value:"\x01A\a\f" ` +
			`> ` +
			`others:<` +
			"  weight:58.9  # Atomic weight of Co\n" +
			`  inner:<` +
			`    host:"lesha.mtv" ` +
			`    port:8002 ` +
			`  >` +
			`>`,
		out: &pb2.MyMessage{
			Count: proto.Int32(42),
			Name:  proto.String("Dave"),
			Quote: proto.String(`"I didn't want to go."`),
			Pet:   []string{"bunny", "kitty", "horsey"},
			Inner: &pb2.InnerMessage{
				Host:      proto.String("footrest.syd"),
				Port:      proto.Int32(7001),
				Connected: proto.Bool(true),
			},
			Others: []*pb2.OtherMessage{
				{
					Key:   proto.Int64(3735928559),
					Value: []byte{0x1, 'A', '\a', '\f'},
				},
				{
					Weight: proto.Float32(58.9),
					Inner: &pb2.InnerMessage{
						Host: proto.String("lesha.mtv"),
						Port: proto.Int32(8002),
					},
				},
			},
		},
	},
}

func TestUnmarshalText(t *testing.T) {
	for _, test := range unmarshalTextTests {
		t.Run("", func(t *testing.T) {
			pb := new(pb2.MyMessage)
			err := proto.UnmarshalText(test.in, pb)
			if test.err == "" {
				// We don't expect failure.
				if err != nil {
					t.Errorf("proto.UnmarshalText error: %v", err)
				} else if !proto.Equal(pb, test.out) {
					t.Errorf("proto.Equal mismatch:\ngot:  %v\nwant: %v", pb, test.out)
				}
			} else {
				// We do expect failure.
				if err == nil {
					t.Errorf("proto.UnmarshalText: got nil error, want %v", test.err)
				} else if !strings.Contains(err.Error(), test.err) {
					t.Errorf("proto.UnmarshalText error mismatch:\ngot:  %v\nwant: %v", err.Error(), test.err)
				} else if _, ok := err.(*proto.RequiredNotSetError); ok && test.out != nil && !proto.Equal(pb, test.out) {
					t.Errorf("proto.Equal mismatch:\ngot  %v\nwant: %v", pb, test.out)
				}
			}
		})
	}
}

func TestUnmarshalTextCustomMessage(t *testing.T) {
	msg := &textMessage{}
	if err := proto.UnmarshalText("custom", msg); err != nil {
		t.Errorf("proto.UnmarshalText error: %v", err)
	}
	if err := proto.UnmarshalText("not custom", msg); err == nil {
		t.Errorf("proto.UnmarshalText: got nil error, want non-nil")
	}
}

// Regression test; this caused a panic.
func TestRepeatedEnum(t *testing.T) {
	pb := new(pb2.RepeatedEnum)
	if err := proto.UnmarshalText("color: RED", pb); err != nil {
		t.Fatal(err)
	}
	exp := &pb2.RepeatedEnum{
		Color: []pb2.RepeatedEnum_Color{pb2.RepeatedEnum_RED},
	}
	if !proto.Equal(pb, exp) {
		t.Errorf("proto.Equal mismatch:\ngot:  %v\nwant %v", pb, exp)
	}
}

func TestProto3TextParsing(t *testing.T) {
	m := new(pb3.Message)
	const in = `name: "Wallace" true_scotsman: true`
	want := &pb3.Message{
		Name:         "Wallace",
		TrueScotsman: true,
	}
	if err := proto.UnmarshalText(in, m); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(m, want) {
		t.Errorf("proto.Equal mismatch:\ngot:  %v\nwant %v", m, want)
	}
}

func TestMapParsing(t *testing.T) {
	m := new(pb2.MessageWithMap)
	const in = `name_mapping:<key:1234 value:"Feist"> name_mapping:<key:1 value:"Beatles">` +
		`msg_mapping:<key:-4, value:<f: 2.0>,>` + // separating commas are okay
		`msg_mapping<key:-2 value<f: 4.0>>` + // no colon after "value"
		`msg_mapping:<value:<f: 5.0>>` + // omitted key
		`byte_mapping:<key:true value:"so be it">` +
		`byte_mapping:<>` // omitted key and value
	want := &pb2.MessageWithMap{
		NameMapping: map[int32]string{
			1:    "Beatles",
			1234: "Feist",
		},
		MsgMapping: map[int64]*pb2.FloatingPoint{
			-4: {F: proto.Float64(2.0)},
			-2: {F: proto.Float64(4.0)},
			0:  {F: proto.Float64(5.0)},
		},
		ByteMapping: map[bool][]byte{
			false: nil,
			true:  []byte("so be it"),
		},
	}
	if err := proto.UnmarshalText(in, m); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(m, want) {
		t.Errorf("proto.Equal mismatch:\ngot:  %v\nwant %v", m, want)
	}
}

func TestOneofParsing(t *testing.T) {
	const in = `name:"Shrek"`
	m := new(pb2.Communique)
	want := &pb2.Communique{Union: &pb2.Communique_Name{"Shrek"}}
	if err := proto.UnmarshalText(in, m); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(m, want) {
		t.Errorf("\n got %v\nwant %v", m, want)
	}

	const inOverwrite = `name:"Shrek" number:42`
	m = new(pb2.Communique)
	testErr := "line 1.13: field 'number' would overwrite already parsed oneof 'union'"
	if err := proto.UnmarshalText(inOverwrite, m); err == nil {
		t.Errorf("proto.UnmarshalText: got nil error, want %v", testErr)
	} else if err.Error() != testErr {
		t.Errorf("error mismatch:\ngot:  %v\nwant: %v", err.Error(), testErr)
	}
}
