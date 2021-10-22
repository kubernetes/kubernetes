// Copyright 2016 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	"bytes"
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
	"unsafe"
)

type test struct {
	in      string
	version Version
	variant Variant
	isuuid  bool
}

var tests = []test{
	{"f47ac10b-58cc-0372-8567-0e02b2c3d479", 0, RFC4122, true},
	{"f47ac10b-58cc-1372-8567-0e02b2c3d479", 1, RFC4122, true},
	{"f47ac10b-58cc-2372-8567-0e02b2c3d479", 2, RFC4122, true},
	{"f47ac10b-58cc-3372-8567-0e02b2c3d479", 3, RFC4122, true},
	{"f47ac10b-58cc-4372-8567-0e02b2c3d479", 4, RFC4122, true},
	{"f47ac10b-58cc-5372-8567-0e02b2c3d479", 5, RFC4122, true},
	{"f47ac10b-58cc-6372-8567-0e02b2c3d479", 6, RFC4122, true},
	{"f47ac10b-58cc-7372-8567-0e02b2c3d479", 7, RFC4122, true},
	{"f47ac10b-58cc-8372-8567-0e02b2c3d479", 8, RFC4122, true},
	{"f47ac10b-58cc-9372-8567-0e02b2c3d479", 9, RFC4122, true},
	{"f47ac10b-58cc-a372-8567-0e02b2c3d479", 10, RFC4122, true},
	{"f47ac10b-58cc-b372-8567-0e02b2c3d479", 11, RFC4122, true},
	{"f47ac10b-58cc-c372-8567-0e02b2c3d479", 12, RFC4122, true},
	{"f47ac10b-58cc-d372-8567-0e02b2c3d479", 13, RFC4122, true},
	{"f47ac10b-58cc-e372-8567-0e02b2c3d479", 14, RFC4122, true},
	{"f47ac10b-58cc-f372-8567-0e02b2c3d479", 15, RFC4122, true},

	{"urn:uuid:f47ac10b-58cc-4372-0567-0e02b2c3d479", 4, Reserved, true},
	{"URN:UUID:f47ac10b-58cc-4372-0567-0e02b2c3d479", 4, Reserved, true},
	{"f47ac10b-58cc-4372-0567-0e02b2c3d479", 4, Reserved, true},
	{"f47ac10b-58cc-4372-1567-0e02b2c3d479", 4, Reserved, true},
	{"f47ac10b-58cc-4372-2567-0e02b2c3d479", 4, Reserved, true},
	{"f47ac10b-58cc-4372-3567-0e02b2c3d479", 4, Reserved, true},
	{"f47ac10b-58cc-4372-4567-0e02b2c3d479", 4, Reserved, true},
	{"f47ac10b-58cc-4372-5567-0e02b2c3d479", 4, Reserved, true},
	{"f47ac10b-58cc-4372-6567-0e02b2c3d479", 4, Reserved, true},
	{"f47ac10b-58cc-4372-7567-0e02b2c3d479", 4, Reserved, true},
	{"f47ac10b-58cc-4372-8567-0e02b2c3d479", 4, RFC4122, true},
	{"f47ac10b-58cc-4372-9567-0e02b2c3d479", 4, RFC4122, true},
	{"f47ac10b-58cc-4372-a567-0e02b2c3d479", 4, RFC4122, true},
	{"f47ac10b-58cc-4372-b567-0e02b2c3d479", 4, RFC4122, true},
	{"f47ac10b-58cc-4372-c567-0e02b2c3d479", 4, Microsoft, true},
	{"f47ac10b-58cc-4372-d567-0e02b2c3d479", 4, Microsoft, true},
	{"f47ac10b-58cc-4372-e567-0e02b2c3d479", 4, Future, true},
	{"f47ac10b-58cc-4372-f567-0e02b2c3d479", 4, Future, true},

	{"f47ac10b158cc-5372-a567-0e02b2c3d479", 0, Invalid, false},
	{"f47ac10b-58cc25372-a567-0e02b2c3d479", 0, Invalid, false},
	{"f47ac10b-58cc-53723a567-0e02b2c3d479", 0, Invalid, false},
	{"f47ac10b-58cc-5372-a56740e02b2c3d479", 0, Invalid, false},
	{"f47ac10b-58cc-5372-a567-0e02-2c3d479", 0, Invalid, false},
	{"g47ac10b-58cc-4372-a567-0e02b2c3d479", 0, Invalid, false},

	{"{f47ac10b-58cc-0372-8567-0e02b2c3d479}", 0, RFC4122, true},
	{"{f47ac10b-58cc-0372-8567-0e02b2c3d479", 0, Invalid, false},
	{"f47ac10b-58cc-0372-8567-0e02b2c3d479}", 0, Invalid, false},

	{"f47ac10b58cc037285670e02b2c3d479", 0, RFC4122, true},
	{"f47ac10b58cc037285670e02b2c3d4790", 0, Invalid, false},
	{"f47ac10b58cc037285670e02b2c3d47", 0, Invalid, false},
}

var constants = []struct {
	c    interface{}
	name string
}{
	{Person, "Person"},
	{Group, "Group"},
	{Org, "Org"},
	{Invalid, "Invalid"},
	{RFC4122, "RFC4122"},
	{Reserved, "Reserved"},
	{Microsoft, "Microsoft"},
	{Future, "Future"},
	{Domain(17), "Domain17"},
	{Variant(42), "BadVariant42"},
}

func testTest(t *testing.T, in string, tt test) {
	uuid, err := Parse(in)
	if ok := (err == nil); ok != tt.isuuid {
		t.Errorf("Parse(%s) got %v expected %v\b", in, ok, tt.isuuid)
	}
	if err != nil {
		return
	}

	if v := uuid.Variant(); v != tt.variant {
		t.Errorf("Variant(%s) got %d expected %d\b", in, v, tt.variant)
	}
	if v := uuid.Version(); v != tt.version {
		t.Errorf("Version(%s) got %d expected %d\b", in, v, tt.version)
	}
}

func testBytes(t *testing.T, in []byte, tt test) {
	uuid, err := ParseBytes(in)
	if ok := (err == nil); ok != tt.isuuid {
		t.Errorf("ParseBytes(%s) got %v expected %v\b", in, ok, tt.isuuid)
	}
	if err != nil {
		return
	}
	suuid, _ := Parse(string(in))
	if uuid != suuid {
		t.Errorf("ParseBytes(%s) got %v expected %v\b", in, uuid, suuid)
	}
}

func TestUUID(t *testing.T) {
	for _, tt := range tests {
		testTest(t, tt.in, tt)
		testTest(t, strings.ToUpper(tt.in), tt)
		testBytes(t, []byte(tt.in), tt)
	}
}

func TestFromBytes(t *testing.T) {
	b := []byte{
		0x7d, 0x44, 0x48, 0x40,
		0x9d, 0xc0,
		0x11, 0xd1,
		0xb2, 0x45,
		0x5f, 0xfd, 0xce, 0x74, 0xfa, 0xd2,
	}
	uuid, err := FromBytes(b)
	if err != nil {
		t.Fatalf("%s", err)
	}
	for i := 0; i < len(uuid); i++ {
		if b[i] != uuid[i] {
			t.Fatalf("FromBytes() got %v expected %v\b", uuid[:], b)
		}
	}
}

func TestConstants(t *testing.T) {
	for x, tt := range constants {
		v, ok := tt.c.(fmt.Stringer)
		if !ok {
			t.Errorf("%x: %v: not a stringer", x, v)
		} else if s := v.String(); s != tt.name {
			v, _ := tt.c.(int)
			t.Errorf("%x: Constant %T:%d gives %q, expected %q", x, tt.c, v, s, tt.name)
		}
	}
}

func TestRandomUUID(t *testing.T) {
	m := make(map[string]bool)
	for x := 1; x < 32; x++ {
		uuid := New()
		s := uuid.String()
		if m[s] {
			t.Errorf("NewRandom returned duplicated UUID %s", s)
		}
		m[s] = true
		if v := uuid.Version(); v != 4 {
			t.Errorf("Random UUID of version %s", v)
		}
		if uuid.Variant() != RFC4122 {
			t.Errorf("Random UUID is variant %d", uuid.Variant())
		}
	}
}

func TestNew(t *testing.T) {
	m := make(map[UUID]bool)
	for x := 1; x < 32; x++ {
		s := New()
		if m[s] {
			t.Errorf("New returned duplicated UUID %s", s)
		}
		m[s] = true
		uuid, err := Parse(s.String())
		if err != nil {
			t.Errorf("New.String() returned %q which does not decode", s)
			continue
		}
		if v := uuid.Version(); v != 4 {
			t.Errorf("Random UUID of version %s", v)
		}
		if uuid.Variant() != RFC4122 {
			t.Errorf("Random UUID is variant %d", uuid.Variant())
		}
	}
}

func TestClockSeq(t *testing.T) {
	// Fake time.Now for this test to return a monotonically advancing time; restore it at end.
	defer func(orig func() time.Time) { timeNow = orig }(timeNow)
	monTime := time.Now()
	timeNow = func() time.Time {
		monTime = monTime.Add(1 * time.Second)
		return monTime
	}

	SetClockSequence(-1)
	uuid1, err := NewUUID()
	if err != nil {
		t.Fatalf("could not create UUID: %v", err)
	}
	uuid2, err := NewUUID()
	if err != nil {
		t.Fatalf("could not create UUID: %v", err)
	}

	if s1, s2 := uuid1.ClockSequence(), uuid2.ClockSequence(); s1 != s2 {
		t.Errorf("clock sequence %d != %d", s1, s2)
	}

	SetClockSequence(-1)
	uuid2, err = NewUUID()
	if err != nil {
		t.Fatalf("could not create UUID: %v", err)
	}

	// Just on the very off chance we generated the same sequence
	// two times we try again.
	if uuid1.ClockSequence() == uuid2.ClockSequence() {
		SetClockSequence(-1)
		uuid2, err = NewUUID()
		if err != nil {
			t.Fatalf("could not create UUID: %v", err)
		}
	}
	if s1, s2 := uuid1.ClockSequence(), uuid2.ClockSequence(); s1 == s2 {
		t.Errorf("Duplicate clock sequence %d", s1)
	}

	SetClockSequence(0x1234)
	uuid1, err = NewUUID()
	if err != nil {
		t.Fatalf("could not create UUID: %v", err)
	}
	if seq := uuid1.ClockSequence(); seq != 0x1234 {
		t.Errorf("%s: expected seq 0x1234 got 0x%04x", uuid1, seq)
	}
}

func TestCoding(t *testing.T) {
	text := "7d444840-9dc0-11d1-b245-5ffdce74fad2"
	urn := "urn:uuid:7d444840-9dc0-11d1-b245-5ffdce74fad2"
	data := UUID{
		0x7d, 0x44, 0x48, 0x40,
		0x9d, 0xc0,
		0x11, 0xd1,
		0xb2, 0x45,
		0x5f, 0xfd, 0xce, 0x74, 0xfa, 0xd2,
	}
	if v := data.String(); v != text {
		t.Errorf("%x: encoded to %s, expected %s", data, v, text)
	}
	if v := data.URN(); v != urn {
		t.Errorf("%x: urn is %s, expected %s", data, v, urn)
	}

	uuid, err := Parse(text)
	if err != nil {
		t.Errorf("Parse returned unexpected error %v", err)
	}
	if data != uuid {
		t.Errorf("%s: decoded to %s, expected %s", text, uuid, data)
	}
}

func TestVersion1(t *testing.T) {
	uuid1, err := NewUUID()
	if err != nil {
		t.Fatalf("could not create UUID: %v", err)
	}
	uuid2, err := NewUUID()
	if err != nil {
		t.Fatalf("could not create UUID: %v", err)
	}

	if uuid1 == uuid2 {
		t.Errorf("%s:duplicate uuid", uuid1)
	}
	if v := uuid1.Version(); v != 1 {
		t.Errorf("%s: version %s expected 1", uuid1, v)
	}
	if v := uuid2.Version(); v != 1 {
		t.Errorf("%s: version %s expected 1", uuid2, v)
	}
	n1 := uuid1.NodeID()
	n2 := uuid2.NodeID()
	if !bytes.Equal(n1, n2) {
		t.Errorf("Different nodes %x != %x", n1, n2)
	}
	t1 := uuid1.Time()
	t2 := uuid2.Time()
	q1 := uuid1.ClockSequence()
	q2 := uuid2.ClockSequence()

	switch {
	case t1 == t2 && q1 == q2:
		t.Error("time stopped")
	case t1 > t2 && q1 == q2:
		t.Error("time reversed")
	case t1 < t2 && q1 != q2:
		t.Error("clock sequence changed unexpectedly")
	}
}

func TestNode(t *testing.T) {
	// This test is mostly to make sure we don't leave nodeMu locked.
	ifname = ""
	if ni := NodeInterface(); ni != "" {
		t.Errorf("NodeInterface got %q, want %q", ni, "")
	}
	if SetNodeInterface("xyzzy") {
		t.Error("SetNodeInterface succeeded on a bad interface name")
	}
	if !SetNodeInterface("") {
		t.Error("SetNodeInterface failed")
	}
	if runtime.GOARCH != "js" {
		if ni := NodeInterface(); ni == "" {
			t.Error("NodeInterface returned an empty string")
		}
	}

	ni := NodeID()
	if len(ni) != 6 {
		t.Errorf("ni got %d bytes, want 6", len(ni))
	}
	hasData := false
	for _, b := range ni {
		if b != 0 {
			hasData = true
		}
	}
	if !hasData {
		t.Error("nodeid is all zeros")
	}

	id := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	SetNodeID(id)
	ni = NodeID()
	if !bytes.Equal(ni, id[:6]) {
		t.Errorf("got nodeid %v, want %v", ni, id[:6])
	}

	if ni := NodeInterface(); ni != "user" {
		t.Errorf("got interface %q, want %q", ni, "user")
	}
}

func TestNodeAndTime(t *testing.T) {
	// Time is February 5, 1998 12:30:23.136364800 AM GMT

	uuid, err := Parse("7d444840-9dc0-11d1-b245-5ffdce74fad2")
	if err != nil {
		t.Fatalf("Parser returned unexpected error %v", err)
	}
	node := []byte{0x5f, 0xfd, 0xce, 0x74, 0xfa, 0xd2}

	ts := uuid.Time()
	c := time.Unix(ts.UnixTime())
	want := time.Date(1998, 2, 5, 0, 30, 23, 136364800, time.UTC)
	if !c.Equal(want) {
		t.Errorf("Got time %v, want %v", c, want)
	}
	if !bytes.Equal(node, uuid.NodeID()) {
		t.Errorf("Expected node %v got %v", node, uuid.NodeID())
	}
}

func TestMD5(t *testing.T) {
	uuid := NewMD5(NameSpaceDNS, []byte("python.org")).String()
	want := "6fa459ea-ee8a-3ca4-894e-db77e160355e"
	if uuid != want {
		t.Errorf("MD5: got %q expected %q", uuid, want)
	}
}

func TestSHA1(t *testing.T) {
	uuid := NewSHA1(NameSpaceDNS, []byte("python.org")).String()
	want := "886313e1-3b8a-5372-9b90-0c9aee199e5d"
	if uuid != want {
		t.Errorf("SHA1: got %q expected %q", uuid, want)
	}
}

func TestNodeID(t *testing.T) {
	nid := []byte{1, 2, 3, 4, 5, 6}
	SetNodeInterface("")
	s := NodeInterface()
	if runtime.GOARCH != "js" {
		if s == "" || s == "user" {
			t.Errorf("NodeInterface %q after SetInterface", s)
		}
	}
	node1 := NodeID()
	if node1 == nil {
		t.Error("NodeID nil after SetNodeInterface", s)
	}
	SetNodeID(nid)
	s = NodeInterface()
	if s != "user" {
		t.Errorf("Expected NodeInterface %q got %q", "user", s)
	}
	node2 := NodeID()
	if node2 == nil {
		t.Error("NodeID nil after SetNodeID", s)
	}
	if bytes.Equal(node1, node2) {
		t.Error("NodeID not changed after SetNodeID", s)
	} else if !bytes.Equal(nid, node2) {
		t.Errorf("NodeID is %x, expected %x", node2, nid)
	}
}

func testDCE(t *testing.T, name string, uuid UUID, err error, domain Domain, id uint32) {
	if err != nil {
		t.Errorf("%s failed: %v", name, err)
		return
	}
	if v := uuid.Version(); v != 2 {
		t.Errorf("%s: %s: expected version 2, got %s", name, uuid, v)
		return
	}
	if v := uuid.Domain(); v != domain {
		t.Errorf("%s: %s: expected domain %d, got %d", name, uuid, domain, v)
	}
	if v := uuid.ID(); v != id {
		t.Errorf("%s: %s: expected id %d, got %d", name, uuid, id, v)
	}
}

func TestDCE(t *testing.T) {
	uuid, err := NewDCESecurity(42, 12345678)
	testDCE(t, "NewDCESecurity", uuid, err, 42, 12345678)
	uuid, err = NewDCEPerson()
	testDCE(t, "NewDCEPerson", uuid, err, Person, uint32(os.Getuid()))
	uuid, err = NewDCEGroup()
	testDCE(t, "NewDCEGroup", uuid, err, Group, uint32(os.Getgid()))
}

type badRand struct{}

func (r badRand) Read(buf []byte) (int, error) {
	for i := range buf {
		buf[i] = byte(i)
	}
	return len(buf), nil
}

func TestBadRand(t *testing.T) {
	SetRand(badRand{})
	uuid1 := New()
	uuid2 := New()
	if uuid1 != uuid2 {
		t.Errorf("expected duplicates, got %q and %q", uuid1, uuid2)
	}
	SetRand(nil)
	uuid1 = New()
	uuid2 = New()
	if uuid1 == uuid2 {
		t.Errorf("unexpected duplicates, got %q", uuid1)
	}
}

func TestSetRand(t *testing.T) {
	myString := "805-9dd6-1a877cb526c678e71d38-7122-44c0-9b7c-04e7001cc78783ac3e82-47a3-4cc3-9951-13f3339d88088f5d685a-11f7-4078-ada9-de44ad2daeb7"

	SetRand(strings.NewReader(myString))
	uuid1 := New()
	uuid2 := New()

	SetRand(strings.NewReader(myString))
	uuid3 := New()
	uuid4 := New()

	if uuid1 != uuid3 {
		t.Errorf("expected duplicates, got %q and %q", uuid1, uuid3)
	}
	if uuid2 != uuid4 {
		t.Errorf("expected duplicates, got %q and %q", uuid2, uuid4)
	}
}

func TestRandomFromReader(t *testing.T) {
	myString := "8059ddhdle77cb52"
	r := bytes.NewReader([]byte(myString))
	r2 := bytes.NewReader([]byte(myString))
	uuid1, err := NewRandomFromReader(r)
	if err != nil {
		t.Errorf("failed generating UUID from a reader")
	}
	_, err = NewRandomFromReader(r)
	if err == nil {
		t.Errorf("expecting an error as reader has no more bytes. Got uuid. NewRandomFromReader may not be using the provided reader")
	}
	uuid3, err := NewRandomFromReader(r2)
	if err != nil {
		t.Errorf("failed generating UUID from a reader")
	}
	if uuid1 != uuid3 {
		t.Errorf("expected duplicates, got %q and %q", uuid1, uuid3)
	}
}

var asString = "f47ac10b-58cc-0372-8567-0e02b2c3d479"
var asBytes = []byte(asString)

func BenchmarkParse(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := Parse(asString)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkParseBytes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := ParseBytes(asBytes)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// parseBytesUnsafe is to benchmark using unsafe.
func parseBytesUnsafe(b []byte) (UUID, error) {
	return Parse(*(*string)(unsafe.Pointer(&b)))
}

func BenchmarkParseBytesUnsafe(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := parseBytesUnsafe(asBytes)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// parseBytesCopy is to benchmark not using unsafe.
func parseBytesCopy(b []byte) (UUID, error) {
	return Parse(string(b))
}

func BenchmarkParseBytesCopy(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := parseBytesCopy(asBytes)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNew(b *testing.B) {
	for i := 0; i < b.N; i++ {
		New()
	}
}

func BenchmarkUUID_String(b *testing.B) {
	uuid, err := Parse("f47ac10b-58cc-0372-8567-0e02b2c3d479")
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < b.N; i++ {
		if uuid.String() == "" {
			b.Fatal("invalid uuid")
		}
	}
}

func BenchmarkUUID_URN(b *testing.B) {
	uuid, err := Parse("f47ac10b-58cc-0372-8567-0e02b2c3d479")
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < b.N; i++ {
		if uuid.URN() == "" {
			b.Fatal("invalid uuid")
		}
	}
}
