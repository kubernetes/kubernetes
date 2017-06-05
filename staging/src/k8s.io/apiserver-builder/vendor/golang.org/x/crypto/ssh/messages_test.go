// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"math/big"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
)

var intLengthTests = []struct {
	val, length int
}{
	{0, 4 + 0},
	{1, 4 + 1},
	{127, 4 + 1},
	{128, 4 + 2},
	{-1, 4 + 1},
}

func TestIntLength(t *testing.T) {
	for _, test := range intLengthTests {
		v := new(big.Int).SetInt64(int64(test.val))
		length := intLength(v)
		if length != test.length {
			t.Errorf("For %d, got length %d but expected %d", test.val, length, test.length)
		}
	}
}

type msgAllTypes struct {
	Bool    bool `sshtype:"21"`
	Array   [16]byte
	Uint64  uint64
	Uint32  uint32
	Uint8   uint8
	String  string
	Strings []string
	Bytes   []byte
	Int     *big.Int
	Rest    []byte `ssh:"rest"`
}

func (t *msgAllTypes) Generate(rand *rand.Rand, size int) reflect.Value {
	m := &msgAllTypes{}
	m.Bool = rand.Intn(2) == 1
	randomBytes(m.Array[:], rand)
	m.Uint64 = uint64(rand.Int63n(1<<63 - 1))
	m.Uint32 = uint32(rand.Intn((1 << 31) - 1))
	m.Uint8 = uint8(rand.Intn(1 << 8))
	m.String = string(m.Array[:])
	m.Strings = randomNameList(rand)
	m.Bytes = m.Array[:]
	m.Int = randomInt(rand)
	m.Rest = m.Array[:]
	return reflect.ValueOf(m)
}

func TestMarshalUnmarshal(t *testing.T) {
	rand := rand.New(rand.NewSource(0))
	iface := &msgAllTypes{}
	ty := reflect.ValueOf(iface).Type()

	n := 100
	if testing.Short() {
		n = 5
	}
	for j := 0; j < n; j++ {
		v, ok := quick.Value(ty, rand)
		if !ok {
			t.Errorf("failed to create value")
			break
		}

		m1 := v.Elem().Interface()
		m2 := iface

		marshaled := Marshal(m1)
		if err := Unmarshal(marshaled, m2); err != nil {
			t.Errorf("Unmarshal %#v: %s", m1, err)
			break
		}

		if !reflect.DeepEqual(v.Interface(), m2) {
			t.Errorf("got: %#v\nwant:%#v\n%x", m2, m1, marshaled)
			break
		}
	}
}

func TestUnmarshalEmptyPacket(t *testing.T) {
	var b []byte
	var m channelRequestSuccessMsg
	if err := Unmarshal(b, &m); err == nil {
		t.Fatalf("unmarshal of empty slice succeeded")
	}
}

func TestUnmarshalUnexpectedPacket(t *testing.T) {
	type S struct {
		I uint32 `sshtype:"43"`
		S string
		B bool
	}

	s := S{11, "hello", true}
	packet := Marshal(s)
	packet[0] = 42
	roundtrip := S{}
	err := Unmarshal(packet, &roundtrip)
	if err == nil {
		t.Fatal("expected error, not nil")
	}
}

func TestMarshalPtr(t *testing.T) {
	s := struct {
		S string
	}{"hello"}

	m1 := Marshal(s)
	m2 := Marshal(&s)
	if !bytes.Equal(m1, m2) {
		t.Errorf("got %q, want %q for marshaled pointer", m2, m1)
	}
}

func TestBareMarshalUnmarshal(t *testing.T) {
	type S struct {
		I uint32
		S string
		B bool
	}

	s := S{42, "hello", true}
	packet := Marshal(s)
	roundtrip := S{}
	Unmarshal(packet, &roundtrip)

	if !reflect.DeepEqual(s, roundtrip) {
		t.Errorf("got %#v, want %#v", roundtrip, s)
	}
}

func TestBareMarshal(t *testing.T) {
	type S2 struct {
		I uint32
	}
	s := S2{42}
	packet := Marshal(s)
	i, rest, ok := parseUint32(packet)
	if len(rest) > 0 || !ok {
		t.Errorf("parseInt(%q): parse error", packet)
	}
	if i != s.I {
		t.Errorf("got %d, want %d", i, s.I)
	}
}

func TestUnmarshalShortKexInitPacket(t *testing.T) {
	// This used to panic.
	// Issue 11348
	packet := []byte{0x14, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0xff, 0xff, 0xff, 0xff}
	kim := &kexInitMsg{}
	if err := Unmarshal(packet, kim); err == nil {
		t.Error("truncated packet unmarshaled without error")
	}
}

func TestMarshalMultiTag(t *testing.T) {
	var res struct {
		A uint32 `sshtype:"1|2"`
	}

	good1 := struct {
		A uint32 `sshtype:"1"`
	}{
		1,
	}
	good2 := struct {
		A uint32 `sshtype:"2"`
	}{
		1,
	}

	if e := Unmarshal(Marshal(good1), &res); e != nil {
		t.Errorf("error unmarshaling multipart tag: %v", e)
	}

	if e := Unmarshal(Marshal(good2), &res); e != nil {
		t.Errorf("error unmarshaling multipart tag: %v", e)
	}

	bad1 := struct {
		A uint32 `sshtype:"3"`
	}{
		1,
	}
	if e := Unmarshal(Marshal(bad1), &res); e == nil {
		t.Errorf("bad struct unmarshaled without error")
	}
}

func randomBytes(out []byte, rand *rand.Rand) {
	for i := 0; i < len(out); i++ {
		out[i] = byte(rand.Int31())
	}
}

func randomNameList(rand *rand.Rand) []string {
	ret := make([]string, rand.Int31()&15)
	for i := range ret {
		s := make([]byte, 1+(rand.Int31()&15))
		for j := range s {
			s[j] = 'a' + uint8(rand.Int31()&15)
		}
		ret[i] = string(s)
	}
	return ret
}

func randomInt(rand *rand.Rand) *big.Int {
	return new(big.Int).SetInt64(int64(int32(rand.Uint32())))
}

func (*kexInitMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	ki := &kexInitMsg{}
	randomBytes(ki.Cookie[:], rand)
	ki.KexAlgos = randomNameList(rand)
	ki.ServerHostKeyAlgos = randomNameList(rand)
	ki.CiphersClientServer = randomNameList(rand)
	ki.CiphersServerClient = randomNameList(rand)
	ki.MACsClientServer = randomNameList(rand)
	ki.MACsServerClient = randomNameList(rand)
	ki.CompressionClientServer = randomNameList(rand)
	ki.CompressionServerClient = randomNameList(rand)
	ki.LanguagesClientServer = randomNameList(rand)
	ki.LanguagesServerClient = randomNameList(rand)
	if rand.Int31()&1 == 1 {
		ki.FirstKexFollows = true
	}
	return reflect.ValueOf(ki)
}

func (*kexDHInitMsg) Generate(rand *rand.Rand, size int) reflect.Value {
	dhi := &kexDHInitMsg{}
	dhi.X = randomInt(rand)
	return reflect.ValueOf(dhi)
}

var (
	_kexInitMsg   = new(kexInitMsg).Generate(rand.New(rand.NewSource(0)), 10).Elem().Interface()
	_kexDHInitMsg = new(kexDHInitMsg).Generate(rand.New(rand.NewSource(0)), 10).Elem().Interface()

	_kexInit   = Marshal(_kexInitMsg)
	_kexDHInit = Marshal(_kexDHInitMsg)
)

func BenchmarkMarshalKexInitMsg(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Marshal(_kexInitMsg)
	}
}

func BenchmarkUnmarshalKexInitMsg(b *testing.B) {
	m := new(kexInitMsg)
	for i := 0; i < b.N; i++ {
		Unmarshal(_kexInit, m)
	}
}

func BenchmarkMarshalKexDHInitMsg(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Marshal(_kexDHInitMsg)
	}
}

func BenchmarkUnmarshalKexDHInitMsg(b *testing.B) {
	m := new(kexDHInitMsg)
	for i := 0; i < b.N; i++ {
		Unmarshal(_kexDHInit, m)
	}
}
