package nl

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"reflect"
	"syscall"
	"testing"
)

type testSerializer interface {
	serializeSafe() []byte
	Serialize() []byte
}

func testDeserializeSerialize(t *testing.T, orig []byte, safemsg testSerializer, msg testSerializer) {
	if !reflect.DeepEqual(safemsg, msg) {
		t.Fatal("Deserialization failed.\n", safemsg, "\n", msg)
	}
	safe := msg.serializeSafe()
	if !bytes.Equal(safe, orig) {
		t.Fatal("Safe serialization failed.\n", safe, "\n", orig)
	}
	b := msg.Serialize()
	if !bytes.Equal(b, safe) {
		t.Fatal("Serialization failed.\n", b, "\n", safe)
	}
}

func (msg *IfInfomsg) write(b []byte) {
	native := NativeEndian()
	b[0] = msg.Family
	b[1] = msg.X__ifi_pad
	native.PutUint16(b[2:4], msg.Type)
	native.PutUint32(b[4:8], uint32(msg.Index))
	native.PutUint32(b[8:12], msg.Flags)
	native.PutUint32(b[12:16], msg.Change)
}

func (msg *IfInfomsg) serializeSafe() []byte {
	length := syscall.SizeofIfInfomsg
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeIfInfomsgSafe(b []byte) *IfInfomsg {
	var msg = IfInfomsg{}
	binary.Read(bytes.NewReader(b[0:syscall.SizeofIfInfomsg]), NativeEndian(), &msg)
	return &msg
}

func TestIfInfomsgDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, syscall.SizeofIfInfomsg)
	rand.Read(orig)
	safemsg := deserializeIfInfomsgSafe(orig)
	msg := DeserializeIfInfomsg(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}
