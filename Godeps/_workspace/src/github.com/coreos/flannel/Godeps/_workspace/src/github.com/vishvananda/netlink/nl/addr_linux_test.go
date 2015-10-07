package nl

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"syscall"
	"testing"
)

func (msg *IfAddrmsg) write(b []byte) {
	native := NativeEndian()
	b[0] = msg.Family
	b[1] = msg.Prefixlen
	b[2] = msg.Flags
	b[3] = msg.Scope
	native.PutUint32(b[4:8], msg.Index)
}

func (msg *IfAddrmsg) serializeSafe() []byte {
	len := syscall.SizeofIfAddrmsg
	b := make([]byte, len)
	msg.write(b)
	return b
}

func deserializeIfAddrmsgSafe(b []byte) *IfAddrmsg {
	var msg = IfAddrmsg{}
	binary.Read(bytes.NewReader(b[0:syscall.SizeofIfAddrmsg]), NativeEndian(), &msg)
	return &msg
}

func TestIfAddrmsgDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, syscall.SizeofIfAddrmsg)
	rand.Read(orig)
	safemsg := deserializeIfAddrmsgSafe(orig)
	msg := DeserializeIfAddrmsg(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}
