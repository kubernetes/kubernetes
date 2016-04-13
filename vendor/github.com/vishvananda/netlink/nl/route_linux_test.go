package nl

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"syscall"
	"testing"
)

func (msg *RtMsg) write(b []byte) {
	native := NativeEndian()
	b[0] = msg.Family
	b[1] = msg.Dst_len
	b[2] = msg.Src_len
	b[3] = msg.Tos
	b[4] = msg.Table
	b[5] = msg.Protocol
	b[6] = msg.Scope
	b[7] = msg.Type
	native.PutUint32(b[8:12], msg.Flags)
}

func (msg *RtMsg) serializeSafe() []byte {
	len := syscall.SizeofRtMsg
	b := make([]byte, len)
	msg.write(b)
	return b
}

func deserializeRtMsgSafe(b []byte) *RtMsg {
	var msg = RtMsg{}
	binary.Read(bytes.NewReader(b[0:syscall.SizeofRtMsg]), NativeEndian(), &msg)
	return &msg
}

func TestRtMsgDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, syscall.SizeofRtMsg)
	rand.Read(orig)
	safemsg := deserializeRtMsgSafe(orig)
	msg := DeserializeRtMsg(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}
