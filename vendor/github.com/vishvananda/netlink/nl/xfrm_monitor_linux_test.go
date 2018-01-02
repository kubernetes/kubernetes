package nl

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"testing"
)

func (msg *XfrmUserExpire) write(b []byte) {
	msg.XfrmUsersaInfo.write(b[0:SizeofXfrmUsersaInfo])
	b[SizeofXfrmUsersaInfo] = msg.Hard
	copy(b[SizeofXfrmUsersaInfo+1:SizeofXfrmUserExpire], msg.Pad[:])
}

func (msg *XfrmUserExpire) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmUserExpire)
	msg.write(b)
	return b
}

func deserializeXfrmUserExpireSafe(b []byte) *XfrmUserExpire {
	var msg = XfrmUserExpire{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmUserExpire]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmUserExpireDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmUserExpire)
	rand.Read(orig)
	safemsg := deserializeXfrmUserExpireSafe(orig)
	msg := DeserializeXfrmUserExpire(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}
