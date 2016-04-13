package nl

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"testing"
)

func (msg *XfrmUserpolicyId) write(b []byte) {
	native := NativeEndian()
	msg.Sel.write(b[0:SizeofXfrmSelector])
	native.PutUint32(b[SizeofXfrmSelector:SizeofXfrmSelector+4], msg.Index)
	b[SizeofXfrmSelector+4] = msg.Dir
	copy(b[SizeofXfrmSelector+5:SizeofXfrmSelector+8], msg.Pad[:])
}

func (msg *XfrmUserpolicyId) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmUserpolicyId)
	msg.write(b)
	return b
}

func deserializeXfrmUserpolicyIdSafe(b []byte) *XfrmUserpolicyId {
	var msg = XfrmUserpolicyId{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmUserpolicyId]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmUserpolicyIdDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmUserpolicyId)
	rand.Read(orig)
	safemsg := deserializeXfrmUserpolicyIdSafe(orig)
	msg := DeserializeXfrmUserpolicyId(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmUserpolicyInfo) write(b []byte) {
	const CfgEnd = SizeofXfrmSelector + SizeofXfrmLifetimeCfg
	const CurEnd = CfgEnd + SizeofXfrmLifetimeCur
	native := NativeEndian()
	msg.Sel.write(b[0:SizeofXfrmSelector])
	msg.Lft.write(b[SizeofXfrmSelector:CfgEnd])
	msg.Curlft.write(b[CfgEnd:CurEnd])
	native.PutUint32(b[CurEnd:CurEnd+4], msg.Priority)
	native.PutUint32(b[CurEnd+4:CurEnd+8], msg.Index)
	b[CurEnd+8] = msg.Dir
	b[CurEnd+9] = msg.Action
	b[CurEnd+10] = msg.Flags
	b[CurEnd+11] = msg.Share
	copy(b[CurEnd+12:CurEnd+16], msg.Pad[:])
}

func (msg *XfrmUserpolicyInfo) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmUserpolicyInfo)
	msg.write(b)
	return b
}

func deserializeXfrmUserpolicyInfoSafe(b []byte) *XfrmUserpolicyInfo {
	var msg = XfrmUserpolicyInfo{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmUserpolicyInfo]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmUserpolicyInfoDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmUserpolicyInfo)
	rand.Read(orig)
	safemsg := deserializeXfrmUserpolicyInfoSafe(orig)
	msg := DeserializeXfrmUserpolicyInfo(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmUserTmpl) write(b []byte) {
	const AddrEnd = SizeofXfrmId + 4 + SizeofXfrmAddress
	native := NativeEndian()
	msg.XfrmId.write(b[0:SizeofXfrmId])
	native.PutUint16(b[SizeofXfrmId:SizeofXfrmId+2], msg.Family)
	copy(b[SizeofXfrmId+2:SizeofXfrmId+4], msg.Pad1[:])
	msg.Saddr.write(b[SizeofXfrmId+4 : AddrEnd])
	native.PutUint32(b[AddrEnd:AddrEnd+4], msg.Reqid)
	b[AddrEnd+4] = msg.Mode
	b[AddrEnd+5] = msg.Share
	b[AddrEnd+6] = msg.Optional
	b[AddrEnd+7] = msg.Pad2
	native.PutUint32(b[AddrEnd+8:AddrEnd+12], msg.Aalgos)
	native.PutUint32(b[AddrEnd+12:AddrEnd+16], msg.Ealgos)
	native.PutUint32(b[AddrEnd+16:AddrEnd+20], msg.Calgos)
}

func (msg *XfrmUserTmpl) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmUserTmpl)
	msg.write(b)
	return b
}

func deserializeXfrmUserTmplSafe(b []byte) *XfrmUserTmpl {
	var msg = XfrmUserTmpl{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmUserTmpl]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmUserTmplDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmUserTmpl)
	rand.Read(orig)
	safemsg := deserializeXfrmUserTmplSafe(orig)
	msg := DeserializeXfrmUserTmpl(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}
