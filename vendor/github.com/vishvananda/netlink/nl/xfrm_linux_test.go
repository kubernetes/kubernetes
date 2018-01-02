package nl

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"testing"
)

func (msg *XfrmAddress) write(b []byte) {
	copy(b[0:SizeofXfrmAddress], msg[:])
}

func (msg *XfrmAddress) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmAddress)
	msg.write(b)
	return b
}

func deserializeXfrmAddressSafe(b []byte) *XfrmAddress {
	var msg = XfrmAddress{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmAddress]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmAddressDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmAddress)
	rand.Read(orig)
	safemsg := deserializeXfrmAddressSafe(orig)
	msg := DeserializeXfrmAddress(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmSelector) write(b []byte) {
	const AddrEnd = SizeofXfrmAddress * 2
	native := NativeEndian()
	msg.Daddr.write(b[0:SizeofXfrmAddress])
	msg.Saddr.write(b[SizeofXfrmAddress:AddrEnd])
	native.PutUint16(b[AddrEnd:AddrEnd+2], msg.Dport)
	native.PutUint16(b[AddrEnd+2:AddrEnd+4], msg.DportMask)
	native.PutUint16(b[AddrEnd+4:AddrEnd+6], msg.Sport)
	native.PutUint16(b[AddrEnd+6:AddrEnd+8], msg.SportMask)
	native.PutUint16(b[AddrEnd+8:AddrEnd+10], msg.Family)
	b[AddrEnd+10] = msg.PrefixlenD
	b[AddrEnd+11] = msg.PrefixlenS
	b[AddrEnd+12] = msg.Proto
	copy(b[AddrEnd+13:AddrEnd+16], msg.Pad[:])
	native.PutUint32(b[AddrEnd+16:AddrEnd+20], uint32(msg.Ifindex))
	native.PutUint32(b[AddrEnd+20:AddrEnd+24], msg.User)
}

func (msg *XfrmSelector) serializeSafe() []byte {
	length := SizeofXfrmSelector
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeXfrmSelectorSafe(b []byte) *XfrmSelector {
	var msg = XfrmSelector{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmSelector]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmSelectorDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmSelector)
	rand.Read(orig)
	safemsg := deserializeXfrmSelectorSafe(orig)
	msg := DeserializeXfrmSelector(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmLifetimeCfg) write(b []byte) {
	native := NativeEndian()
	native.PutUint64(b[0:8], msg.SoftByteLimit)
	native.PutUint64(b[8:16], msg.HardByteLimit)
	native.PutUint64(b[16:24], msg.SoftPacketLimit)
	native.PutUint64(b[24:32], msg.HardPacketLimit)
	native.PutUint64(b[32:40], msg.SoftAddExpiresSeconds)
	native.PutUint64(b[40:48], msg.HardAddExpiresSeconds)
	native.PutUint64(b[48:56], msg.SoftUseExpiresSeconds)
	native.PutUint64(b[56:64], msg.HardUseExpiresSeconds)
}

func (msg *XfrmLifetimeCfg) serializeSafe() []byte {
	length := SizeofXfrmLifetimeCfg
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeXfrmLifetimeCfgSafe(b []byte) *XfrmLifetimeCfg {
	var msg = XfrmLifetimeCfg{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmLifetimeCfg]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmLifetimeCfgDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmLifetimeCfg)
	rand.Read(orig)
	safemsg := deserializeXfrmLifetimeCfgSafe(orig)
	msg := DeserializeXfrmLifetimeCfg(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmLifetimeCur) write(b []byte) {
	native := NativeEndian()
	native.PutUint64(b[0:8], msg.Bytes)
	native.PutUint64(b[8:16], msg.Packets)
	native.PutUint64(b[16:24], msg.AddTime)
	native.PutUint64(b[24:32], msg.UseTime)
}

func (msg *XfrmLifetimeCur) serializeSafe() []byte {
	length := SizeofXfrmLifetimeCur
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeXfrmLifetimeCurSafe(b []byte) *XfrmLifetimeCur {
	var msg = XfrmLifetimeCur{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmLifetimeCur]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmLifetimeCurDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmLifetimeCur)
	rand.Read(orig)
	safemsg := deserializeXfrmLifetimeCurSafe(orig)
	msg := DeserializeXfrmLifetimeCur(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmId) write(b []byte) {
	native := NativeEndian()
	msg.Daddr.write(b[0:SizeofXfrmAddress])
	native.PutUint32(b[SizeofXfrmAddress:SizeofXfrmAddress+4], msg.Spi)
	b[SizeofXfrmAddress+4] = msg.Proto
	copy(b[SizeofXfrmAddress+5:SizeofXfrmAddress+8], msg.Pad[:])
}

func (msg *XfrmId) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmId)
	msg.write(b)
	return b
}

func deserializeXfrmIdSafe(b []byte) *XfrmId {
	var msg = XfrmId{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmId]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmIdDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmId)
	rand.Read(orig)
	safemsg := deserializeXfrmIdSafe(orig)
	msg := DeserializeXfrmId(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}
