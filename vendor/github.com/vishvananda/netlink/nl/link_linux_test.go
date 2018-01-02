package nl

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"testing"
)

func (msg *VfMac) write(b []byte) {
	native := NativeEndian()
	native.PutUint32(b[0:4], uint32(msg.Vf))
	copy(b[4:36], msg.Mac[:])
}

func (msg *VfMac) serializeSafe() []byte {
	length := SizeofVfMac
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeVfMacSafe(b []byte) *VfMac {
	var msg = VfMac{}
	binary.Read(bytes.NewReader(b[0:SizeofVfMac]), NativeEndian(), &msg)
	return &msg
}

func TestVfMacDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofVfMac)
	rand.Read(orig)
	safemsg := deserializeVfMacSafe(orig)
	msg := DeserializeVfMac(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *VfVlan) write(b []byte) {
	native := NativeEndian()
	native.PutUint32(b[0:4], uint32(msg.Vf))
	native.PutUint32(b[4:8], uint32(msg.Vlan))
	native.PutUint32(b[8:12], uint32(msg.Qos))
}

func (msg *VfVlan) serializeSafe() []byte {
	length := SizeofVfVlan
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeVfVlanSafe(b []byte) *VfVlan {
	var msg = VfVlan{}
	binary.Read(bytes.NewReader(b[0:SizeofVfVlan]), NativeEndian(), &msg)
	return &msg
}

func TestVfVlanDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofVfVlan)
	rand.Read(orig)
	safemsg := deserializeVfVlanSafe(orig)
	msg := DeserializeVfVlan(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *VfTxRate) write(b []byte) {
	native := NativeEndian()
	native.PutUint32(b[0:4], uint32(msg.Vf))
	native.PutUint32(b[4:8], uint32(msg.Rate))
}

func (msg *VfTxRate) serializeSafe() []byte {
	length := SizeofVfTxRate
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeVfTxRateSafe(b []byte) *VfTxRate {
	var msg = VfTxRate{}
	binary.Read(bytes.NewReader(b[0:SizeofVfTxRate]), NativeEndian(), &msg)
	return &msg
}

func TestVfTxRateDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofVfTxRate)
	rand.Read(orig)
	safemsg := deserializeVfTxRateSafe(orig)
	msg := DeserializeVfTxRate(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *VfRate) write(b []byte) {
	native := NativeEndian()
	native.PutUint32(b[0:4], uint32(msg.Vf))
	native.PutUint32(b[4:8], uint32(msg.MinTxRate))
	native.PutUint32(b[8:12], uint32(msg.MaxTxRate))
}

func (msg *VfRate) serializeSafe() []byte {
	length := SizeofVfRate
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeVfRateSafe(b []byte) *VfRate {
	var msg = VfRate{}
	binary.Read(bytes.NewReader(b[0:SizeofVfRate]), NativeEndian(), &msg)
	return &msg
}

func TestVfRateDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofVfRate)
	rand.Read(orig)
	safemsg := deserializeVfRateSafe(orig)
	msg := DeserializeVfRate(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *VfSpoofchk) write(b []byte) {
	native := NativeEndian()
	native.PutUint32(b[0:4], uint32(msg.Vf))
	native.PutUint32(b[4:8], uint32(msg.Setting))
}

func (msg *VfSpoofchk) serializeSafe() []byte {
	length := SizeofVfSpoofchk
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeVfSpoofchkSafe(b []byte) *VfSpoofchk {
	var msg = VfSpoofchk{}
	binary.Read(bytes.NewReader(b[0:SizeofVfSpoofchk]), NativeEndian(), &msg)
	return &msg
}

func TestVfSpoofchkDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofVfSpoofchk)
	rand.Read(orig)
	safemsg := deserializeVfSpoofchkSafe(orig)
	msg := DeserializeVfSpoofchk(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *VfLinkState) write(b []byte) {
	native := NativeEndian()
	native.PutUint32(b[0:4], uint32(msg.Vf))
	native.PutUint32(b[4:8], uint32(msg.LinkState))
}

func (msg *VfLinkState) serializeSafe() []byte {
	length := SizeofVfLinkState
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeVfLinkStateSafe(b []byte) *VfLinkState {
	var msg = VfLinkState{}
	binary.Read(bytes.NewReader(b[0:SizeofVfLinkState]), NativeEndian(), &msg)
	return &msg
}

func TestVfLinkStateDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofVfLinkState)
	rand.Read(orig)
	safemsg := deserializeVfLinkStateSafe(orig)
	msg := DeserializeVfLinkState(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *VfRssQueryEn) write(b []byte) {
	native := NativeEndian()
	native.PutUint32(b[0:4], uint32(msg.Vf))
	native.PutUint32(b[4:8], uint32(msg.Setting))
}

func (msg *VfRssQueryEn) serializeSafe() []byte {
	length := SizeofVfRssQueryEn
	b := make([]byte, length)
	msg.write(b)
	return b
}

func deserializeVfRssQueryEnSafe(b []byte) *VfRssQueryEn {
	var msg = VfRssQueryEn{}
	binary.Read(bytes.NewReader(b[0:SizeofVfRssQueryEn]), NativeEndian(), &msg)
	return &msg
}

func TestVfRssQueryEnDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofVfRssQueryEn)
	rand.Read(orig)
	safemsg := deserializeVfRssQueryEnSafe(orig)
	msg := DeserializeVfRssQueryEn(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}
