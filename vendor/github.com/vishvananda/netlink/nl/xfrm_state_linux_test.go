package nl

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"testing"
)

func (msg *XfrmUsersaId) write(b []byte) {
	native := NativeEndian()
	msg.Daddr.write(b[0:SizeofXfrmAddress])
	native.PutUint32(b[SizeofXfrmAddress:SizeofXfrmAddress+4], msg.Spi)
	native.PutUint16(b[SizeofXfrmAddress+4:SizeofXfrmAddress+6], msg.Family)
	b[SizeofXfrmAddress+6] = msg.Proto
	b[SizeofXfrmAddress+7] = msg.Pad
}

func (msg *XfrmUsersaId) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmUsersaId)
	msg.write(b)
	return b
}

func deserializeXfrmUsersaIdSafe(b []byte) *XfrmUsersaId {
	var msg = XfrmUsersaId{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmUsersaId]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmUsersaIdDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmUsersaId)
	rand.Read(orig)
	safemsg := deserializeXfrmUsersaIdSafe(orig)
	msg := DeserializeXfrmUsersaId(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmStats) write(b []byte) {
	native := NativeEndian()
	native.PutUint32(b[0:4], msg.ReplayWindow)
	native.PutUint32(b[4:8], msg.Replay)
	native.PutUint32(b[8:12], msg.IntegrityFailed)
}

func (msg *XfrmStats) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmStats)
	msg.write(b)
	return b
}

func deserializeXfrmStatsSafe(b []byte) *XfrmStats {
	var msg = XfrmStats{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmStats]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmStatsDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmStats)
	rand.Read(orig)
	safemsg := deserializeXfrmStatsSafe(orig)
	msg := DeserializeXfrmStats(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmUsersaInfo) write(b []byte) {
	const IdEnd = SizeofXfrmSelector + SizeofXfrmId
	const AddressEnd = IdEnd + SizeofXfrmAddress
	const CfgEnd = AddressEnd + SizeofXfrmLifetimeCfg
	const CurEnd = CfgEnd + SizeofXfrmLifetimeCur
	const StatsEnd = CurEnd + SizeofXfrmStats
	native := NativeEndian()
	msg.Sel.write(b[0:SizeofXfrmSelector])
	msg.Id.write(b[SizeofXfrmSelector:IdEnd])
	msg.Saddr.write(b[IdEnd:AddressEnd])
	msg.Lft.write(b[AddressEnd:CfgEnd])
	msg.Curlft.write(b[CfgEnd:CurEnd])
	msg.Stats.write(b[CurEnd:StatsEnd])
	native.PutUint32(b[StatsEnd:StatsEnd+4], msg.Seq)
	native.PutUint32(b[StatsEnd+4:StatsEnd+8], msg.Reqid)
	native.PutUint16(b[StatsEnd+8:StatsEnd+10], msg.Family)
	b[StatsEnd+10] = msg.Mode
	b[StatsEnd+11] = msg.ReplayWindow
	b[StatsEnd+12] = msg.Flags
	copy(b[StatsEnd+13:StatsEnd+20], msg.Pad[:])
}

func (msg *XfrmUsersaInfo) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmUsersaInfo)
	msg.write(b)
	return b
}

func deserializeXfrmUsersaInfoSafe(b []byte) *XfrmUsersaInfo {
	var msg = XfrmUsersaInfo{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmUsersaInfo]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmUsersaInfoDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmUsersaInfo)
	rand.Read(orig)
	safemsg := deserializeXfrmUsersaInfoSafe(orig)
	msg := DeserializeXfrmUsersaInfo(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmAlgo) write(b []byte) {
	native := NativeEndian()
	copy(b[0:64], msg.AlgName[:])
	native.PutUint32(b[64:68], msg.AlgKeyLen)
	copy(b[68:msg.Len()], msg.AlgKey[:])
}

func (msg *XfrmAlgo) serializeSafe() []byte {
	b := make([]byte, msg.Len())
	msg.write(b)
	return b
}

func (msg *XfrmUserSpiInfo) write(b []byte) {
	native := NativeEndian()
	msg.XfrmUsersaInfo.write(b[0:SizeofXfrmUsersaInfo])
	native.PutUint32(b[SizeofXfrmUsersaInfo:SizeofXfrmUsersaInfo+4], msg.Min)
	native.PutUint32(b[SizeofXfrmUsersaInfo+4:SizeofXfrmUsersaInfo+8], msg.Max)
}

func (msg *XfrmUserSpiInfo) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmUserSpiInfo)
	msg.write(b)
	return b
}

func deserializeXfrmUserSpiInfoSafe(b []byte) *XfrmUserSpiInfo {
	var msg = XfrmUserSpiInfo{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmUserSpiInfo]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmUserSpiInfoDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmUserSpiInfo)
	rand.Read(orig)
	safemsg := deserializeXfrmUserSpiInfoSafe(orig)
	msg := DeserializeXfrmUserSpiInfo(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func deserializeXfrmAlgoSafe(b []byte) *XfrmAlgo {
	var msg = XfrmAlgo{}
	copy(msg.AlgName[:], b[0:64])
	binary.Read(bytes.NewReader(b[64:68]), NativeEndian(), &msg.AlgKeyLen)
	msg.AlgKey = b[68:msg.Len()]
	return &msg
}

func TestXfrmAlgoDeserializeSerialize(t *testing.T) {
	native := NativeEndian()
	// use a 32 byte key len
	var orig = make([]byte, SizeofXfrmAlgo+32)
	rand.Read(orig)
	// set the key len to 256 bits
	var KeyLen uint32 = 0x00000100
	// Little Endian    Big Endian
	// orig[64] = 0     orig[64] = 0
	// orig[65] = 1     orig[65] = 0
	// orig[66] = 0     orig[66] = 1
	// orig[67] = 0     orig[67] = 0
	native.PutUint32(orig[64:68], KeyLen)
	safemsg := deserializeXfrmAlgoSafe(orig)
	msg := DeserializeXfrmAlgo(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmAlgoAuth) write(b []byte) {
	native := NativeEndian()
	copy(b[0:64], msg.AlgName[:])
	native.PutUint32(b[64:68], msg.AlgKeyLen)
	native.PutUint32(b[68:72], msg.AlgTruncLen)
	copy(b[72:msg.Len()], msg.AlgKey[:])
}

func (msg *XfrmAlgoAuth) serializeSafe() []byte {
	b := make([]byte, msg.Len())
	msg.write(b)
	return b
}

func deserializeXfrmAlgoAuthSafe(b []byte) *XfrmAlgoAuth {
	var msg = XfrmAlgoAuth{}
	copy(msg.AlgName[:], b[0:64])
	binary.Read(bytes.NewReader(b[64:68]), NativeEndian(), &msg.AlgKeyLen)
	binary.Read(bytes.NewReader(b[68:72]), NativeEndian(), &msg.AlgTruncLen)
	msg.AlgKey = b[72:msg.Len()]
	return &msg
}

func TestXfrmAlgoAuthDeserializeSerialize(t *testing.T) {
	native := NativeEndian()
	// use a 32 byte key len
	var orig = make([]byte, SizeofXfrmAlgoAuth+32)
	rand.Read(orig)
	// set the key len to 256 bits
	var KeyLen uint32 = 0x00000100
	// Little Endian    Big Endian
	// orig[64] = 0     orig[64] = 0
	// orig[65] = 1     orig[65] = 0
	// orig[66] = 0     orig[66] = 1
	// orig[67] = 0     orig[67] = 0
	native.PutUint32(orig[64:68], KeyLen)
	safemsg := deserializeXfrmAlgoAuthSafe(orig)
	msg := DeserializeXfrmAlgoAuth(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmEncapTmpl) write(b []byte) {
	native := NativeEndian()
	native.PutUint16(b[0:2], msg.EncapType)
	native.PutUint16(b[2:4], msg.EncapSport)
	native.PutUint16(b[4:6], msg.EncapDport)
	copy(b[6:8], msg.Pad[:])
	msg.EncapOa.write(b[8:SizeofXfrmAddress])
}

func (msg *XfrmEncapTmpl) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmEncapTmpl)
	msg.write(b)
	return b
}

func deserializeXfrmEncapTmplSafe(b []byte) *XfrmEncapTmpl {
	var msg = XfrmEncapTmpl{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmEncapTmpl]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmEncapTmplDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmEncapTmpl)
	rand.Read(orig)
	safemsg := deserializeXfrmEncapTmplSafe(orig)
	msg := DeserializeXfrmEncapTmpl(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmMark) write(b []byte) {
	native := NativeEndian()
	native.PutUint32(b[0:4], msg.Value)
	native.PutUint32(b[4:8], msg.Mask)
}

func (msg *XfrmMark) serializeSafe() []byte {
	b := make([]byte, SizeofXfrmMark)
	msg.write(b)
	return b
}

func deserializeXfrmMarkSafe(b []byte) *XfrmMark {
	var msg = XfrmMark{}
	binary.Read(bytes.NewReader(b[0:SizeofXfrmMark]), NativeEndian(), &msg)
	return &msg
}

func TestXfrmMarkDeserializeSerialize(t *testing.T) {
	var orig = make([]byte, SizeofXfrmMark)
	rand.Read(orig)
	safemsg := deserializeXfrmMarkSafe(orig)
	msg := DeserializeXfrmMark(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}

func (msg *XfrmAlgoAEAD) write(b []byte) {
	native := NativeEndian()
	copy(b[0:64], msg.AlgName[:])
	native.PutUint32(b[64:68], msg.AlgKeyLen)
	native.PutUint32(b[68:72], msg.AlgICVLen)
	copy(b[72:msg.Len()], msg.AlgKey[:])
}

func (msg *XfrmAlgoAEAD) serializeSafe() []byte {
	b := make([]byte, msg.Len())
	msg.write(b)
	return b
}

func deserializeXfrmAlgoAEADSafe(b []byte) *XfrmAlgoAEAD {
	var msg = XfrmAlgoAEAD{}
	copy(msg.AlgName[:], b[0:64])
	binary.Read(bytes.NewReader(b[64:68]), NativeEndian(), &msg.AlgKeyLen)
	binary.Read(bytes.NewReader(b[68:72]), NativeEndian(), &msg.AlgICVLen)
	msg.AlgKey = b[72:msg.Len()]
	return &msg
}

func TestXfrmXfrmAlgoAeadDeserializeSerialize(t *testing.T) {
	native := NativeEndian()
	// use a 32 byte key len
	var orig = make([]byte, SizeofXfrmAlgoAEAD+36)
	rand.Read(orig)
	// set the key len to (256 + 32) bits
	var KeyLen uint32 = 0x00000120
	native.PutUint32(orig[64:68], KeyLen)
	safemsg := deserializeXfrmAlgoAEADSafe(orig)
	msg := DeserializeXfrmAlgoAEAD(orig)
	testDeserializeSerialize(t, orig, safemsg, msg)
}
