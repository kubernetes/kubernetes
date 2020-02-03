package roaring

// NOTE: THIS FILE WAS PRODUCED BY THE
// MSGP CODE GENERATION TOOL (github.com/tinylib/msgp)
// DO NOT EDIT

import (
	"github.com/tinylib/msgp/msgp"
)

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *containerSerz) DecodeMsg(dc *msgp.Reader) (err error) {
	var field []byte
	_ = field
	var zxvk uint32
	zxvk, err = dc.ReadMapHeader()
	if err != nil {
		return
	}
	for zxvk > 0 {
		zxvk--
		field, err = dc.ReadMapKeyPtr()
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "t":
			{
				var zbzg uint8
				zbzg, err = dc.ReadUint8()
				z.t = contype(zbzg)
			}
			if err != nil {
				return
			}
		case "r":
			err = z.r.DecodeMsg(dc)
			if err != nil {
				return
			}
		default:
			err = dc.Skip()
			if err != nil {
				return
			}
		}
	}
	return
}

// Deprecated: EncodeMsg implements msgp.Encodable
func (z *containerSerz) EncodeMsg(en *msgp.Writer) (err error) {
	// map header, size 2
	// write "t"
	err = en.Append(0x82, 0xa1, 0x74)
	if err != nil {
		return err
	}
	err = en.WriteUint8(uint8(z.t))
	if err != nil {
		return
	}
	// write "r"
	err = en.Append(0xa1, 0x72)
	if err != nil {
		return err
	}
	err = z.r.EncodeMsg(en)
	if err != nil {
		return
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z *containerSerz) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	// map header, size 2
	// string "t"
	o = append(o, 0x82, 0xa1, 0x74)
	o = msgp.AppendUint8(o, uint8(z.t))
	// string "r"
	o = append(o, 0xa1, 0x72)
	o, err = z.r.MarshalMsg(o)
	if err != nil {
		return
	}
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *containerSerz) UnmarshalMsg(bts []byte) (o []byte, err error) {
	var field []byte
	_ = field
	var zbai uint32
	zbai, bts, err = msgp.ReadMapHeaderBytes(bts)
	if err != nil {
		return
	}
	for zbai > 0 {
		zbai--
		field, bts, err = msgp.ReadMapKeyZC(bts)
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "t":
			{
				var zcmr uint8
				zcmr, bts, err = msgp.ReadUint8Bytes(bts)
				z.t = contype(zcmr)
			}
			if err != nil {
				return
			}
		case "r":
			bts, err = z.r.UnmarshalMsg(bts)
			if err != nil {
				return
			}
		default:
			bts, err = msgp.Skip(bts)
			if err != nil {
				return
			}
		}
	}
	o = bts
	return
}

// Deprecated: Msgsize returns an upper bound estimate of the number of bytes occupied by the serialized message
func (z *containerSerz) Msgsize() (s int) {
	s = 1 + 2 + msgp.Uint8Size + 2 + z.r.Msgsize()
	return
}

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *contype) DecodeMsg(dc *msgp.Reader) (err error) {
	{
		var zajw uint8
		zajw, err = dc.ReadUint8()
		(*z) = contype(zajw)
	}
	if err != nil {
		return
	}
	return
}

// Deprecated: EncodeMsg implements msgp.Encodable
func (z contype) EncodeMsg(en *msgp.Writer) (err error) {
	err = en.WriteUint8(uint8(z))
	if err != nil {
		return
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z contype) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	o = msgp.AppendUint8(o, uint8(z))
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *contype) UnmarshalMsg(bts []byte) (o []byte, err error) {
	{
		var zwht uint8
		zwht, bts, err = msgp.ReadUint8Bytes(bts)
		(*z) = contype(zwht)
	}
	if err != nil {
		return
	}
	o = bts
	return
}

// Deprecated: Msgsize returns an upper bound estimate of the number of bytes occupied by the serialized message
func (z contype) Msgsize() (s int) {
	s = msgp.Uint8Size
	return
}

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *roaringArray) DecodeMsg(dc *msgp.Reader) (err error) {
	var field []byte
	_ = field
	var zlqf uint32
	zlqf, err = dc.ReadMapHeader()
	if err != nil {
		return
	}
	for zlqf > 0 {
		zlqf--
		field, err = dc.ReadMapKeyPtr()
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "keys":
			var zdaf uint32
			zdaf, err = dc.ReadArrayHeader()
			if err != nil {
				return
			}
			if cap(z.keys) >= int(zdaf) {
				z.keys = (z.keys)[:zdaf]
			} else {
				z.keys = make([]uint16, zdaf)
			}
			for zhct := range z.keys {
				z.keys[zhct], err = dc.ReadUint16()
				if err != nil {
					return
				}
			}
		case "needCopyOnWrite":
			var zpks uint32
			zpks, err = dc.ReadArrayHeader()
			if err != nil {
				return
			}
			if cap(z.needCopyOnWrite) >= int(zpks) {
				z.needCopyOnWrite = (z.needCopyOnWrite)[:zpks]
			} else {
				z.needCopyOnWrite = make([]bool, zpks)
			}
			for zcua := range z.needCopyOnWrite {
				z.needCopyOnWrite[zcua], err = dc.ReadBool()
				if err != nil {
					return
				}
			}
		case "copyOnWrite":
			z.copyOnWrite, err = dc.ReadBool()
			if err != nil {
				return
			}
		case "conserz":
			var zjfb uint32
			zjfb, err = dc.ReadArrayHeader()
			if err != nil {
				return
			}
			if cap(z.conserz) >= int(zjfb) {
				z.conserz = (z.conserz)[:zjfb]
			} else {
				z.conserz = make([]containerSerz, zjfb)
			}
			for zxhx := range z.conserz {
				var zcxo uint32
				zcxo, err = dc.ReadMapHeader()
				if err != nil {
					return
				}
				for zcxo > 0 {
					zcxo--
					field, err = dc.ReadMapKeyPtr()
					if err != nil {
						return
					}
					switch msgp.UnsafeString(field) {
					case "t":
						{
							var zeff uint8
							zeff, err = dc.ReadUint8()
							z.conserz[zxhx].t = contype(zeff)
						}
						if err != nil {
							return
						}
					case "r":
						err = z.conserz[zxhx].r.DecodeMsg(dc)
						if err != nil {
							return
						}
					default:
						err = dc.Skip()
						if err != nil {
							return
						}
					}
				}
			}
		default:
			err = dc.Skip()
			if err != nil {
				return
			}
		}
	}
	return
}

// Deprecated: EncodeMsg implements msgp.Encodable
func (z *roaringArray) EncodeMsg(en *msgp.Writer) (err error) {
	// map header, size 4
	// write "keys"
	err = en.Append(0x84, 0xa4, 0x6b, 0x65, 0x79, 0x73)
	if err != nil {
		return err
	}
	err = en.WriteArrayHeader(uint32(len(z.keys)))
	if err != nil {
		return
	}
	for zhct := range z.keys {
		err = en.WriteUint16(z.keys[zhct])
		if err != nil {
			return
		}
	}
	// write "needCopyOnWrite"
	err = en.Append(0xaf, 0x6e, 0x65, 0x65, 0x64, 0x43, 0x6f, 0x70, 0x79, 0x4f, 0x6e, 0x57, 0x72, 0x69, 0x74, 0x65)
	if err != nil {
		return err
	}
	err = en.WriteArrayHeader(uint32(len(z.needCopyOnWrite)))
	if err != nil {
		return
	}
	for zcua := range z.needCopyOnWrite {
		err = en.WriteBool(z.needCopyOnWrite[zcua])
		if err != nil {
			return
		}
	}
	// write "copyOnWrite"
	err = en.Append(0xab, 0x63, 0x6f, 0x70, 0x79, 0x4f, 0x6e, 0x57, 0x72, 0x69, 0x74, 0x65)
	if err != nil {
		return err
	}
	err = en.WriteBool(z.copyOnWrite)
	if err != nil {
		return
	}
	// write "conserz"
	err = en.Append(0xa7, 0x63, 0x6f, 0x6e, 0x73, 0x65, 0x72, 0x7a)
	if err != nil {
		return err
	}
	err = en.WriteArrayHeader(uint32(len(z.conserz)))
	if err != nil {
		return
	}
	for zxhx := range z.conserz {
		// map header, size 2
		// write "t"
		err = en.Append(0x82, 0xa1, 0x74)
		if err != nil {
			return err
		}
		err = en.WriteUint8(uint8(z.conserz[zxhx].t))
		if err != nil {
			return
		}
		// write "r"
		err = en.Append(0xa1, 0x72)
		if err != nil {
			return err
		}
		err = z.conserz[zxhx].r.EncodeMsg(en)
		if err != nil {
			return
		}
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z *roaringArray) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	// map header, size 4
	// string "keys"
	o = append(o, 0x84, 0xa4, 0x6b, 0x65, 0x79, 0x73)
	o = msgp.AppendArrayHeader(o, uint32(len(z.keys)))
	for zhct := range z.keys {
		o = msgp.AppendUint16(o, z.keys[zhct])
	}
	// string "needCopyOnWrite"
	o = append(o, 0xaf, 0x6e, 0x65, 0x65, 0x64, 0x43, 0x6f, 0x70, 0x79, 0x4f, 0x6e, 0x57, 0x72, 0x69, 0x74, 0x65)
	o = msgp.AppendArrayHeader(o, uint32(len(z.needCopyOnWrite)))
	for zcua := range z.needCopyOnWrite {
		o = msgp.AppendBool(o, z.needCopyOnWrite[zcua])
	}
	// string "copyOnWrite"
	o = append(o, 0xab, 0x63, 0x6f, 0x70, 0x79, 0x4f, 0x6e, 0x57, 0x72, 0x69, 0x74, 0x65)
	o = msgp.AppendBool(o, z.copyOnWrite)
	// string "conserz"
	o = append(o, 0xa7, 0x63, 0x6f, 0x6e, 0x73, 0x65, 0x72, 0x7a)
	o = msgp.AppendArrayHeader(o, uint32(len(z.conserz)))
	for zxhx := range z.conserz {
		// map header, size 2
		// string "t"
		o = append(o, 0x82, 0xa1, 0x74)
		o = msgp.AppendUint8(o, uint8(z.conserz[zxhx].t))
		// string "r"
		o = append(o, 0xa1, 0x72)
		o, err = z.conserz[zxhx].r.MarshalMsg(o)
		if err != nil {
			return
		}
	}
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *roaringArray) UnmarshalMsg(bts []byte) (o []byte, err error) {
	var field []byte
	_ = field
	var zrsw uint32
	zrsw, bts, err = msgp.ReadMapHeaderBytes(bts)
	if err != nil {
		return
	}
	for zrsw > 0 {
		zrsw--
		field, bts, err = msgp.ReadMapKeyZC(bts)
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "keys":
			var zxpk uint32
			zxpk, bts, err = msgp.ReadArrayHeaderBytes(bts)
			if err != nil {
				return
			}
			if cap(z.keys) >= int(zxpk) {
				z.keys = (z.keys)[:zxpk]
			} else {
				z.keys = make([]uint16, zxpk)
			}
			for zhct := range z.keys {
				z.keys[zhct], bts, err = msgp.ReadUint16Bytes(bts)
				if err != nil {
					return
				}
			}
		case "needCopyOnWrite":
			var zdnj uint32
			zdnj, bts, err = msgp.ReadArrayHeaderBytes(bts)
			if err != nil {
				return
			}
			if cap(z.needCopyOnWrite) >= int(zdnj) {
				z.needCopyOnWrite = (z.needCopyOnWrite)[:zdnj]
			} else {
				z.needCopyOnWrite = make([]bool, zdnj)
			}
			for zcua := range z.needCopyOnWrite {
				z.needCopyOnWrite[zcua], bts, err = msgp.ReadBoolBytes(bts)
				if err != nil {
					return
				}
			}
		case "copyOnWrite":
			z.copyOnWrite, bts, err = msgp.ReadBoolBytes(bts)
			if err != nil {
				return
			}
		case "conserz":
			var zobc uint32
			zobc, bts, err = msgp.ReadArrayHeaderBytes(bts)
			if err != nil {
				return
			}
			if cap(z.conserz) >= int(zobc) {
				z.conserz = (z.conserz)[:zobc]
			} else {
				z.conserz = make([]containerSerz, zobc)
			}
			for zxhx := range z.conserz {
				var zsnv uint32
				zsnv, bts, err = msgp.ReadMapHeaderBytes(bts)
				if err != nil {
					return
				}
				for zsnv > 0 {
					zsnv--
					field, bts, err = msgp.ReadMapKeyZC(bts)
					if err != nil {
						return
					}
					switch msgp.UnsafeString(field) {
					case "t":
						{
							var zkgt uint8
							zkgt, bts, err = msgp.ReadUint8Bytes(bts)
							z.conserz[zxhx].t = contype(zkgt)
						}
						if err != nil {
							return
						}
					case "r":
						bts, err = z.conserz[zxhx].r.UnmarshalMsg(bts)
						if err != nil {
							return
						}
					default:
						bts, err = msgp.Skip(bts)
						if err != nil {
							return
						}
					}
				}
			}
		default:
			bts, err = msgp.Skip(bts)
			if err != nil {
				return
			}
		}
	}
	o = bts
	return
}

// Deprecated: Msgsize returns an upper bound estimate of the number of bytes occupied by the serialized message
func (z *roaringArray) Msgsize() (s int) {
	s = 1 + 5 + msgp.ArrayHeaderSize + (len(z.keys) * (msgp.Uint16Size)) + 16 + msgp.ArrayHeaderSize + (len(z.needCopyOnWrite) * (msgp.BoolSize)) + 12 + msgp.BoolSize + 8 + msgp.ArrayHeaderSize
	for zxhx := range z.conserz {
		s += 1 + 2 + msgp.Uint8Size + 2 + z.conserz[zxhx].r.Msgsize()
	}
	return
}
