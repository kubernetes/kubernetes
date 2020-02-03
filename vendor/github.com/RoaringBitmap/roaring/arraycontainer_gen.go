package roaring

// NOTE: THIS FILE WAS PRODUCED BY THE
// MSGP CODE GENERATION TOOL (github.com/tinylib/msgp)
// DO NOT EDIT

import "github.com/tinylib/msgp/msgp"

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *arrayContainer) DecodeMsg(dc *msgp.Reader) (err error) {
	var field []byte
	_ = field
	var zbzg uint32
	zbzg, err = dc.ReadMapHeader()
	if err != nil {
		return
	}
	for zbzg > 0 {
		zbzg--
		field, err = dc.ReadMapKeyPtr()
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "content":
			var zbai uint32
			zbai, err = dc.ReadArrayHeader()
			if err != nil {
				return
			}
			if cap(z.content) >= int(zbai) {
				z.content = (z.content)[:zbai]
			} else {
				z.content = make([]uint16, zbai)
			}
			for zxvk := range z.content {
				z.content[zxvk], err = dc.ReadUint16()
				if err != nil {
					return
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
func (z *arrayContainer) EncodeMsg(en *msgp.Writer) (err error) {
	// map header, size 1
	// write "content"
	err = en.Append(0x81, 0xa7, 0x63, 0x6f, 0x6e, 0x74, 0x65, 0x6e, 0x74)
	if err != nil {
		return err
	}
	err = en.WriteArrayHeader(uint32(len(z.content)))
	if err != nil {
		return
	}
	for zxvk := range z.content {
		err = en.WriteUint16(z.content[zxvk])
		if err != nil {
			return
		}
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z *arrayContainer) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	// map header, size 1
	// string "content"
	o = append(o, 0x81, 0xa7, 0x63, 0x6f, 0x6e, 0x74, 0x65, 0x6e, 0x74)
	o = msgp.AppendArrayHeader(o, uint32(len(z.content)))
	for zxvk := range z.content {
		o = msgp.AppendUint16(o, z.content[zxvk])
	}
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *arrayContainer) UnmarshalMsg(bts []byte) (o []byte, err error) {
	var field []byte
	_ = field
	var zcmr uint32
	zcmr, bts, err = msgp.ReadMapHeaderBytes(bts)
	if err != nil {
		return
	}
	for zcmr > 0 {
		zcmr--
		field, bts, err = msgp.ReadMapKeyZC(bts)
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "content":
			var zajw uint32
			zajw, bts, err = msgp.ReadArrayHeaderBytes(bts)
			if err != nil {
				return
			}
			if cap(z.content) >= int(zajw) {
				z.content = (z.content)[:zajw]
			} else {
				z.content = make([]uint16, zajw)
			}
			for zxvk := range z.content {
				z.content[zxvk], bts, err = msgp.ReadUint16Bytes(bts)
				if err != nil {
					return
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
func (z *arrayContainer) Msgsize() (s int) {
	s = 1 + 8 + msgp.ArrayHeaderSize + (len(z.content) * (msgp.Uint16Size))
	return
}
