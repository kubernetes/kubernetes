package roaring

// NOTE: THIS FILE WAS PRODUCED BY THE
// MSGP CODE GENERATION TOOL (github.com/tinylib/msgp)
// DO NOT EDIT

import "github.com/tinylib/msgp/msgp"

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *bitmapContainer) DecodeMsg(dc *msgp.Reader) (err error) {
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
		case "cardinality":
			z.cardinality, err = dc.ReadInt()
			if err != nil {
				return
			}
		case "bitmap":
			var zbai uint32
			zbai, err = dc.ReadArrayHeader()
			if err != nil {
				return
			}
			if cap(z.bitmap) >= int(zbai) {
				z.bitmap = (z.bitmap)[:zbai]
			} else {
				z.bitmap = make([]uint64, zbai)
			}
			for zxvk := range z.bitmap {
				z.bitmap[zxvk], err = dc.ReadUint64()
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
func (z *bitmapContainer) EncodeMsg(en *msgp.Writer) (err error) {
	// map header, size 2
	// write "cardinality"
	err = en.Append(0x82, 0xab, 0x63, 0x61, 0x72, 0x64, 0x69, 0x6e, 0x61, 0x6c, 0x69, 0x74, 0x79)
	if err != nil {
		return err
	}
	err = en.WriteInt(z.cardinality)
	if err != nil {
		return
	}
	// write "bitmap"
	err = en.Append(0xa6, 0x62, 0x69, 0x74, 0x6d, 0x61, 0x70)
	if err != nil {
		return err
	}
	err = en.WriteArrayHeader(uint32(len(z.bitmap)))
	if err != nil {
		return
	}
	for zxvk := range z.bitmap {
		err = en.WriteUint64(z.bitmap[zxvk])
		if err != nil {
			return
		}
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z *bitmapContainer) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	// map header, size 2
	// string "cardinality"
	o = append(o, 0x82, 0xab, 0x63, 0x61, 0x72, 0x64, 0x69, 0x6e, 0x61, 0x6c, 0x69, 0x74, 0x79)
	o = msgp.AppendInt(o, z.cardinality)
	// string "bitmap"
	o = append(o, 0xa6, 0x62, 0x69, 0x74, 0x6d, 0x61, 0x70)
	o = msgp.AppendArrayHeader(o, uint32(len(z.bitmap)))
	for zxvk := range z.bitmap {
		o = msgp.AppendUint64(o, z.bitmap[zxvk])
	}
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *bitmapContainer) UnmarshalMsg(bts []byte) (o []byte, err error) {
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
		case "cardinality":
			z.cardinality, bts, err = msgp.ReadIntBytes(bts)
			if err != nil {
				return
			}
		case "bitmap":
			var zajw uint32
			zajw, bts, err = msgp.ReadArrayHeaderBytes(bts)
			if err != nil {
				return
			}
			if cap(z.bitmap) >= int(zajw) {
				z.bitmap = (z.bitmap)[:zajw]
			} else {
				z.bitmap = make([]uint64, zajw)
			}
			for zxvk := range z.bitmap {
				z.bitmap[zxvk], bts, err = msgp.ReadUint64Bytes(bts)
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
func (z *bitmapContainer) Msgsize() (s int) {
	s = 1 + 12 + msgp.IntSize + 7 + msgp.ArrayHeaderSize + (len(z.bitmap) * (msgp.Uint64Size))
	return
}

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *bitmapContainerShortIterator) DecodeMsg(dc *msgp.Reader) (err error) {
	var field []byte
	_ = field
	var zhct uint32
	zhct, err = dc.ReadMapHeader()
	if err != nil {
		return
	}
	for zhct > 0 {
		zhct--
		field, err = dc.ReadMapKeyPtr()
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "ptr":
			if dc.IsNil() {
				err = dc.ReadNil()
				if err != nil {
					return
				}
				z.ptr = nil
			} else {
				if z.ptr == nil {
					z.ptr = new(bitmapContainer)
				}
				var zcua uint32
				zcua, err = dc.ReadMapHeader()
				if err != nil {
					return
				}
				for zcua > 0 {
					zcua--
					field, err = dc.ReadMapKeyPtr()
					if err != nil {
						return
					}
					switch msgp.UnsafeString(field) {
					case "cardinality":
						z.ptr.cardinality, err = dc.ReadInt()
						if err != nil {
							return
						}
					case "bitmap":
						var zxhx uint32
						zxhx, err = dc.ReadArrayHeader()
						if err != nil {
							return
						}
						if cap(z.ptr.bitmap) >= int(zxhx) {
							z.ptr.bitmap = (z.ptr.bitmap)[:zxhx]
						} else {
							z.ptr.bitmap = make([]uint64, zxhx)
						}
						for zwht := range z.ptr.bitmap {
							z.ptr.bitmap[zwht], err = dc.ReadUint64()
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
			}
		case "i":
			z.i, err = dc.ReadInt()
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
func (z *bitmapContainerShortIterator) EncodeMsg(en *msgp.Writer) (err error) {
	// map header, size 2
	// write "ptr"
	err = en.Append(0x82, 0xa3, 0x70, 0x74, 0x72)
	if err != nil {
		return err
	}
	if z.ptr == nil {
		err = en.WriteNil()
		if err != nil {
			return
		}
	} else {
		// map header, size 2
		// write "cardinality"
		err = en.Append(0x82, 0xab, 0x63, 0x61, 0x72, 0x64, 0x69, 0x6e, 0x61, 0x6c, 0x69, 0x74, 0x79)
		if err != nil {
			return err
		}
		err = en.WriteInt(z.ptr.cardinality)
		if err != nil {
			return
		}
		// write "bitmap"
		err = en.Append(0xa6, 0x62, 0x69, 0x74, 0x6d, 0x61, 0x70)
		if err != nil {
			return err
		}
		err = en.WriteArrayHeader(uint32(len(z.ptr.bitmap)))
		if err != nil {
			return
		}
		for zwht := range z.ptr.bitmap {
			err = en.WriteUint64(z.ptr.bitmap[zwht])
			if err != nil {
				return
			}
		}
	}
	// write "i"
	err = en.Append(0xa1, 0x69)
	if err != nil {
		return err
	}
	err = en.WriteInt(z.i)
	if err != nil {
		return
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z *bitmapContainerShortIterator) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	// map header, size 2
	// string "ptr"
	o = append(o, 0x82, 0xa3, 0x70, 0x74, 0x72)
	if z.ptr == nil {
		o = msgp.AppendNil(o)
	} else {
		// map header, size 2
		// string "cardinality"
		o = append(o, 0x82, 0xab, 0x63, 0x61, 0x72, 0x64, 0x69, 0x6e, 0x61, 0x6c, 0x69, 0x74, 0x79)
		o = msgp.AppendInt(o, z.ptr.cardinality)
		// string "bitmap"
		o = append(o, 0xa6, 0x62, 0x69, 0x74, 0x6d, 0x61, 0x70)
		o = msgp.AppendArrayHeader(o, uint32(len(z.ptr.bitmap)))
		for zwht := range z.ptr.bitmap {
			o = msgp.AppendUint64(o, z.ptr.bitmap[zwht])
		}
	}
	// string "i"
	o = append(o, 0xa1, 0x69)
	o = msgp.AppendInt(o, z.i)
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *bitmapContainerShortIterator) UnmarshalMsg(bts []byte) (o []byte, err error) {
	var field []byte
	_ = field
	var zlqf uint32
	zlqf, bts, err = msgp.ReadMapHeaderBytes(bts)
	if err != nil {
		return
	}
	for zlqf > 0 {
		zlqf--
		field, bts, err = msgp.ReadMapKeyZC(bts)
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "ptr":
			if msgp.IsNil(bts) {
				bts, err = msgp.ReadNilBytes(bts)
				if err != nil {
					return
				}
				z.ptr = nil
			} else {
				if z.ptr == nil {
					z.ptr = new(bitmapContainer)
				}
				var zdaf uint32
				zdaf, bts, err = msgp.ReadMapHeaderBytes(bts)
				if err != nil {
					return
				}
				for zdaf > 0 {
					zdaf--
					field, bts, err = msgp.ReadMapKeyZC(bts)
					if err != nil {
						return
					}
					switch msgp.UnsafeString(field) {
					case "cardinality":
						z.ptr.cardinality, bts, err = msgp.ReadIntBytes(bts)
						if err != nil {
							return
						}
					case "bitmap":
						var zpks uint32
						zpks, bts, err = msgp.ReadArrayHeaderBytes(bts)
						if err != nil {
							return
						}
						if cap(z.ptr.bitmap) >= int(zpks) {
							z.ptr.bitmap = (z.ptr.bitmap)[:zpks]
						} else {
							z.ptr.bitmap = make([]uint64, zpks)
						}
						for zwht := range z.ptr.bitmap {
							z.ptr.bitmap[zwht], bts, err = msgp.ReadUint64Bytes(bts)
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
			}
		case "i":
			z.i, bts, err = msgp.ReadIntBytes(bts)
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
func (z *bitmapContainerShortIterator) Msgsize() (s int) {
	s = 1 + 4
	if z.ptr == nil {
		s += msgp.NilSize
	} else {
		s += 1 + 12 + msgp.IntSize + 7 + msgp.ArrayHeaderSize + (len(z.ptr.bitmap) * (msgp.Uint64Size))
	}
	s += 2 + msgp.IntSize
	return
}
