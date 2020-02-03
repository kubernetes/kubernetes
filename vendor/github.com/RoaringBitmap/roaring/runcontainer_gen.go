package roaring

// NOTE: THIS FILE WAS PRODUCED BY THE
// MSGP CODE GENERATION TOOL (github.com/tinylib/msgp)
// DO NOT EDIT

import "github.com/tinylib/msgp/msgp"

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *addHelper16) DecodeMsg(dc *msgp.Reader) (err error) {
	var field []byte
	_ = field
	var zbai uint32
	zbai, err = dc.ReadMapHeader()
	if err != nil {
		return
	}
	for zbai > 0 {
		zbai--
		field, err = dc.ReadMapKeyPtr()
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "runstart":
			z.runstart, err = dc.ReadUint16()
			if err != nil {
				return
			}
		case "runlen":
			z.runlen, err = dc.ReadUint16()
			if err != nil {
				return
			}
		case "actuallyAdded":
			z.actuallyAdded, err = dc.ReadUint16()
			if err != nil {
				return
			}
		case "m":
			var zcmr uint32
			zcmr, err = dc.ReadArrayHeader()
			if err != nil {
				return
			}
			if cap(z.m) >= int(zcmr) {
				z.m = (z.m)[:zcmr]
			} else {
				z.m = make([]interval16, zcmr)
			}
			for zxvk := range z.m {
				var zajw uint32
				zajw, err = dc.ReadMapHeader()
				if err != nil {
					return
				}
				for zajw > 0 {
					zajw--
					field, err = dc.ReadMapKeyPtr()
					if err != nil {
						return
					}
					switch msgp.UnsafeString(field) {
					case "start":
						z.m[zxvk].start, err = dc.ReadUint16()
						if err != nil {
							return
						}
					case "last":
						z.m[zxvk].length, err = dc.ReadUint16()
						z.m[zxvk].length -= z.m[zxvk].start
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
		case "rc":
			if dc.IsNil() {
				err = dc.ReadNil()
				if err != nil {
					return
				}
				z.rc = nil
			} else {
				if z.rc == nil {
					z.rc = new(runContainer16)
				}
				var zwht uint32
				zwht, err = dc.ReadMapHeader()
				if err != nil {
					return
				}
				for zwht > 0 {
					zwht--
					field, err = dc.ReadMapKeyPtr()
					if err != nil {
						return
					}
					switch msgp.UnsafeString(field) {
					case "iv":
						var zhct uint32
						zhct, err = dc.ReadArrayHeader()
						if err != nil {
							return
						}
						if cap(z.rc.iv) >= int(zhct) {
							z.rc.iv = (z.rc.iv)[:zhct]
						} else {
							z.rc.iv = make([]interval16, zhct)
						}
						for zbzg := range z.rc.iv {
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
								case "start":
									z.rc.iv[zbzg].start, err = dc.ReadUint16()
									if err != nil {
										return
									}
								case "last":
									z.rc.iv[zbzg].length, err = dc.ReadUint16()
									z.rc.iv[zbzg].length -= z.rc.iv[zbzg].start
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
					case "card":
						z.rc.card, err = dc.ReadInt64()
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
func (z *addHelper16) EncodeMsg(en *msgp.Writer) (err error) {
	// map header, size 5
	// write "runstart"
	err = en.Append(0x85, 0xa8, 0x72, 0x75, 0x6e, 0x73, 0x74, 0x61, 0x72, 0x74)
	if err != nil {
		return err
	}
	err = en.WriteUint16(z.runstart)
	if err != nil {
		return
	}
	// write "runlen"
	err = en.Append(0xa6, 0x72, 0x75, 0x6e, 0x6c, 0x65, 0x6e)
	if err != nil {
		return err
	}
	err = en.WriteUint16(z.runlen)
	if err != nil {
		return
	}
	// write "actuallyAdded"
	err = en.Append(0xad, 0x61, 0x63, 0x74, 0x75, 0x61, 0x6c, 0x6c, 0x79, 0x41, 0x64, 0x64, 0x65, 0x64)
	if err != nil {
		return err
	}
	err = en.WriteUint16(z.actuallyAdded)
	if err != nil {
		return
	}
	// write "m"
	err = en.Append(0xa1, 0x6d)
	if err != nil {
		return err
	}
	err = en.WriteArrayHeader(uint32(len(z.m)))
	if err != nil {
		return
	}
	for zxvk := range z.m {
		// map header, size 2
		// write "start"
		err = en.Append(0x82, 0xa5, 0x73, 0x74, 0x61, 0x72, 0x74)
		if err != nil {
			return err
		}
		err = en.WriteUint16(z.m[zxvk].start)
		if err != nil {
			return
		}
		// write "last"
		err = en.Append(0xa4, 0x6c, 0x61, 0x73, 0x74)
		if err != nil {
			return err
		}
		err = en.WriteUint16(z.m[zxvk].last())
		if err != nil {
			return
		}
	}
	// write "rc"
	err = en.Append(0xa2, 0x72, 0x63)
	if err != nil {
		return err
	}
	if z.rc == nil {
		err = en.WriteNil()
		if err != nil {
			return
		}
	} else {
		// map header, size 2
		// write "iv"
		err = en.Append(0x82, 0xa2, 0x69, 0x76)
		if err != nil {
			return err
		}
		err = en.WriteArrayHeader(uint32(len(z.rc.iv)))
		if err != nil {
			return
		}
		for zbzg := range z.rc.iv {
			// map header, size 2
			// write "start"
			err = en.Append(0x82, 0xa5, 0x73, 0x74, 0x61, 0x72, 0x74)
			if err != nil {
				return err
			}
			err = en.WriteUint16(z.rc.iv[zbzg].start)
			if err != nil {
				return
			}
			// write "last"
			err = en.Append(0xa4, 0x6c, 0x61, 0x73, 0x74)
			if err != nil {
				return err
			}
			err = en.WriteUint16(z.rc.iv[zbzg].last())
			if err != nil {
				return
			}
		}
		// write "card"
		err = en.Append(0xa4, 0x63, 0x61, 0x72, 0x64)
		if err != nil {
			return err
		}
		err = en.WriteInt64(z.rc.card)
		if err != nil {
			return
		}
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z *addHelper16) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	// map header, size 5
	// string "runstart"
	o = append(o, 0x85, 0xa8, 0x72, 0x75, 0x6e, 0x73, 0x74, 0x61, 0x72, 0x74)
	o = msgp.AppendUint16(o, z.runstart)
	// string "runlen"
	o = append(o, 0xa6, 0x72, 0x75, 0x6e, 0x6c, 0x65, 0x6e)
	o = msgp.AppendUint16(o, z.runlen)
	// string "actuallyAdded"
	o = append(o, 0xad, 0x61, 0x63, 0x74, 0x75, 0x61, 0x6c, 0x6c, 0x79, 0x41, 0x64, 0x64, 0x65, 0x64)
	o = msgp.AppendUint16(o, z.actuallyAdded)
	// string "m"
	o = append(o, 0xa1, 0x6d)
	o = msgp.AppendArrayHeader(o, uint32(len(z.m)))
	for zxvk := range z.m {
		// map header, size 2
		// string "start"
		o = append(o, 0x82, 0xa5, 0x73, 0x74, 0x61, 0x72, 0x74)
		o = msgp.AppendUint16(o, z.m[zxvk].start)
		// string "last"
		o = append(o, 0xa4, 0x6c, 0x61, 0x73, 0x74)
		o = msgp.AppendUint16(o, z.m[zxvk].last())
	}
	// string "rc"
	o = append(o, 0xa2, 0x72, 0x63)
	if z.rc == nil {
		o = msgp.AppendNil(o)
	} else {
		// map header, size 2
		// string "iv"
		o = append(o, 0x82, 0xa2, 0x69, 0x76)
		o = msgp.AppendArrayHeader(o, uint32(len(z.rc.iv)))
		for zbzg := range z.rc.iv {
			// map header, size 2
			// string "start"
			o = append(o, 0x82, 0xa5, 0x73, 0x74, 0x61, 0x72, 0x74)
			o = msgp.AppendUint16(o, z.rc.iv[zbzg].start)
			// string "last"
			o = append(o, 0xa4, 0x6c, 0x61, 0x73, 0x74)
			o = msgp.AppendUint16(o, z.rc.iv[zbzg].last())
		}
		// string "card"
		o = append(o, 0xa4, 0x63, 0x61, 0x72, 0x64)
		o = msgp.AppendInt64(o, z.rc.card)
	}
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *addHelper16) UnmarshalMsg(bts []byte) (o []byte, err error) {
	var field []byte
	_ = field
	var zxhx uint32
	zxhx, bts, err = msgp.ReadMapHeaderBytes(bts)
	if err != nil {
		return
	}
	for zxhx > 0 {
		zxhx--
		field, bts, err = msgp.ReadMapKeyZC(bts)
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "runstart":
			z.runstart, bts, err = msgp.ReadUint16Bytes(bts)
			if err != nil {
				return
			}
		case "runlen":
			z.runlen, bts, err = msgp.ReadUint16Bytes(bts)
			if err != nil {
				return
			}
		case "actuallyAdded":
			z.actuallyAdded, bts, err = msgp.ReadUint16Bytes(bts)
			if err != nil {
				return
			}
		case "m":
			var zlqf uint32
			zlqf, bts, err = msgp.ReadArrayHeaderBytes(bts)
			if err != nil {
				return
			}
			if cap(z.m) >= int(zlqf) {
				z.m = (z.m)[:zlqf]
			} else {
				z.m = make([]interval16, zlqf)
			}
			for zxvk := range z.m {
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
					case "start":
						z.m[zxvk].start, bts, err = msgp.ReadUint16Bytes(bts)
						if err != nil {
							return
						}
					case "last":
						z.m[zxvk].length, bts, err = msgp.ReadUint16Bytes(bts)
						z.m[zxvk].length -= z.m[zxvk].start
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
		case "rc":
			if msgp.IsNil(bts) {
				bts, err = msgp.ReadNilBytes(bts)
				if err != nil {
					return
				}
				z.rc = nil
			} else {
				if z.rc == nil {
					z.rc = new(runContainer16)
				}
				var zpks uint32
				zpks, bts, err = msgp.ReadMapHeaderBytes(bts)
				if err != nil {
					return
				}
				for zpks > 0 {
					zpks--
					field, bts, err = msgp.ReadMapKeyZC(bts)
					if err != nil {
						return
					}
					switch msgp.UnsafeString(field) {
					case "iv":
						var zjfb uint32
						zjfb, bts, err = msgp.ReadArrayHeaderBytes(bts)
						if err != nil {
							return
						}
						if cap(z.rc.iv) >= int(zjfb) {
							z.rc.iv = (z.rc.iv)[:zjfb]
						} else {
							z.rc.iv = make([]interval16, zjfb)
						}
						for zbzg := range z.rc.iv {
							var zcxo uint32
							zcxo, bts, err = msgp.ReadMapHeaderBytes(bts)
							if err != nil {
								return
							}
							for zcxo > 0 {
								zcxo--
								field, bts, err = msgp.ReadMapKeyZC(bts)
								if err != nil {
									return
								}
								switch msgp.UnsafeString(field) {
								case "start":
									z.rc.iv[zbzg].start, bts, err = msgp.ReadUint16Bytes(bts)
									if err != nil {
										return
									}
								case "last":
									z.rc.iv[zbzg].length, bts, err = msgp.ReadUint16Bytes(bts)
									z.rc.iv[zbzg].length -= z.rc.iv[zbzg].start
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
					case "card":
						z.rc.card, bts, err = msgp.ReadInt64Bytes(bts)
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
func (z *addHelper16) Msgsize() (s int) {
	s = 1 + 9 + msgp.Uint16Size + 7 + msgp.Uint16Size + 14 + msgp.Uint16Size + 2 + msgp.ArrayHeaderSize + (len(z.m) * (12 + msgp.Uint16Size + msgp.Uint16Size)) + 3
	if z.rc == nil {
		s += msgp.NilSize
	} else {
		s += 1 + 3 + msgp.ArrayHeaderSize + (len(z.rc.iv) * (12 + msgp.Uint16Size + msgp.Uint16Size)) + 5 + msgp.Int64Size
	}
	return
}

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *interval16) DecodeMsg(dc *msgp.Reader) (err error) {
	var field []byte
	_ = field
	var zeff uint32
	zeff, err = dc.ReadMapHeader()
	if err != nil {
		return
	}
	for zeff > 0 {
		zeff--
		field, err = dc.ReadMapKeyPtr()
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "start":
			z.start, err = dc.ReadUint16()
			if err != nil {
				return
			}
		case "last":
			z.length, err = dc.ReadUint16()
			z.length = -z.start
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
func (z interval16) EncodeMsg(en *msgp.Writer) (err error) {
	// map header, size 2
	// write "start"
	err = en.Append(0x82, 0xa5, 0x73, 0x74, 0x61, 0x72, 0x74)
	if err != nil {
		return err
	}
	err = en.WriteUint16(z.start)
	if err != nil {
		return
	}
	// write "last"
	err = en.Append(0xa4, 0x6c, 0x61, 0x73, 0x74)
	if err != nil {
		return err
	}
	err = en.WriteUint16(z.last())
	if err != nil {
		return
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z interval16) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	// map header, size 2
	// string "start"
	o = append(o, 0x82, 0xa5, 0x73, 0x74, 0x61, 0x72, 0x74)
	o = msgp.AppendUint16(o, z.start)
	// string "last"
	o = append(o, 0xa4, 0x6c, 0x61, 0x73, 0x74)
	o = msgp.AppendUint16(o, z.last())
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *interval16) UnmarshalMsg(bts []byte) (o []byte, err error) {
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
		case "start":
			z.start, bts, err = msgp.ReadUint16Bytes(bts)
			if err != nil {
				return
			}
		case "last":
			z.length, bts, err = msgp.ReadUint16Bytes(bts)
			z.length -= z.start
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
func (z interval16) Msgsize() (s int) {
	s = 1 + 6 + msgp.Uint16Size + 5 + msgp.Uint16Size
	return
}

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *runContainer16) DecodeMsg(dc *msgp.Reader) (err error) {
	var field []byte
	_ = field
	var zdnj uint32
	zdnj, err = dc.ReadMapHeader()
	if err != nil {
		return
	}
	for zdnj > 0 {
		zdnj--
		field, err = dc.ReadMapKeyPtr()
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "iv":
			var zobc uint32
			zobc, err = dc.ReadArrayHeader()
			if err != nil {
				return
			}
			if cap(z.iv) >= int(zobc) {
				z.iv = (z.iv)[:zobc]
			} else {
				z.iv = make([]interval16, zobc)
			}
			for zxpk := range z.iv {
				var zsnv uint32
				zsnv, err = dc.ReadMapHeader()
				if err != nil {
					return
				}
				for zsnv > 0 {
					zsnv--
					field, err = dc.ReadMapKeyPtr()
					if err != nil {
						return
					}
					switch msgp.UnsafeString(field) {
					case "start":
						z.iv[zxpk].start, err = dc.ReadUint16()
						if err != nil {
							return
						}
					case "last":
						z.iv[zxpk].length, err = dc.ReadUint16()
						z.iv[zxpk].length -= z.iv[zxpk].start
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
		case "card":
			z.card, err = dc.ReadInt64()
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
func (z *runContainer16) EncodeMsg(en *msgp.Writer) (err error) {
	// map header, size 2
	// write "iv"
	err = en.Append(0x82, 0xa2, 0x69, 0x76)
	if err != nil {
		return err
	}
	err = en.WriteArrayHeader(uint32(len(z.iv)))
	if err != nil {
		return
	}
	for zxpk := range z.iv {
		// map header, size 2
		// write "start"
		err = en.Append(0x82, 0xa5, 0x73, 0x74, 0x61, 0x72, 0x74)
		if err != nil {
			return err
		}
		err = en.WriteUint16(z.iv[zxpk].start)
		if err != nil {
			return
		}
		// write "last"
		err = en.Append(0xa4, 0x6c, 0x61, 0x73, 0x74)
		if err != nil {
			return err
		}
		err = en.WriteUint16(z.iv[zxpk].last())
		if err != nil {
			return
		}
	}
	// write "card"
	err = en.Append(0xa4, 0x63, 0x61, 0x72, 0x64)
	if err != nil {
		return err
	}
	err = en.WriteInt64(z.card)
	if err != nil {
		return
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z *runContainer16) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	// map header, size 2
	// string "iv"
	o = append(o, 0x82, 0xa2, 0x69, 0x76)
	o = msgp.AppendArrayHeader(o, uint32(len(z.iv)))
	for zxpk := range z.iv {
		// map header, size 2
		// string "start"
		o = append(o, 0x82, 0xa5, 0x73, 0x74, 0x61, 0x72, 0x74)
		o = msgp.AppendUint16(o, z.iv[zxpk].start)
		// string "last"
		o = append(o, 0xa4, 0x6c, 0x61, 0x73, 0x74)
		o = msgp.AppendUint16(o, z.iv[zxpk].last())
	}
	// string "card"
	o = append(o, 0xa4, 0x63, 0x61, 0x72, 0x64)
	o = msgp.AppendInt64(o, z.card)
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *runContainer16) UnmarshalMsg(bts []byte) (o []byte, err error) {
	var field []byte
	_ = field
	var zkgt uint32
	zkgt, bts, err = msgp.ReadMapHeaderBytes(bts)
	if err != nil {
		return
	}
	for zkgt > 0 {
		zkgt--
		field, bts, err = msgp.ReadMapKeyZC(bts)
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "iv":
			var zema uint32
			zema, bts, err = msgp.ReadArrayHeaderBytes(bts)
			if err != nil {
				return
			}
			if cap(z.iv) >= int(zema) {
				z.iv = (z.iv)[:zema]
			} else {
				z.iv = make([]interval16, zema)
			}
			for zxpk := range z.iv {
				var zpez uint32
				zpez, bts, err = msgp.ReadMapHeaderBytes(bts)
				if err != nil {
					return
				}
				for zpez > 0 {
					zpez--
					field, bts, err = msgp.ReadMapKeyZC(bts)
					if err != nil {
						return
					}
					switch msgp.UnsafeString(field) {
					case "start":
						z.iv[zxpk].start, bts, err = msgp.ReadUint16Bytes(bts)
						if err != nil {
							return
						}
					case "last":
						z.iv[zxpk].length, bts, err = msgp.ReadUint16Bytes(bts)
						z.iv[zxpk].length -= z.iv[zxpk].start
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
		case "card":
			z.card, bts, err = msgp.ReadInt64Bytes(bts)
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
func (z *runContainer16) Msgsize() (s int) {
	s = 1 + 3 + msgp.ArrayHeaderSize + (len(z.iv) * (12 + msgp.Uint16Size + msgp.Uint16Size)) + 5 + msgp.Int64Size
	return
}

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *runIterator16) DecodeMsg(dc *msgp.Reader) (err error) {
	var field []byte
	_ = field
	var zqke uint32
	zqke, err = dc.ReadMapHeader()
	if err != nil {
		return
	}
	for zqke > 0 {
		zqke--
		field, err = dc.ReadMapKeyPtr()
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "rc":
			if dc.IsNil() {
				err = dc.ReadNil()
				if err != nil {
					return
				}
				z.rc = nil
			} else {
				if z.rc == nil {
					z.rc = new(runContainer16)
				}
				err = z.rc.DecodeMsg(dc)
				if err != nil {
					return
				}
			}
		case "curIndex":
			z.curIndex, err = dc.ReadInt64()
			if err != nil {
				return
			}
		case "curPosInIndex":
			z.curPosInIndex, err = dc.ReadUint16()
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
func (z *runIterator16) EncodeMsg(en *msgp.Writer) (err error) {
	// map header, size 3
	// write "rc"
	err = en.Append(0x83, 0xa2, 0x72, 0x63)
	if err != nil {
		return err
	}
	if z.rc == nil {
		err = en.WriteNil()
		if err != nil {
			return
		}
	} else {
		err = z.rc.EncodeMsg(en)
		if err != nil {
			return
		}
	}
	// write "curIndex"
	err = en.Append(0xa8, 0x63, 0x75, 0x72, 0x49, 0x6e, 0x64, 0x65, 0x78)
	if err != nil {
		return err
	}
	err = en.WriteInt64(z.curIndex)
	if err != nil {
		return
	}
	// write "curPosInIndex"
	err = en.Append(0xad, 0x63, 0x75, 0x72, 0x50, 0x6f, 0x73, 0x49, 0x6e, 0x49, 0x6e, 0x64, 0x65, 0x78)
	if err != nil {
		return err
	}
	err = en.WriteUint16(z.curPosInIndex)
	if err != nil {
		return
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z *runIterator16) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	// map header, size 3
	// string "rc"
	o = append(o, 0x83, 0xa2, 0x72, 0x63)
	if z.rc == nil {
		o = msgp.AppendNil(o)
	} else {
		o, err = z.rc.MarshalMsg(o)
		if err != nil {
			return
		}
	}
	// string "curIndex"
	o = append(o, 0xa8, 0x63, 0x75, 0x72, 0x49, 0x6e, 0x64, 0x65, 0x78)
	o = msgp.AppendInt64(o, z.curIndex)
	// string "curPosInIndex"
	o = append(o, 0xad, 0x63, 0x75, 0x72, 0x50, 0x6f, 0x73, 0x49, 0x6e, 0x49, 0x6e, 0x64, 0x65, 0x78)
	o = msgp.AppendUint16(o, z.curPosInIndex)
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *runIterator16) UnmarshalMsg(bts []byte) (o []byte, err error) {
	var field []byte
	_ = field
	var zqyh uint32
	zqyh, bts, err = msgp.ReadMapHeaderBytes(bts)
	if err != nil {
		return
	}
	for zqyh > 0 {
		zqyh--
		field, bts, err = msgp.ReadMapKeyZC(bts)
		if err != nil {
			return
		}
		switch msgp.UnsafeString(field) {
		case "rc":
			if msgp.IsNil(bts) {
				bts, err = msgp.ReadNilBytes(bts)
				if err != nil {
					return
				}
				z.rc = nil
			} else {
				if z.rc == nil {
					z.rc = new(runContainer16)
				}
				bts, err = z.rc.UnmarshalMsg(bts)
				if err != nil {
					return
				}
			}
		case "curIndex":
			z.curIndex, bts, err = msgp.ReadInt64Bytes(bts)
			if err != nil {
				return
			}
		case "curPosInIndex":
			z.curPosInIndex, bts, err = msgp.ReadUint16Bytes(bts)
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
func (z *runIterator16) Msgsize() (s int) {
	s = 1 + 3
	if z.rc == nil {
		s += msgp.NilSize
	} else {
		s += z.rc.Msgsize()
	}
	s += 9 + msgp.Int64Size + 14 + msgp.Uint16Size
	return
}

// Deprecated: DecodeMsg implements msgp.Decodable
func (z *uint16Slice) DecodeMsg(dc *msgp.Reader) (err error) {
	var zjpj uint32
	zjpj, err = dc.ReadArrayHeader()
	if err != nil {
		return
	}
	if cap((*z)) >= int(zjpj) {
		(*z) = (*z)[:zjpj]
	} else {
		(*z) = make(uint16Slice, zjpj)
	}
	for zywj := range *z {
		(*z)[zywj], err = dc.ReadUint16()
		if err != nil {
			return
		}
	}
	return
}

// Deprecated: EncodeMsg implements msgp.Encodable
func (z uint16Slice) EncodeMsg(en *msgp.Writer) (err error) {
	err = en.WriteArrayHeader(uint32(len(z)))
	if err != nil {
		return
	}
	for zzpf := range z {
		err = en.WriteUint16(z[zzpf])
		if err != nil {
			return
		}
	}
	return
}

// Deprecated: MarshalMsg implements msgp.Marshaler
func (z uint16Slice) MarshalMsg(b []byte) (o []byte, err error) {
	o = msgp.Require(b, z.Msgsize())
	o = msgp.AppendArrayHeader(o, uint32(len(z)))
	for zzpf := range z {
		o = msgp.AppendUint16(o, z[zzpf])
	}
	return
}

// Deprecated: UnmarshalMsg implements msgp.Unmarshaler
func (z *uint16Slice) UnmarshalMsg(bts []byte) (o []byte, err error) {
	var zgmo uint32
	zgmo, bts, err = msgp.ReadArrayHeaderBytes(bts)
	if err != nil {
		return
	}
	if cap((*z)) >= int(zgmo) {
		(*z) = (*z)[:zgmo]
	} else {
		(*z) = make(uint16Slice, zgmo)
	}
	for zrfe := range *z {
		(*z)[zrfe], bts, err = msgp.ReadUint16Bytes(bts)
		if err != nil {
			return
		}
	}
	o = bts
	return
}

// Deprecated: Msgsize returns an upper bound estimate of the number of bytes occupied by the serialized message
func (z uint16Slice) Msgsize() (s int) {
	s = msgp.ArrayHeaderSize + (len(z) * (msgp.Uint16Size))
	return
}
