package nlgo

import (
	"bytes"
	"encoding/binary"
	"syscall"
	"unsafe"
)

/*
   NLA_U8            U8
   NLA_U16           U16
   NLA_U32           U32
   NLA_U64           U64
   NLA_STRING        NulString
   NLA_FLAG          Flag
   NLA_MSECS         U64
   NLA_NESTED        AttrList or Attr
   NLA_NESTED_COMPAT AttrList or Attr
   NLA_NUL_STRING    NulString
   NLA_BINARY        Binary
   NLA_S8            S8
   NLA_S16           S16
   NLA_S32           S32
   NLA_S64           S64
*/

type U8 uint8

func (self U8) Build(hdr syscall.NlAttr) []byte {
	length := NLA_HDRLEN + 1
	hdr.Len = uint16(length)
	ret := make([]byte, NLA_ALIGN(length))
	copy(ret, (*[syscall.SizeofNlAttr]byte)(unsafe.Pointer(&hdr))[:])
	ret[NLA_HDRLEN] = uint8(self)
	return ret
}

func setU8(attr *Attr) error {
	if attr.Header.Len != uint16(NLA_HDRLEN+1) {
		return NLE_RANGE
	}
	attr.Value = U8(attr.Bytes()[NLA_HDRLEN])
	return nil
}

type U16 uint16

func (self U16) Build(hdr syscall.NlAttr) []byte {
	length := NLA_HDRLEN + 2
	hdr.Len = uint16(length)
	ret := make([]byte, NLA_ALIGN(length))
	copy(ret, (*[syscall.SizeofNlAttr]byte)(unsafe.Pointer(&hdr))[:])
	if hdr.Type&syscall.NLA_F_NET_BYTEORDER == 0 {
		*(*uint16)(unsafe.Pointer(&ret[NLA_HDRLEN])) = uint16(self)
	} else {
		binary.BigEndian.PutUint16(ret[NLA_HDRLEN:], uint16(self))
	}
	return ret
}

func setU16(attr *Attr) error {
	if attr.Header.Len != uint16(NLA_HDRLEN+2) {
		return NLE_RANGE
	}
	nla := attr.Bytes()
	if attr.Header.Type&syscall.NLA_F_NET_BYTEORDER == 0 {
		attr.Value = U16(*(*uint16)(unsafe.Pointer(&nla[NLA_HDRLEN])))
	} else {
		attr.Value = U16(binary.BigEndian.Uint16(nla[NLA_HDRLEN:]))
	}
	return nil
}

type U32 uint32

func (self U32) Build(hdr syscall.NlAttr) []byte {
	length := NLA_HDRLEN + 4
	hdr.Len = uint16(length)
	ret := make([]byte, NLA_ALIGN(length))
	copy(ret, (*[syscall.SizeofNlAttr]byte)(unsafe.Pointer(&hdr))[:])
	if hdr.Type&syscall.NLA_F_NET_BYTEORDER == 0 {
		*(*uint32)(unsafe.Pointer(&ret[NLA_HDRLEN])) = uint32(self)
	} else {
		binary.BigEndian.PutUint32(ret[NLA_HDRLEN:], uint32(self))
	}
	return ret
}

func setU32(attr *Attr) error {
	if attr.Header.Len != uint16(NLA_HDRLEN+4) {
		return NLE_RANGE
	}
	nla := attr.Bytes()
	if attr.Header.Type&syscall.NLA_F_NET_BYTEORDER == 0 {
		attr.Value = U32(*(*uint32)(unsafe.Pointer(&nla[NLA_HDRLEN])))
	} else {
		attr.Value = U32(binary.BigEndian.Uint32(nla[NLA_HDRLEN:]))
	}
	return nil
}

type U64 uint64

func (self U64) Build(hdr syscall.NlAttr) []byte {
	length := NLA_HDRLEN + 8
	hdr.Len = uint16(length)
	ret := make([]byte, NLA_ALIGN(length))
	copy(ret, (*[syscall.SizeofNlAttr]byte)(unsafe.Pointer(&hdr))[:])
	if hdr.Type&syscall.NLA_F_NET_BYTEORDER == 0 {
		*(*uint64)(unsafe.Pointer(&ret[NLA_HDRLEN])) = uint64(self)
	} else {
		binary.BigEndian.PutUint64(ret[NLA_HDRLEN:], uint64(self))
	}
	return ret
}

func setU64(attr *Attr) error {
	if attr.Header.Len != uint16(NLA_HDRLEN+8) {
		return NLE_RANGE
	}
	nla := attr.Bytes()
	if attr.Header.Type&syscall.NLA_F_NET_BYTEORDER == 0 {
		attr.Value = U64(*(*uint32)(unsafe.Pointer(&nla[NLA_HDRLEN])))
	} else {
		attr.Value = U64(binary.BigEndian.Uint64(nla[NLA_HDRLEN:]))
	}
	return nil
}

type Binary []byte

func (self Binary) Build(hdr syscall.NlAttr) []byte {
	length := NLA_HDRLEN + len(self)
	hdr.Len = uint16(length)
	ret := make([]byte, NLA_ALIGN(length))
	copy(ret, (*[syscall.SizeofNlAttr]byte)(unsafe.Pointer(&hdr))[:])
	copy(ret[NLA_HDRLEN:], []byte(self))
	return ret
}

func setBinary(attr *Attr) error {
	nla := attr.Bytes()
	attr.Value = Binary(nla[NLA_HDRLEN:attr.Header.Len])
	return nil
}

type String string

func (self String) Build(hdr syscall.NlAttr) []byte {
	return Binary(self).Build(hdr)
}

func setString(attr *Attr) error {
	nla := attr.Bytes()
	attr.Value = String(nla[NLA_HDRLEN:attr.Header.Len])
	return nil
}

type NulString string

func (self NulString) Build(hdr syscall.NlAttr) []byte {
	length := NLA_HDRLEN + len(self) + 1
	hdr.Len = uint16(length)
	ret := make([]byte, NLA_ALIGN(length))
	copy(ret, (*[syscall.SizeofNlAttr]byte)(unsafe.Pointer(&hdr))[:])
	copy(ret[NLA_HDRLEN:], []byte(self))
	return ret
}

func setNulString(attr *Attr) error {
	nla := attr.Bytes()
	attr.Value = NulString(bytes.Split(nla[NLA_HDRLEN:attr.Header.Len], []byte{0})[0])
	return nil
}

type Flag bool

func (self Flag) Build(hdr syscall.NlAttr) []byte {
	if bool(self) {
		length := NLA_HDRLEN
		hdr.Len = uint16(length)
		ret := make([]byte, NLA_ALIGN(length))
		copy(ret, (*[syscall.SizeofNlAttr]byte)(unsafe.Pointer(&hdr))[:])
		return ret
	} else {
		return nil
	}
}

func setFlag(attr *Attr) error {
	if attr.Header.Len != uint16(NLA_HDRLEN) {
		return NLE_RANGE
	}
	attr.Value = Flag(true)
	return nil
}

type S8 int8

func (self S8) Build(hdr syscall.NlAttr) []byte {
	return U8(self).Build(hdr)
}

func setS8(attr *Attr) error {
	if err := setU8(attr); err != nil {
		return err
	}
	attr.Value = S8(attr.Value.(U8))
	return nil
}

type S16 int16

func (self S16) Build(hdr syscall.NlAttr) []byte {
	return S16(self).Build(hdr)
}

func setS16(attr *Attr) error {
	if err := setU16(attr); err != nil {
		return err
	}
	attr.Value = S16(attr.Value.(U16))
	return nil
}

type S32 int32

func (self S32) Build(hdr syscall.NlAttr) []byte {
	return S32(self).Build(hdr)
}

func setS32(attr *Attr) error {
	if err := setU32(attr); err != nil {
		return err
	}
	attr.Value = S32(attr.Value.(U32))
	return nil
}

type S64 int64

func (self S64) Build(hdr syscall.NlAttr) []byte {
	return S64(self).Build(hdr)
}

func setS64(attr *Attr) error {
	if err := setU64(attr); err != nil {
		return err
	}
	attr.Value = S64(attr.Value.(U64))
	return nil
}
