package btf

import (
	"encoding/binary"
	"fmt"
	"io"
)

//go:generate stringer -linecomment -output=btf_types_string.go -type=FuncLinkage,VarLinkage,btfKind

// btfKind describes a Type.
type btfKind uint8

// Equivalents of the BTF_KIND_* constants.
const (
	kindUnknown  btfKind = iota // Unknown
	kindInt                     // Int
	kindPointer                 // Pointer
	kindArray                   // Array
	kindStruct                  // Struct
	kindUnion                   // Union
	kindEnum                    // Enum
	kindForward                 // Forward
	kindTypedef                 // Typedef
	kindVolatile                // Volatile
	kindConst                   // Const
	kindRestrict                // Restrict
	// Added ~4.20
	kindFunc      // Func
	kindFuncProto // FuncProto
	// Added ~5.1
	kindVar     // Var
	kindDatasec // Datasec
	// Added ~5.13
	kindFloat // Float
	// Added 5.16
	kindDeclTag // DeclTag
	kindTypeTag // TypeTag
	// Added 6.0
	kindEnum64 // Enum64
)

// FuncLinkage describes BTF function linkage metadata.
type FuncLinkage int

// Equivalent of enum btf_func_linkage.
const (
	StaticFunc FuncLinkage = iota // static
	GlobalFunc                    // global
	ExternFunc                    // extern
)

// VarLinkage describes BTF variable linkage metadata.
type VarLinkage int

const (
	StaticVar VarLinkage = iota // static
	GlobalVar                   // global
	ExternVar                   // extern
)

const (
	btfTypeKindShift     = 24
	btfTypeKindLen       = 5
	btfTypeVlenShift     = 0
	btfTypeVlenMask      = 16
	btfTypeKindFlagShift = 31
	btfTypeKindFlagMask  = 1
)

var btfTypeLen = binary.Size(btfType{})

// btfType is equivalent to struct btf_type in Documentation/bpf/btf.rst.
type btfType struct {
	NameOff uint32
	/* "info" bits arrangement
	 * bits  0-15: vlen (e.g. # of struct's members), linkage
	 * bits 16-23: unused
	 * bits 24-28: kind (e.g. int, ptr, array...etc)
	 * bits 29-30: unused
	 * bit     31: kind_flag, currently used by
	 *             struct, union and fwd
	 */
	Info uint32
	/* "size" is used by INT, ENUM, STRUCT and UNION.
	 * "size" tells the size of the type it is describing.
	 *
	 * "type" is used by PTR, TYPEDEF, VOLATILE, CONST, RESTRICT,
	 * FUNC and FUNC_PROTO.
	 * "type" is a type_id referring to another type.
	 */
	SizeType uint32
}

func mask(len uint32) uint32 {
	return (1 << len) - 1
}

func readBits(value, len, shift uint32) uint32 {
	return (value >> shift) & mask(len)
}

func writeBits(value, len, shift, new uint32) uint32 {
	value &^= mask(len) << shift
	value |= (new & mask(len)) << shift
	return value
}

func (bt *btfType) info(len, shift uint32) uint32 {
	return readBits(bt.Info, len, shift)
}

func (bt *btfType) setInfo(value, len, shift uint32) {
	bt.Info = writeBits(bt.Info, len, shift, value)
}

func (bt *btfType) Kind() btfKind {
	return btfKind(bt.info(btfTypeKindLen, btfTypeKindShift))
}

func (bt *btfType) SetKind(kind btfKind) {
	bt.setInfo(uint32(kind), btfTypeKindLen, btfTypeKindShift)
}

func (bt *btfType) Vlen() int {
	return int(bt.info(btfTypeVlenMask, btfTypeVlenShift))
}

func (bt *btfType) SetVlen(vlen int) {
	bt.setInfo(uint32(vlen), btfTypeVlenMask, btfTypeVlenShift)
}

func (bt *btfType) kindFlagBool() bool {
	return bt.info(btfTypeKindFlagMask, btfTypeKindFlagShift) == 1
}

func (bt *btfType) setKindFlagBool(set bool) {
	var value uint32
	if set {
		value = 1
	}
	bt.setInfo(value, btfTypeKindFlagMask, btfTypeKindFlagShift)
}

// Bitfield returns true if the struct or union contain a bitfield.
func (bt *btfType) Bitfield() bool {
	return bt.kindFlagBool()
}

func (bt *btfType) SetBitfield(isBitfield bool) {
	bt.setKindFlagBool(isBitfield)
}

func (bt *btfType) FwdKind() FwdKind {
	return FwdKind(bt.info(btfTypeKindFlagMask, btfTypeKindFlagShift))
}

func (bt *btfType) SetFwdKind(kind FwdKind) {
	bt.setInfo(uint32(kind), btfTypeKindFlagMask, btfTypeKindFlagShift)
}

func (bt *btfType) Signed() bool {
	return bt.kindFlagBool()
}

func (bt *btfType) SetSigned(signed bool) {
	bt.setKindFlagBool(signed)
}

func (bt *btfType) Linkage() FuncLinkage {
	return FuncLinkage(bt.info(btfTypeVlenMask, btfTypeVlenShift))
}

func (bt *btfType) SetLinkage(linkage FuncLinkage) {
	bt.setInfo(uint32(linkage), btfTypeVlenMask, btfTypeVlenShift)
}

func (bt *btfType) Type() TypeID {
	// TODO: Panic here if wrong kind?
	return TypeID(bt.SizeType)
}

func (bt *btfType) SetType(id TypeID) {
	bt.SizeType = uint32(id)
}

func (bt *btfType) Size() uint32 {
	// TODO: Panic here if wrong kind?
	return bt.SizeType
}

func (bt *btfType) SetSize(size uint32) {
	bt.SizeType = size
}

type rawType struct {
	btfType
	data interface{}
}

func (rt *rawType) Marshal(w io.Writer, bo binary.ByteOrder) error {
	if err := binary.Write(w, bo, &rt.btfType); err != nil {
		return err
	}

	if rt.data == nil {
		return nil
	}

	return binary.Write(w, bo, rt.data)
}

// btfInt encodes additional data for integers.
//
//	? ? ? ? e e e e o o o o o o o o ? ? ? ? ? ? ? ? b b b b b b b b
//	? = undefined
//	e = encoding
//	o = offset (bitfields?)
//	b = bits (bitfields)
type btfInt struct {
	Raw uint32
}

const (
	btfIntEncodingLen   = 4
	btfIntEncodingShift = 24
	btfIntOffsetLen     = 8
	btfIntOffsetShift   = 16
	btfIntBitsLen       = 8
	btfIntBitsShift     = 0
)

func (bi btfInt) Encoding() IntEncoding {
	return IntEncoding(readBits(bi.Raw, btfIntEncodingLen, btfIntEncodingShift))
}

func (bi *btfInt) SetEncoding(e IntEncoding) {
	bi.Raw = writeBits(uint32(bi.Raw), btfIntEncodingLen, btfIntEncodingShift, uint32(e))
}

func (bi btfInt) Offset() Bits {
	return Bits(readBits(bi.Raw, btfIntOffsetLen, btfIntOffsetShift))
}

func (bi *btfInt) SetOffset(offset uint32) {
	bi.Raw = writeBits(bi.Raw, btfIntOffsetLen, btfIntOffsetShift, offset)
}

func (bi btfInt) Bits() Bits {
	return Bits(readBits(bi.Raw, btfIntBitsLen, btfIntBitsShift))
}

func (bi *btfInt) SetBits(bits byte) {
	bi.Raw = writeBits(bi.Raw, btfIntBitsLen, btfIntBitsShift, uint32(bits))
}

type btfArray struct {
	Type      TypeID
	IndexType TypeID
	Nelems    uint32
}

type btfMember struct {
	NameOff uint32
	Type    TypeID
	Offset  uint32
}

type btfVarSecinfo struct {
	Type   TypeID
	Offset uint32
	Size   uint32
}

type btfVariable struct {
	Linkage uint32
}

type btfEnum struct {
	NameOff uint32
	Val     uint32
}

type btfEnum64 struct {
	NameOff uint32
	ValLo32 uint32
	ValHi32 uint32
}

type btfParam struct {
	NameOff uint32
	Type    TypeID
}

type btfDeclTag struct {
	ComponentIdx uint32
}

func readTypes(r io.Reader, bo binary.ByteOrder, typeLen uint32) ([]rawType, error) {
	var header btfType
	// because of the interleaving between types and struct members it is difficult to
	// precompute the numbers of raw types this will parse
	// this "guess" is a good first estimation
	sizeOfbtfType := uintptr(btfTypeLen)
	tyMaxCount := uintptr(typeLen) / sizeOfbtfType / 2
	types := make([]rawType, 0, tyMaxCount)

	for id := TypeID(1); ; id++ {
		if err := binary.Read(r, bo, &header); err == io.EOF {
			return types, nil
		} else if err != nil {
			return nil, fmt.Errorf("can't read type info for id %v: %v", id, err)
		}

		var data interface{}
		switch header.Kind() {
		case kindInt:
			data = new(btfInt)
		case kindPointer:
		case kindArray:
			data = new(btfArray)
		case kindStruct:
			fallthrough
		case kindUnion:
			data = make([]btfMember, header.Vlen())
		case kindEnum:
			data = make([]btfEnum, header.Vlen())
		case kindForward:
		case kindTypedef:
		case kindVolatile:
		case kindConst:
		case kindRestrict:
		case kindFunc:
		case kindFuncProto:
			data = make([]btfParam, header.Vlen())
		case kindVar:
			data = new(btfVariable)
		case kindDatasec:
			data = make([]btfVarSecinfo, header.Vlen())
		case kindFloat:
		case kindDeclTag:
			data = new(btfDeclTag)
		case kindTypeTag:
		case kindEnum64:
			data = make([]btfEnum64, header.Vlen())
		default:
			return nil, fmt.Errorf("type id %v: unknown kind: %v", id, header.Kind())
		}

		if data == nil {
			types = append(types, rawType{header, nil})
			continue
		}

		if err := binary.Read(r, bo, data); err != nil {
			return nil, fmt.Errorf("type id %d: kind %v: can't read %T: %v", id, header.Kind(), data, err)
		}

		types = append(types, rawType{header, data})
	}
}
