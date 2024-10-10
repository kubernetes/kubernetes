package btf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"unsafe"

	"github.com/cilium/ebpf/internal"
)

//go:generate go run golang.org/x/tools/cmd/stringer@latest -linecomment -output=btf_types_string.go -type=FuncLinkage,VarLinkage,btfKind

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

var btfHeaderLen = binary.Size(&btfHeader{})

type btfHeader struct {
	Magic   uint16
	Version uint8
	Flags   uint8
	HdrLen  uint32

	TypeOff   uint32
	TypeLen   uint32
	StringOff uint32
	StringLen uint32
}

// typeStart returns the offset from the beginning of the .BTF section
// to the start of its type entries.
func (h *btfHeader) typeStart() int64 {
	return int64(h.HdrLen + h.TypeOff)
}

// stringStart returns the offset from the beginning of the .BTF section
// to the start of its string table.
func (h *btfHeader) stringStart() int64 {
	return int64(h.HdrLen + h.StringOff)
}

// parseBTFHeader parses the header of the .BTF section.
func parseBTFHeader(r io.Reader, bo binary.ByteOrder) (*btfHeader, error) {
	var header btfHeader
	if err := binary.Read(r, bo, &header); err != nil {
		return nil, fmt.Errorf("can't read header: %v", err)
	}

	if header.Magic != btfMagic {
		return nil, fmt.Errorf("incorrect magic value %v", header.Magic)
	}

	if header.Version != 1 {
		return nil, fmt.Errorf("unexpected version %v", header.Version)
	}

	if header.Flags != 0 {
		return nil, fmt.Errorf("unsupported flags %v", header.Flags)
	}

	remainder := int64(header.HdrLen) - int64(binary.Size(&header))
	if remainder < 0 {
		return nil, errors.New("header length shorter than btfHeader size")
	}

	if _, err := io.CopyN(internal.DiscardZeroes{}, r, remainder); err != nil {
		return nil, fmt.Errorf("header padding: %v", err)
	}

	return &header, nil
}

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

var btfTypeSize = int(unsafe.Sizeof(btfType{}))

func unmarshalBtfType(bt *btfType, b []byte, bo binary.ByteOrder) (int, error) {
	if len(b) < btfTypeSize {
		return 0, fmt.Errorf("not enough bytes to unmarshal btfType")
	}

	bt.NameOff = bo.Uint32(b[0:])
	bt.Info = bo.Uint32(b[4:])
	bt.SizeType = bo.Uint32(b[8:])
	return btfTypeSize, nil
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

func (bt *btfType) Marshal(w io.Writer, bo binary.ByteOrder) error {
	buf := make([]byte, unsafe.Sizeof(*bt))
	bo.PutUint32(buf[0:], bt.NameOff)
	bo.PutUint32(buf[4:], bt.Info)
	bo.PutUint32(buf[8:], bt.SizeType)
	_, err := w.Write(buf)
	return err
}

type rawType struct {
	btfType
	data interface{}
}

func (rt *rawType) Marshal(w io.Writer, bo binary.ByteOrder) error {
	if err := rt.btfType.Marshal(w, bo); err != nil {
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

var btfIntLen = int(unsafe.Sizeof(btfInt{}))

func unmarshalBtfInt(bi *btfInt, b []byte, bo binary.ByteOrder) (int, error) {
	if len(b) < btfIntLen {
		return 0, fmt.Errorf("not enough bytes to unmarshal btfInt")
	}

	bi.Raw = bo.Uint32(b[0:])
	return btfIntLen, nil
}

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

var btfArrayLen = int(unsafe.Sizeof(btfArray{}))

func unmarshalBtfArray(ba *btfArray, b []byte, bo binary.ByteOrder) (int, error) {
	if len(b) < btfArrayLen {
		return 0, fmt.Errorf("not enough bytes to unmarshal btfArray")
	}

	ba.Type = TypeID(bo.Uint32(b[0:]))
	ba.IndexType = TypeID(bo.Uint32(b[4:]))
	ba.Nelems = bo.Uint32(b[8:])
	return btfArrayLen, nil
}

type btfMember struct {
	NameOff uint32
	Type    TypeID
	Offset  uint32
}

var btfMemberLen = int(unsafe.Sizeof(btfMember{}))

func unmarshalBtfMembers(members []btfMember, b []byte, bo binary.ByteOrder) (int, error) {
	off := 0
	for i := range members {
		if off+btfMemberLen > len(b) {
			return 0, fmt.Errorf("not enough bytes to unmarshal btfMember %d", i)
		}

		members[i].NameOff = bo.Uint32(b[off+0:])
		members[i].Type = TypeID(bo.Uint32(b[off+4:]))
		members[i].Offset = bo.Uint32(b[off+8:])

		off += btfMemberLen
	}

	return off, nil
}

type btfVarSecinfo struct {
	Type   TypeID
	Offset uint32
	Size   uint32
}

var btfVarSecinfoLen = int(unsafe.Sizeof(btfVarSecinfo{}))

func unmarshalBtfVarSecInfos(secinfos []btfVarSecinfo, b []byte, bo binary.ByteOrder) (int, error) {
	off := 0
	for i := range secinfos {
		if off+btfVarSecinfoLen > len(b) {
			return 0, fmt.Errorf("not enough bytes to unmarshal btfVarSecinfo %d", i)
		}

		secinfos[i].Type = TypeID(bo.Uint32(b[off+0:]))
		secinfos[i].Offset = bo.Uint32(b[off+4:])
		secinfos[i].Size = bo.Uint32(b[off+8:])

		off += btfVarSecinfoLen
	}

	return off, nil
}

type btfVariable struct {
	Linkage uint32
}

var btfVariableLen = int(unsafe.Sizeof(btfVariable{}))

func unmarshalBtfVariable(bv *btfVariable, b []byte, bo binary.ByteOrder) (int, error) {
	if len(b) < btfVariableLen {
		return 0, fmt.Errorf("not enough bytes to unmarshal btfVariable")
	}

	bv.Linkage = bo.Uint32(b[0:])
	return btfVariableLen, nil
}

type btfEnum struct {
	NameOff uint32
	Val     uint32
}

var btfEnumLen = int(unsafe.Sizeof(btfEnum{}))

func unmarshalBtfEnums(enums []btfEnum, b []byte, bo binary.ByteOrder) (int, error) {
	off := 0
	for i := range enums {
		if off+btfEnumLen > len(b) {
			return 0, fmt.Errorf("not enough bytes to unmarshal btfEnum %d", i)
		}

		enums[i].NameOff = bo.Uint32(b[off+0:])
		enums[i].Val = bo.Uint32(b[off+4:])

		off += btfEnumLen
	}

	return off, nil
}

type btfEnum64 struct {
	NameOff uint32
	ValLo32 uint32
	ValHi32 uint32
}

var btfEnum64Len = int(unsafe.Sizeof(btfEnum64{}))

func unmarshalBtfEnums64(enums []btfEnum64, b []byte, bo binary.ByteOrder) (int, error) {
	off := 0
	for i := range enums {
		if off+btfEnum64Len > len(b) {
			return 0, fmt.Errorf("not enough bytes to unmarshal btfEnum64 %d", i)
		}

		enums[i].NameOff = bo.Uint32(b[off+0:])
		enums[i].ValLo32 = bo.Uint32(b[off+4:])
		enums[i].ValHi32 = bo.Uint32(b[off+8:])

		off += btfEnum64Len
	}

	return off, nil
}

type btfParam struct {
	NameOff uint32
	Type    TypeID
}

var btfParamLen = int(unsafe.Sizeof(btfParam{}))

func unmarshalBtfParams(params []btfParam, b []byte, bo binary.ByteOrder) (int, error) {
	off := 0
	for i := range params {
		if off+btfParamLen > len(b) {
			return 0, fmt.Errorf("not enough bytes to unmarshal btfParam %d", i)
		}

		params[i].NameOff = bo.Uint32(b[off+0:])
		params[i].Type = TypeID(bo.Uint32(b[off+4:]))

		off += btfParamLen
	}

	return off, nil
}

type btfDeclTag struct {
	ComponentIdx uint32
}

var btfDeclTagLen = int(unsafe.Sizeof(btfDeclTag{}))

func unmarshalBtfDeclTag(bdt *btfDeclTag, b []byte, bo binary.ByteOrder) (int, error) {
	if len(b) < btfDeclTagLen {
		return 0, fmt.Errorf("not enough bytes to unmarshal btfDeclTag")
	}

	bdt.ComponentIdx = bo.Uint32(b[0:])
	return btfDeclTagLen, nil
}
