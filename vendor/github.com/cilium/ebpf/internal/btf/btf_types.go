package btf

import (
	"encoding/binary"
	"io"

	"github.com/pkg/errors"
)

// btfKind describes a Type.
type btfKind uint8

// Equivalents of the BTF_KIND_* constants.
const (
	kindUnknown btfKind = iota
	kindInt
	kindPointer
	kindArray
	kindStruct
	kindUnion
	kindEnum
	kindForward
	kindTypedef
	kindVolatile
	kindConst
	kindRestrict
	// Added ~4.20
	kindFunc
	kindFuncProto
	// Added ~5.1
	kindVar
	kindDatasec
)

const (
	btfTypeKindShift = 24
	btfTypeKindLen   = 4
	btfTypeVlenShift = 0
	btfTypeVlenMask  = 16
)

// btfType is equivalent to struct btf_type in Documentation/bpf/btf.rst.
type btfType struct {
	NameOff uint32
	/* "info" bits arrangement
	 * bits  0-15: vlen (e.g. # of struct's members)
	 * bits 16-23: unused
	 * bits 24-27: kind (e.g. int, ptr, array...etc)
	 * bits 28-30: unused
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

func (bt *btfType) info(len, shift uint32) uint32 {
	return (bt.Info >> shift) & mask(len)
}

func (bt *btfType) setInfo(value, len, shift uint32) {
	bt.Info &^= mask(len) << shift
	bt.Info |= (value & mask(len)) << shift
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

func (bt *btfType) Type() TypeID {
	// TODO: Panic here if wrong kind?
	return TypeID(bt.SizeType)
}

func (bt *btfType) Size() uint32 {
	// TODO: Panic here if wrong kind?
	return bt.SizeType
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

func readTypes(r io.Reader, bo binary.ByteOrder) ([]rawType, error) {
	var (
		header btfType
		types  []rawType
	)

	for id := TypeID(1); ; id++ {
		if err := binary.Read(r, bo, &header); err == io.EOF {
			return types, nil
		} else if err != nil {
			return nil, errors.Wrapf(err, "can't read type info for id %v", id)
		}

		var data interface{}
		switch header.Kind() {
		case kindInt:
			// sizeof(uint32)
			data = make([]byte, 4)
		case kindPointer:
		case kindArray:
			data = new(btfArray)
		case kindStruct:
			fallthrough
		case kindUnion:
			data = make([]btfMember, header.Vlen())
		case kindEnum:
			// sizeof(struct btf_enum)
			data = make([]byte, header.Vlen()*4*2)
		case kindForward:
		case kindTypedef:
		case kindVolatile:
		case kindConst:
		case kindRestrict:
		case kindFunc:
		case kindFuncProto:
			// sizeof(struct btf_param)
			data = make([]byte, header.Vlen()*4*2)
		case kindVar:
			// sizeof(struct btf_variable)
			data = make([]byte, 4)
		case kindDatasec:
			// sizeof(struct btf_var_secinfo)
			data = make([]byte, header.Vlen()*4*3)
		default:
			return nil, errors.Errorf("type id %v: unknown kind: %v", id, header.Kind())
		}

		if data == nil {
			types = append(types, rawType{header, nil})
			continue
		}

		if err := binary.Read(r, bo, data); err != nil {
			return nil, errors.Wrapf(err, "type id %d: kind %v: can't read %T", id, header.Kind(), data)
		}

		types = append(types, rawType{header, data})
	}
}
