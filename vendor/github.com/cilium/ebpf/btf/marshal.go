package btf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/cilium/ebpf/internal"
)

type encoderOptions struct {
	ByteOrder binary.ByteOrder
	// Remove function linkage information for compatibility with <5.6 kernels.
	StripFuncLinkage bool
}

// kernelEncoderOptions will generate BTF suitable for the current kernel.
var kernelEncoderOptions encoderOptions

func init() {
	kernelEncoderOptions = encoderOptions{
		ByteOrder:        internal.NativeEndian,
		StripFuncLinkage: haveFuncLinkage() != nil,
	}
}

// encoder turns Types into raw BTF.
type encoder struct {
	opts encoderOptions

	buf          *bytes.Buffer
	strings      *stringTableBuilder
	allocatedIDs map[Type]TypeID
	nextID       TypeID
	// Temporary storage for Add.
	pending internal.Deque[Type]
	// Temporary storage for deflateType.
	raw rawType
}

// newEncoder returns a new builder for the given byte order.
//
// See [KernelEncoderOptions] to build BTF for the current system.
func newEncoder(opts encoderOptions, strings *stringTableBuilder) *encoder {
	enc := &encoder{
		opts: opts,
		buf:  bytes.NewBuffer(make([]byte, btfHeaderLen)),
	}
	enc.reset(strings)
	return enc
}

// Reset internal state to be able to reuse the Encoder.
func (e *encoder) Reset() {
	e.reset(nil)
}

func (e *encoder) reset(strings *stringTableBuilder) {
	if strings == nil {
		strings = newStringTableBuilder()
	}

	e.buf.Truncate(btfHeaderLen)
	e.strings = strings
	e.allocatedIDs = make(map[Type]TypeID)
	e.nextID = 1
}

// Add a Type.
//
// Adding the same Type multiple times is valid and will return a stable ID.
//
// Calling the method has undefined behaviour if it previously returned an error.
func (e *encoder) Add(typ Type) (TypeID, error) {
	if typ == nil {
		return 0, errors.New("cannot Add a nil Type")
	}

	hasID := func(t Type) (skip bool) {
		_, isVoid := t.(*Void)
		_, alreadyEncoded := e.allocatedIDs[t]
		return isVoid || alreadyEncoded
	}

	e.pending.Reset()

	allocateID := func(typ Type) {
		e.pending.Push(typ)
		e.allocatedIDs[typ] = e.nextID
		e.nextID++
	}

	iter := postorderTraversal(typ, hasID)
	for iter.Next() {
		if hasID(iter.Type) {
			// This type is part of a cycle and we've already deflated it.
			continue
		}

		// Allocate an ID for the next type.
		allocateID(iter.Type)

		for !e.pending.Empty() {
			t := e.pending.Shift()

			// Ensure that all direct descendants have been allocated an ID
			// before calling deflateType.
			walkType(t, func(child *Type) {
				if !hasID(*child) {
					// t refers to a type which hasn't been allocated an ID
					// yet, which only happens for circular types.
					allocateID(*child)
				}
			})

			if err := e.deflateType(t); err != nil {
				return 0, fmt.Errorf("deflate %s: %w", t, err)
			}
		}
	}

	return e.allocatedIDs[typ], nil
}

// Encode the raw BTF blob.
//
// The returned slice is valid until the next call to Add.
func (e *encoder) Encode() ([]byte, error) {
	length := e.buf.Len()

	// Truncate the string table on return to allow adding more types.
	defer e.buf.Truncate(length)

	typeLen := uint32(length - btfHeaderLen)

	// Reserve space for the string table.
	stringLen := e.strings.Length()
	e.buf.Grow(stringLen)

	buf := e.buf.Bytes()[:length+stringLen]
	e.strings.MarshalBuffer(buf[length:])

	// Fill out the header, and write it out.
	header := &btfHeader{
		Magic:     btfMagic,
		Version:   1,
		Flags:     0,
		HdrLen:    uint32(btfHeaderLen),
		TypeOff:   0,
		TypeLen:   typeLen,
		StringOff: typeLen,
		StringLen: uint32(stringLen),
	}

	err := binary.Write(sliceWriter(buf[:btfHeaderLen]), e.opts.ByteOrder, header)
	if err != nil {
		return nil, fmt.Errorf("can't write header: %v", err)
	}

	return buf, nil
}

func (e *encoder) deflateType(typ Type) (err error) {
	raw := &e.raw
	*raw = rawType{}
	raw.NameOff, err = e.strings.Add(typ.TypeName())
	if err != nil {
		return err
	}

	switch v := typ.(type) {
	case *Int:
		raw.SetKind(kindInt)
		raw.SetSize(v.Size)

		var bi btfInt
		bi.SetEncoding(v.Encoding)
		// We need to set bits in addition to size, since btf_type_int_is_regular
		// otherwise flags this as a bitfield.
		bi.SetBits(byte(v.Size) * 8)
		raw.data = bi

	case *Pointer:
		raw.SetKind(kindPointer)
		raw.SetType(e.allocatedIDs[v.Target])

	case *Array:
		raw.SetKind(kindArray)
		raw.data = &btfArray{
			e.allocatedIDs[v.Type],
			e.allocatedIDs[v.Index],
			v.Nelems,
		}

	case *Struct:
		raw.SetKind(kindStruct)
		raw.SetSize(v.Size)
		raw.data, err = e.convertMembers(&raw.btfType, v.Members)

	case *Union:
		raw.SetKind(kindUnion)
		raw.SetSize(v.Size)
		raw.data, err = e.convertMembers(&raw.btfType, v.Members)

	case *Enum:
		raw.SetSize(v.size())
		raw.SetVlen(len(v.Values))
		raw.SetSigned(v.Signed)

		if v.has64BitValues() {
			raw.SetKind(kindEnum64)
			raw.data, err = e.deflateEnum64Values(v.Values)
		} else {
			raw.SetKind(kindEnum)
			raw.data, err = e.deflateEnumValues(v.Values)
		}

	case *Fwd:
		raw.SetKind(kindForward)
		raw.SetFwdKind(v.Kind)

	case *Typedef:
		raw.SetKind(kindTypedef)
		raw.SetType(e.allocatedIDs[v.Type])

	case *Volatile:
		raw.SetKind(kindVolatile)
		raw.SetType(e.allocatedIDs[v.Type])

	case *Const:
		raw.SetKind(kindConst)
		raw.SetType(e.allocatedIDs[v.Type])

	case *Restrict:
		raw.SetKind(kindRestrict)
		raw.SetType(e.allocatedIDs[v.Type])

	case *Func:
		raw.SetKind(kindFunc)
		raw.SetType(e.allocatedIDs[v.Type])
		if !e.opts.StripFuncLinkage {
			raw.SetLinkage(v.Linkage)
		}

	case *FuncProto:
		raw.SetKind(kindFuncProto)
		raw.SetType(e.allocatedIDs[v.Return])
		raw.SetVlen(len(v.Params))
		raw.data, err = e.deflateFuncParams(v.Params)

	case *Var:
		raw.SetKind(kindVar)
		raw.SetType(e.allocatedIDs[v.Type])
		raw.data = btfVariable{uint32(v.Linkage)}

	case *Datasec:
		raw.SetKind(kindDatasec)
		raw.SetSize(v.Size)
		raw.SetVlen(len(v.Vars))
		raw.data = e.deflateVarSecinfos(v.Vars)

	case *Float:
		raw.SetKind(kindFloat)
		raw.SetSize(v.Size)

	case *declTag:
		raw.SetKind(kindDeclTag)
		raw.data = &btfDeclTag{uint32(v.Index)}

	case *typeTag:
		raw.SetKind(kindTypeTag)
		raw.NameOff, err = e.strings.Add(v.Value)

	default:
		return fmt.Errorf("don't know how to deflate %T", v)
	}

	if err != nil {
		return err
	}

	return raw.Marshal(e.buf, e.opts.ByteOrder)
}

func (e *encoder) convertMembers(header *btfType, members []Member) ([]btfMember, error) {
	bms := make([]btfMember, 0, len(members))
	isBitfield := false
	for _, member := range members {
		isBitfield = isBitfield || member.BitfieldSize > 0

		offset := member.Offset
		if isBitfield {
			offset = member.BitfieldSize<<24 | (member.Offset & 0xffffff)
		}

		nameOff, err := e.strings.Add(member.Name)
		if err != nil {
			return nil, err
		}

		bms = append(bms, btfMember{
			nameOff,
			e.allocatedIDs[member.Type],
			uint32(offset),
		})
	}

	header.SetVlen(len(members))
	header.SetBitfield(isBitfield)
	return bms, nil
}

func (e *encoder) deflateEnumValues(values []EnumValue) ([]btfEnum, error) {
	bes := make([]btfEnum, 0, len(values))
	for _, value := range values {
		nameOff, err := e.strings.Add(value.Name)
		if err != nil {
			return nil, err
		}

		if value.Value > math.MaxUint32 {
			return nil, fmt.Errorf("value of enum %q exceeds 32 bits", value.Name)
		}

		bes = append(bes, btfEnum{
			nameOff,
			uint32(value.Value),
		})
	}

	return bes, nil
}

func (e *encoder) deflateEnum64Values(values []EnumValue) ([]btfEnum64, error) {
	bes := make([]btfEnum64, 0, len(values))
	for _, value := range values {
		nameOff, err := e.strings.Add(value.Name)
		if err != nil {
			return nil, err
		}

		bes = append(bes, btfEnum64{
			nameOff,
			uint32(value.Value),
			uint32(value.Value >> 32),
		})
	}

	return bes, nil
}

func (e *encoder) deflateFuncParams(params []FuncParam) ([]btfParam, error) {
	bps := make([]btfParam, 0, len(params))
	for _, param := range params {
		nameOff, err := e.strings.Add(param.Name)
		if err != nil {
			return nil, err
		}

		bps = append(bps, btfParam{
			nameOff,
			e.allocatedIDs[param.Type],
		})
	}
	return bps, nil
}

func (e *encoder) deflateVarSecinfos(vars []VarSecinfo) []btfVarSecinfo {
	vsis := make([]btfVarSecinfo, 0, len(vars))
	for _, v := range vars {
		vsis = append(vsis, btfVarSecinfo{
			e.allocatedIDs[v.Type],
			v.Offset,
			v.Size,
		})
	}
	return vsis
}

// MarshalMapKV creates a BTF object containing a map key and value.
//
// The function is intended for the use of the ebpf package and may be removed
// at any point in time.
func MarshalMapKV(key, value Type) (_ *Handle, keyID, valueID TypeID, _ error) {
	enc := nativeEncoderPool.Get().(*encoder)
	defer nativeEncoderPool.Put(enc)

	enc.Reset()

	var err error
	if key != nil {
		keyID, err = enc.Add(key)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("adding map key to BTF encoder: %w", err)
		}
	}

	if value != nil {
		valueID, err = enc.Add(value)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("adding map value to BTF encoder: %w", err)
		}
	}

	btf, err := enc.Encode()
	if err != nil {
		return nil, 0, 0, fmt.Errorf("marshal BTF: %w", err)
	}

	handle, err := newHandleFromRawBTF(btf)
	if err != nil {
		// Check for 'full' map BTF support, since kernels between 4.18 and 5.2
		// already support BTF blobs for maps without Var or Datasec just fine.
		if err := haveMapBTF(); err != nil {
			return nil, 0, 0, err
		}
	}

	return handle, keyID, valueID, err
}
