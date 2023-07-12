package btf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/cilium/ebpf/internal"

	"golang.org/x/exp/slices"
)

type MarshalOptions struct {
	// Target byte order. Defaults to the system's native endianness.
	Order binary.ByteOrder
	// Remove function linkage information for compatibility with <5.6 kernels.
	StripFuncLinkage bool
}

// KernelMarshalOptions will generate BTF suitable for the current kernel.
func KernelMarshalOptions() *MarshalOptions {
	return &MarshalOptions{
		Order:            internal.NativeEndian,
		StripFuncLinkage: haveFuncLinkage() != nil,
	}
}

// encoder turns Types into raw BTF.
type encoder struct {
	MarshalOptions

	pending internal.Deque[Type]
	buf     *bytes.Buffer
	strings *stringTableBuilder
	ids     map[Type]TypeID
	lastID  TypeID
}

var bufferPool = sync.Pool{
	New: func() any {
		buf := make([]byte, btfHeaderLen+128)
		return &buf
	},
}

func getByteSlice() *[]byte {
	return bufferPool.Get().(*[]byte)
}

func putByteSlice(buf *[]byte) {
	*buf = (*buf)[:0]
	bufferPool.Put(buf)
}

// Builder turns Types into raw BTF.
//
// The default value may be used and represents an empty BTF blob. Void is
// added implicitly if necessary.
type Builder struct {
	// Explicitly added types.
	types []Type
	// IDs for all added types which the user knows about.
	stableIDs map[Type]TypeID
	// Explicitly added strings.
	strings *stringTableBuilder
}

// NewBuilder creates a Builder from a list of types.
//
// It is more efficient than calling [Add] individually.
//
// Returns an error if adding any of the types fails.
func NewBuilder(types []Type) (*Builder, error) {
	b := &Builder{
		make([]Type, 0, len(types)),
		make(map[Type]TypeID, len(types)),
		nil,
	}

	for _, typ := range types {
		_, err := b.Add(typ)
		if err != nil {
			return nil, fmt.Errorf("add %s: %w", typ, err)
		}
	}

	return b, nil
}

// Add a Type and allocate a stable ID for it.
//
// Adding the identical Type multiple times is valid and will return the same ID.
//
// See [Type] for details on identity.
func (b *Builder) Add(typ Type) (TypeID, error) {
	if b.stableIDs == nil {
		b.stableIDs = make(map[Type]TypeID)
	}

	if _, ok := typ.(*Void); ok {
		// Equality is weird for void, since it is a zero sized type.
		return 0, nil
	}

	if ds, ok := typ.(*Datasec); ok {
		if err := datasecResolveWorkaround(b, ds); err != nil {
			return 0, err
		}
	}

	id, ok := b.stableIDs[typ]
	if ok {
		return id, nil
	}

	b.types = append(b.types, typ)

	id = TypeID(len(b.types))
	if int(id) != len(b.types) {
		return 0, fmt.Errorf("no more type IDs")
	}

	b.stableIDs[typ] = id
	return id, nil
}

// Marshal encodes all types in the Marshaler into BTF wire format.
//
// opts may be nil.
func (b *Builder) Marshal(buf []byte, opts *MarshalOptions) ([]byte, error) {
	stb := b.strings
	if stb == nil {
		// Assume that most types are named. This makes encoding large BTF like
		// vmlinux a lot cheaper.
		stb = newStringTableBuilder(len(b.types))
	} else {
		// Avoid modifying the Builder's string table.
		stb = b.strings.Copy()
	}

	if opts == nil {
		opts = &MarshalOptions{Order: internal.NativeEndian}
	}

	// Reserve space for the BTF header.
	buf = slices.Grow(buf, btfHeaderLen)[:btfHeaderLen]

	w := internal.NewBuffer(buf)
	defer internal.PutBuffer(w)

	e := encoder{
		MarshalOptions: *opts,
		buf:            w,
		strings:        stb,
		lastID:         TypeID(len(b.types)),
		ids:            make(map[Type]TypeID, len(b.types)),
	}

	// Ensure that types are marshaled in the exact order they were Add()ed.
	// Otherwise the ID returned from Add() won't match.
	e.pending.Grow(len(b.types))
	for _, typ := range b.types {
		e.pending.Push(typ)
		e.ids[typ] = b.stableIDs[typ]
	}

	if err := e.deflatePending(); err != nil {
		return nil, err
	}

	length := e.buf.Len()
	typeLen := uint32(length - btfHeaderLen)

	stringLen := e.strings.Length()
	buf = e.strings.AppendEncoded(e.buf.Bytes())

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

	err := binary.Write(sliceWriter(buf[:btfHeaderLen]), e.Order, header)
	if err != nil {
		return nil, fmt.Errorf("write header: %v", err)
	}

	return buf, nil
}

// addString adds a string to the resulting BTF.
//
// Adding the same string multiple times will return the same result.
//
// Returns an identifier into the string table or an error if the string
// contains invalid characters.
func (b *Builder) addString(str string) (uint32, error) {
	if b.strings == nil {
		b.strings = newStringTableBuilder(0)
	}

	return b.strings.Add(str)
}

func (e *encoder) allocateID(typ Type) error {
	id := e.lastID + 1
	if id < e.lastID {
		return errors.New("type ID overflow")
	}

	e.pending.Push(typ)
	e.ids[typ] = id
	e.lastID = id
	return nil
}

// id returns the ID for the given type or panics with an error.
func (e *encoder) id(typ Type) TypeID {
	if _, ok := typ.(*Void); ok {
		return 0
	}

	id, ok := e.ids[typ]
	if !ok {
		panic(fmt.Errorf("no ID for type %v", typ))
	}

	return id
}

func (e *encoder) deflatePending() error {
	// Declare root outside of the loop to avoid repeated heap allocations.
	var root Type
	skip := func(t Type) (skip bool) {
		if t == root {
			// Force descending into the current root type even if it already
			// has an ID. Otherwise we miss children of types that have their
			// ID pre-allocated via Add.
			return false
		}

		_, isVoid := t.(*Void)
		_, alreadyEncoded := e.ids[t]
		return isVoid || alreadyEncoded
	}

	for !e.pending.Empty() {
		root = e.pending.Shift()

		// Allocate IDs for all children of typ, including transitive dependencies.
		iter := postorderTraversal(root, skip)
		for iter.Next() {
			if iter.Type == root {
				// The iterator yields root at the end, do not allocate another ID.
				break
			}

			if err := e.allocateID(iter.Type); err != nil {
				return err
			}
		}

		if err := e.deflateType(root); err != nil {
			id := e.ids[root]
			return fmt.Errorf("deflate %v with ID %d: %w", root, id, err)
		}
	}

	return nil
}

func (e *encoder) deflateType(typ Type) (err error) {
	defer func() {
		if r := recover(); r != nil {
			var ok bool
			err, ok = r.(error)
			if !ok {
				panic(r)
			}
		}
	}()

	var raw rawType
	raw.NameOff, err = e.strings.Add(typ.TypeName())
	if err != nil {
		return err
	}

	switch v := typ.(type) {
	case *Void:
		return errors.New("Void is implicit in BTF wire format")

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
		raw.SetType(e.id(v.Target))

	case *Array:
		raw.SetKind(kindArray)
		raw.data = &btfArray{
			e.id(v.Type),
			e.id(v.Index),
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
		raw.SetType(e.id(v.Type))

	case *Volatile:
		raw.SetKind(kindVolatile)
		raw.SetType(e.id(v.Type))

	case *Const:
		raw.SetKind(kindConst)
		raw.SetType(e.id(v.Type))

	case *Restrict:
		raw.SetKind(kindRestrict)
		raw.SetType(e.id(v.Type))

	case *Func:
		raw.SetKind(kindFunc)
		raw.SetType(e.id(v.Type))
		if !e.StripFuncLinkage {
			raw.SetLinkage(v.Linkage)
		}

	case *FuncProto:
		raw.SetKind(kindFuncProto)
		raw.SetType(e.id(v.Return))
		raw.SetVlen(len(v.Params))
		raw.data, err = e.deflateFuncParams(v.Params)

	case *Var:
		raw.SetKind(kindVar)
		raw.SetType(e.id(v.Type))
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
		raw.SetType(e.id(v.Type))
		raw.data = &btfDeclTag{uint32(v.Index)}
		raw.NameOff, err = e.strings.Add(v.Value)

	case *typeTag:
		raw.SetKind(kindTypeTag)
		raw.SetType(e.id(v.Type))
		raw.NameOff, err = e.strings.Add(v.Value)

	default:
		return fmt.Errorf("don't know how to deflate %T", v)
	}

	if err != nil {
		return err
	}

	return raw.Marshal(e.buf, e.Order)
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
			e.id(member.Type),
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
			e.id(param.Type),
		})
	}
	return bps, nil
}

func (e *encoder) deflateVarSecinfos(vars []VarSecinfo) []btfVarSecinfo {
	vsis := make([]btfVarSecinfo, 0, len(vars))
	for _, v := range vars {
		vsis = append(vsis, btfVarSecinfo{
			e.id(v.Type),
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
func MarshalMapKV(key, value Type) (_ *Handle, keyID, valueID TypeID, err error) {
	var b Builder

	if key != nil {
		keyID, err = b.Add(key)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("add key type: %w", err)
		}
	}

	if value != nil {
		valueID, err = b.Add(value)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("add value type: %w", err)
		}
	}

	handle, err := NewHandle(&b)
	if err != nil {
		// Check for 'full' map BTF support, since kernels between 4.18 and 5.2
		// already support BTF blobs for maps without Var or Datasec just fine.
		if err := haveMapBTF(); err != nil {
			return nil, 0, 0, err
		}
	}
	return handle, keyID, valueID, err
}
