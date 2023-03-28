package btf

import (
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
)

const maxTypeDepth = 32

// TypeID identifies a type in a BTF section.
type TypeID uint32

// Type represents a type described by BTF.
type Type interface {
	// Type can be formatted using the %s and %v verbs. %s outputs only the
	// identity of the type, without any detail. %v outputs additional detail.
	//
	// Use the '+' flag to include the address of the type.
	//
	// Use the width to specify how many levels of detail to output, for example
	// %1v will output detail for the root type and a short description of its
	// children. %2v would output details of the root type and its children
	// as well as a short description of the grandchildren.
	fmt.Formatter

	// Name of the type, empty for anonymous types and types that cannot
	// carry a name, like Void and Pointer.
	TypeName() string

	// Make a copy of the type, without copying Type members.
	copy() Type

	// New implementations must update walkType.
}

var (
	_ Type = (*Int)(nil)
	_ Type = (*Struct)(nil)
	_ Type = (*Union)(nil)
	_ Type = (*Enum)(nil)
	_ Type = (*Fwd)(nil)
	_ Type = (*Func)(nil)
	_ Type = (*Typedef)(nil)
	_ Type = (*Var)(nil)
	_ Type = (*Datasec)(nil)
	_ Type = (*Float)(nil)
	_ Type = (*declTag)(nil)
	_ Type = (*typeTag)(nil)
	_ Type = (*cycle)(nil)
)

// types is a list of Type.
//
// The order determines the ID of a type.
type types []Type

func (ts types) ByID(id TypeID) (Type, error) {
	if int(id) > len(ts) {
		return nil, fmt.Errorf("type ID %d: %w", id, ErrNotFound)
	}
	return ts[id], nil
}

// Void is the unit type of BTF.
type Void struct{}

func (v *Void) Format(fs fmt.State, verb rune) { formatType(fs, verb, v) }
func (v *Void) TypeName() string               { return "" }
func (v *Void) size() uint32                   { return 0 }
func (v *Void) copy() Type                     { return (*Void)(nil) }

type IntEncoding byte

// Valid IntEncodings.
//
// These may look like they are flags, but they aren't.
const (
	Unsigned IntEncoding = 0
	Signed   IntEncoding = 1
	Char     IntEncoding = 2
	Bool     IntEncoding = 4
)

func (ie IntEncoding) String() string {
	switch ie {
	case Char:
		// NB: There is no way to determine signedness for char.
		return "char"
	case Bool:
		return "bool"
	case Signed:
		return "signed"
	case Unsigned:
		return "unsigned"
	default:
		return fmt.Sprintf("IntEncoding(%d)", byte(ie))
	}
}

// Int is an integer of a given length.
//
// See https://www.kernel.org/doc/html/latest/bpf/btf.html#btf-kind-int
type Int struct {
	Name string

	// The size of the integer in bytes.
	Size     uint32
	Encoding IntEncoding
}

func (i *Int) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, i, i.Encoding, "size=", i.Size*8)
}

func (i *Int) TypeName() string { return i.Name }
func (i *Int) size() uint32     { return i.Size }
func (i *Int) copy() Type {
	cpy := *i
	return &cpy
}

// Pointer is a pointer to another type.
type Pointer struct {
	Target Type
}

func (p *Pointer) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, p, "target=", p.Target)
}

func (p *Pointer) TypeName() string { return "" }
func (p *Pointer) size() uint32     { return 8 }
func (p *Pointer) copy() Type {
	cpy := *p
	return &cpy
}

// Array is an array with a fixed number of elements.
type Array struct {
	Index  Type
	Type   Type
	Nelems uint32
}

func (arr *Array) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, arr, "index=", arr.Index, "type=", arr.Type, "n=", arr.Nelems)
}

func (arr *Array) TypeName() string { return "" }

func (arr *Array) copy() Type {
	cpy := *arr
	return &cpy
}

// Struct is a compound type of consecutive members.
type Struct struct {
	Name string
	// The size of the struct including padding, in bytes
	Size    uint32
	Members []Member
}

func (s *Struct) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, s, "fields=", len(s.Members))
}

func (s *Struct) TypeName() string { return s.Name }

func (s *Struct) size() uint32 { return s.Size }

func (s *Struct) copy() Type {
	cpy := *s
	cpy.Members = copyMembers(s.Members)
	return &cpy
}

func (s *Struct) members() []Member {
	return s.Members
}

// Union is a compound type where members occupy the same memory.
type Union struct {
	Name string
	// The size of the union including padding, in bytes.
	Size    uint32
	Members []Member
}

func (u *Union) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, u, "fields=", len(u.Members))
}

func (u *Union) TypeName() string { return u.Name }

func (u *Union) size() uint32 { return u.Size }

func (u *Union) copy() Type {
	cpy := *u
	cpy.Members = copyMembers(u.Members)
	return &cpy
}

func (u *Union) members() []Member {
	return u.Members
}

func copyMembers(orig []Member) []Member {
	cpy := make([]Member, len(orig))
	copy(cpy, orig)
	return cpy
}

type composite interface {
	members() []Member
}

var (
	_ composite = (*Struct)(nil)
	_ composite = (*Union)(nil)
)

// A value in bits.
type Bits uint32

// Bytes converts a bit value into bytes.
func (b Bits) Bytes() uint32 {
	return uint32(b / 8)
}

// Member is part of a Struct or Union.
//
// It is not a valid Type.
type Member struct {
	Name         string
	Type         Type
	Offset       Bits
	BitfieldSize Bits
}

// Enum lists possible values.
type Enum struct {
	Name string
	// Size of the enum value in bytes.
	Size uint32
	// True if the values should be interpreted as signed integers.
	Signed bool
	Values []EnumValue
}

func (e *Enum) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, e, "size=", e.Size, "values=", len(e.Values))
}

func (e *Enum) TypeName() string { return e.Name }

// EnumValue is part of an Enum
//
// Is is not a valid Type
type EnumValue struct {
	Name  string
	Value uint64
}

func (e *Enum) size() uint32 { return e.Size }
func (e *Enum) copy() Type {
	cpy := *e
	cpy.Values = make([]EnumValue, len(e.Values))
	copy(cpy.Values, e.Values)
	return &cpy
}

// has64BitValues returns true if the Enum contains a value larger than 32 bits.
// Kernels before 6.0 have enum values that overrun u32 replaced with zeroes.
//
// 64-bit enums have their Enum.Size attributes correctly set to 8, but if we
// use the size attribute as a heuristic during BTF marshaling, we'll emit
// ENUM64s to kernels that don't support them.
func (e *Enum) has64BitValues() bool {
	for _, v := range e.Values {
		if v.Value > math.MaxUint32 {
			return true
		}
	}
	return false
}

// FwdKind is the type of forward declaration.
type FwdKind int

// Valid types of forward declaration.
const (
	FwdStruct FwdKind = iota
	FwdUnion
)

func (fk FwdKind) String() string {
	switch fk {
	case FwdStruct:
		return "struct"
	case FwdUnion:
		return "union"
	default:
		return fmt.Sprintf("%T(%d)", fk, int(fk))
	}
}

// Fwd is a forward declaration of a Type.
type Fwd struct {
	Name string
	Kind FwdKind
}

func (f *Fwd) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, f, f.Kind)
}

func (f *Fwd) TypeName() string { return f.Name }

func (f *Fwd) copy() Type {
	cpy := *f
	return &cpy
}

// Typedef is an alias of a Type.
type Typedef struct {
	Name string
	Type Type
}

func (td *Typedef) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, td, td.Type)
}

func (td *Typedef) TypeName() string { return td.Name }

func (td *Typedef) copy() Type {
	cpy := *td
	return &cpy
}

// Volatile is a qualifier.
type Volatile struct {
	Type Type
}

func (v *Volatile) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, v, v.Type)
}

func (v *Volatile) TypeName() string { return "" }

func (v *Volatile) qualify() Type { return v.Type }
func (v *Volatile) copy() Type {
	cpy := *v
	return &cpy
}

// Const is a qualifier.
type Const struct {
	Type Type
}

func (c *Const) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, c, c.Type)
}

func (c *Const) TypeName() string { return "" }

func (c *Const) qualify() Type { return c.Type }
func (c *Const) copy() Type {
	cpy := *c
	return &cpy
}

// Restrict is a qualifier.
type Restrict struct {
	Type Type
}

func (r *Restrict) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, r, r.Type)
}

func (r *Restrict) TypeName() string { return "" }

func (r *Restrict) qualify() Type { return r.Type }
func (r *Restrict) copy() Type {
	cpy := *r
	return &cpy
}

// Func is a function definition.
type Func struct {
	Name    string
	Type    Type
	Linkage FuncLinkage
}

func FuncMetadata(ins *asm.Instruction) *Func {
	fn, _ := ins.Metadata.Get(funcInfoMeta{}).(*Func)
	return fn
}

// WithFuncMetadata adds a btf.Func to the Metadata of asm.Instruction.
func WithFuncMetadata(ins asm.Instruction, fn *Func) asm.Instruction {
	ins.Metadata.Set(funcInfoMeta{}, fn)
	return ins
}

func (f *Func) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, f, f.Linkage, "proto=", f.Type)
}

func (f *Func) TypeName() string { return f.Name }

func (f *Func) copy() Type {
	cpy := *f
	return &cpy
}

// FuncProto is a function declaration.
type FuncProto struct {
	Return Type
	Params []FuncParam
}

func (fp *FuncProto) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, fp, "args=", len(fp.Params), "return=", fp.Return)
}

func (fp *FuncProto) TypeName() string { return "" }

func (fp *FuncProto) copy() Type {
	cpy := *fp
	cpy.Params = make([]FuncParam, len(fp.Params))
	copy(cpy.Params, fp.Params)
	return &cpy
}

type FuncParam struct {
	Name string
	Type Type
}

// Var is a global variable.
type Var struct {
	Name    string
	Type    Type
	Linkage VarLinkage
}

func (v *Var) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, v, v.Linkage)
}

func (v *Var) TypeName() string { return v.Name }

func (v *Var) copy() Type {
	cpy := *v
	return &cpy
}

// Datasec is a global program section containing data.
type Datasec struct {
	Name string
	Size uint32
	Vars []VarSecinfo
}

func (ds *Datasec) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, ds)
}

func (ds *Datasec) TypeName() string { return ds.Name }

func (ds *Datasec) size() uint32 { return ds.Size }

func (ds *Datasec) copy() Type {
	cpy := *ds
	cpy.Vars = make([]VarSecinfo, len(ds.Vars))
	copy(cpy.Vars, ds.Vars)
	return &cpy
}

// VarSecinfo describes variable in a Datasec.
//
// It is not a valid Type.
type VarSecinfo struct {
	// Var or Func.
	Type   Type
	Offset uint32
	Size   uint32
}

// Float is a float of a given length.
type Float struct {
	Name string

	// The size of the float in bytes.
	Size uint32
}

func (f *Float) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, f, "size=", f.Size*8)
}

func (f *Float) TypeName() string { return f.Name }
func (f *Float) size() uint32     { return f.Size }
func (f *Float) copy() Type {
	cpy := *f
	return &cpy
}

// declTag associates metadata with a declaration.
type declTag struct {
	Type  Type
	Value string
	// The index this tag refers to in the target type. For composite types,
	// a value of -1 indicates that the tag refers to the whole type. Otherwise
	// it indicates which member or argument the tag applies to.
	Index int
}

func (dt *declTag) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, dt, "type=", dt.Type, "value=", dt.Value, "index=", dt.Index)
}

func (dt *declTag) TypeName() string { return "" }
func (dt *declTag) copy() Type {
	cpy := *dt
	return &cpy
}

// typeTag associates metadata with a type.
type typeTag struct {
	Type  Type
	Value string
}

func (tt *typeTag) Format(fs fmt.State, verb rune) {
	formatType(fs, verb, tt, "type=", tt.Type, "value=", tt.Value)
}

func (tt *typeTag) TypeName() string { return "" }
func (tt *typeTag) qualify() Type    { return tt.Type }
func (tt *typeTag) copy() Type {
	cpy := *tt
	return &cpy
}

// cycle is a type which had to be elided since it exceeded maxTypeDepth.
type cycle struct {
	root Type
}

func (c *cycle) ID() TypeID                     { return math.MaxUint32 }
func (c *cycle) Format(fs fmt.State, verb rune) { formatType(fs, verb, c, "root=", c.root) }
func (c *cycle) TypeName() string               { return "" }
func (c *cycle) copy() Type {
	cpy := *c
	return &cpy
}

type sizer interface {
	size() uint32
}

var (
	_ sizer = (*Int)(nil)
	_ sizer = (*Pointer)(nil)
	_ sizer = (*Struct)(nil)
	_ sizer = (*Union)(nil)
	_ sizer = (*Enum)(nil)
	_ sizer = (*Datasec)(nil)
)

type qualifier interface {
	qualify() Type
}

var (
	_ qualifier = (*Const)(nil)
	_ qualifier = (*Restrict)(nil)
	_ qualifier = (*Volatile)(nil)
	_ qualifier = (*typeTag)(nil)
)

// Sizeof returns the size of a type in bytes.
//
// Returns an error if the size can't be computed.
func Sizeof(typ Type) (int, error) {
	var (
		n    = int64(1)
		elem int64
	)

	for i := 0; i < maxTypeDepth; i++ {
		switch v := typ.(type) {
		case *Array:
			if n > 0 && int64(v.Nelems) > math.MaxInt64/n {
				return 0, fmt.Errorf("type %s: overflow", typ)
			}

			// Arrays may be of zero length, which allows
			// n to be zero as well.
			n *= int64(v.Nelems)
			typ = v.Type
			continue

		case sizer:
			elem = int64(v.size())

		case *Typedef:
			typ = v.Type
			continue

		case qualifier:
			typ = v.qualify()
			continue

		default:
			return 0, fmt.Errorf("unsized type %T", typ)
		}

		if n > 0 && elem > math.MaxInt64/n {
			return 0, fmt.Errorf("type %s: overflow", typ)
		}

		size := n * elem
		if int64(int(size)) != size {
			return 0, fmt.Errorf("type %s: overflow", typ)
		}

		return int(size), nil
	}

	return 0, fmt.Errorf("type %s: exceeded type depth", typ)
}

// alignof returns the alignment of a type.
//
// Currently only supports the subset of types necessary for bitfield relocations.
func alignof(typ Type) (int, error) {
	switch t := UnderlyingType(typ).(type) {
	case *Enum:
		return int(t.size()), nil
	case *Int:
		return int(t.Size), nil
	default:
		return 0, fmt.Errorf("can't calculate alignment of %T", t)
	}
}

// Transformer modifies a given Type and returns the result.
//
// For example, UnderlyingType removes any qualifiers or typedefs from a type.
// See the example on Copy for how to use a transform.
type Transformer func(Type) Type

// Copy a Type recursively.
//
// typ may form a cycle. If transform is not nil, it is called with the
// to be copied type, and the returned value is copied instead.
func Copy(typ Type, transform Transformer) Type {
	copies := make(copier)
	copies.copy(&typ, transform)
	return typ
}

// copy a slice of Types recursively.
//
// See Copy for the semantics.
func copyTypes(types []Type, transform Transformer) []Type {
	result := make([]Type, len(types))
	copy(result, types)

	copies := make(copier)
	for i := range result {
		copies.copy(&result[i], transform)
	}

	return result
}

type copier map[Type]Type

func (c copier) copy(typ *Type, transform Transformer) {
	var work typeDeque
	for t := typ; t != nil; t = work.Pop() {
		// *t is the identity of the type.
		if cpy := c[*t]; cpy != nil {
			*t = cpy
			continue
		}

		var cpy Type
		if transform != nil {
			cpy = transform(*t).copy()
		} else {
			cpy = (*t).copy()
		}

		c[*t] = cpy
		*t = cpy

		// Mark any nested types for copying.
		walkType(cpy, work.Push)
	}
}

type typeDeque = internal.Deque[*Type]

// inflateRawTypes takes a list of raw btf types linked via type IDs, and turns
// it into a graph of Types connected via pointers.
//
// If baseTypes are provided, then the raw types are
// considered to be of a split BTF (e.g., a kernel module).
//
// Returns  a slice of types indexed by TypeID. Since BTF ignores compilation
// units, multiple types may share the same name. A Type may form a cyclic graph
// by pointing at itself.
func inflateRawTypes(rawTypes []rawType, baseTypes types, rawStrings *stringTable) ([]Type, error) {
	types := make([]Type, 0, len(rawTypes)+1) // +1 for Void added to base types

	typeIDOffset := TypeID(1) // Void is TypeID(0), so the rest starts from TypeID(1)

	if baseTypes == nil {
		// Void is defined to always be type ID 0, and is thus omitted from BTF.
		types = append(types, (*Void)(nil))
	} else {
		// For split BTF, the next ID is max base BTF type ID + 1
		typeIDOffset = TypeID(len(baseTypes))
	}

	type fixupDef struct {
		id  TypeID
		typ *Type
	}

	var fixups []fixupDef
	fixup := func(id TypeID, typ *Type) bool {
		if id < TypeID(len(baseTypes)) {
			*typ = baseTypes[id]
			return true
		}

		idx := id
		if baseTypes != nil {
			idx = id - TypeID(len(baseTypes))
		}
		if idx < TypeID(len(types)) {
			// We've already inflated this type, fix it up immediately.
			*typ = types[idx]
			return true
		}
		fixups = append(fixups, fixupDef{id, typ})
		return false
	}

	type assertion struct {
		id   TypeID
		typ  *Type
		want reflect.Type
	}

	var assertions []assertion
	fixupAndAssert := func(id TypeID, typ *Type, want reflect.Type) error {
		if !fixup(id, typ) {
			assertions = append(assertions, assertion{id, typ, want})
			return nil
		}

		// The type has already been fixed up, check the type immediately.
		if reflect.TypeOf(*typ) != want {
			return fmt.Errorf("type ID %d: expected %s, got %T", id, want, *typ)
		}
		return nil
	}

	type bitfieldFixupDef struct {
		id TypeID
		m  *Member
	}

	var (
		legacyBitfields = make(map[TypeID][2]Bits) // offset, size
		bitfieldFixups  []bitfieldFixupDef
	)
	convertMembers := func(raw []btfMember, kindFlag bool) ([]Member, error) {
		// NB: The fixup below relies on pre-allocating this array to
		// work, since otherwise append might re-allocate members.
		members := make([]Member, 0, len(raw))
		for i, btfMember := range raw {
			name, err := rawStrings.Lookup(btfMember.NameOff)
			if err != nil {
				return nil, fmt.Errorf("can't get name for member %d: %w", i, err)
			}

			members = append(members, Member{
				Name:   name,
				Offset: Bits(btfMember.Offset),
			})

			m := &members[i]
			fixup(raw[i].Type, &m.Type)

			if kindFlag {
				m.BitfieldSize = Bits(btfMember.Offset >> 24)
				m.Offset &= 0xffffff
				// We ignore legacy bitfield definitions if the current composite
				// is a new-style bitfield. This is kind of safe since offset and
				// size on the type of the member must be zero if kindFlat is set
				// according to spec.
				continue
			}

			// This may be a legacy bitfield, try to fix it up.
			data, ok := legacyBitfields[raw[i].Type]
			if ok {
				// Bingo!
				m.Offset += data[0]
				m.BitfieldSize = data[1]
				continue
			}

			if m.Type != nil {
				// We couldn't find a legacy bitfield, but we know that the member's
				// type has already been inflated. Hence we know that it can't be
				// a legacy bitfield and there is nothing left to do.
				continue
			}

			// We don't have fixup data, and the type we're pointing
			// at hasn't been inflated yet. No choice but to defer
			// the fixup.
			bitfieldFixups = append(bitfieldFixups, bitfieldFixupDef{
				raw[i].Type,
				m,
			})
		}
		return members, nil
	}

	var declTags []*declTag
	for i, raw := range rawTypes {
		var (
			id  = typeIDOffset + TypeID(i)
			typ Type
		)

		name, err := rawStrings.Lookup(raw.NameOff)
		if err != nil {
			return nil, fmt.Errorf("get name for type id %d: %w", id, err)
		}

		switch raw.Kind() {
		case kindInt:
			size := raw.Size()
			bi := raw.data.(*btfInt)
			if bi.Offset() > 0 || bi.Bits().Bytes() != size {
				legacyBitfields[id] = [2]Bits{bi.Offset(), bi.Bits()}
			}
			typ = &Int{name, raw.Size(), bi.Encoding()}

		case kindPointer:
			ptr := &Pointer{nil}
			fixup(raw.Type(), &ptr.Target)
			typ = ptr

		case kindArray:
			btfArr := raw.data.(*btfArray)
			arr := &Array{nil, nil, btfArr.Nelems}
			fixup(btfArr.IndexType, &arr.Index)
			fixup(btfArr.Type, &arr.Type)
			typ = arr

		case kindStruct:
			members, err := convertMembers(raw.data.([]btfMember), raw.Bitfield())
			if err != nil {
				return nil, fmt.Errorf("struct %s (id %d): %w", name, id, err)
			}
			typ = &Struct{name, raw.Size(), members}

		case kindUnion:
			members, err := convertMembers(raw.data.([]btfMember), raw.Bitfield())
			if err != nil {
				return nil, fmt.Errorf("union %s (id %d): %w", name, id, err)
			}
			typ = &Union{name, raw.Size(), members}

		case kindEnum:
			rawvals := raw.data.([]btfEnum)
			vals := make([]EnumValue, 0, len(rawvals))
			signed := raw.Signed()
			for i, btfVal := range rawvals {
				name, err := rawStrings.Lookup(btfVal.NameOff)
				if err != nil {
					return nil, fmt.Errorf("get name for enum value %d: %s", i, err)
				}
				value := uint64(btfVal.Val)
				if signed {
					// Sign extend values to 64 bit.
					value = uint64(int32(btfVal.Val))
				}
				vals = append(vals, EnumValue{name, value})
			}
			typ = &Enum{name, raw.Size(), signed, vals}

		case kindForward:
			typ = &Fwd{name, raw.FwdKind()}

		case kindTypedef:
			typedef := &Typedef{name, nil}
			fixup(raw.Type(), &typedef.Type)
			typ = typedef

		case kindVolatile:
			volatile := &Volatile{nil}
			fixup(raw.Type(), &volatile.Type)
			typ = volatile

		case kindConst:
			cnst := &Const{nil}
			fixup(raw.Type(), &cnst.Type)
			typ = cnst

		case kindRestrict:
			restrict := &Restrict{nil}
			fixup(raw.Type(), &restrict.Type)
			typ = restrict

		case kindFunc:
			fn := &Func{name, nil, raw.Linkage()}
			if err := fixupAndAssert(raw.Type(), &fn.Type, reflect.TypeOf((*FuncProto)(nil))); err != nil {
				return nil, err
			}
			typ = fn

		case kindFuncProto:
			rawparams := raw.data.([]btfParam)
			params := make([]FuncParam, 0, len(rawparams))
			for i, param := range rawparams {
				name, err := rawStrings.Lookup(param.NameOff)
				if err != nil {
					return nil, fmt.Errorf("get name for func proto parameter %d: %s", i, err)
				}
				params = append(params, FuncParam{
					Name: name,
				})
			}
			for i := range params {
				fixup(rawparams[i].Type, &params[i].Type)
			}

			fp := &FuncProto{nil, params}
			fixup(raw.Type(), &fp.Return)
			typ = fp

		case kindVar:
			variable := raw.data.(*btfVariable)
			v := &Var{name, nil, VarLinkage(variable.Linkage)}
			fixup(raw.Type(), &v.Type)
			typ = v

		case kindDatasec:
			btfVars := raw.data.([]btfVarSecinfo)
			vars := make([]VarSecinfo, 0, len(btfVars))
			for _, btfVar := range btfVars {
				vars = append(vars, VarSecinfo{
					Offset: btfVar.Offset,
					Size:   btfVar.Size,
				})
			}
			for i := range vars {
				fixup(btfVars[i].Type, &vars[i].Type)
			}
			typ = &Datasec{name, raw.Size(), vars}

		case kindFloat:
			typ = &Float{name, raw.Size()}

		case kindDeclTag:
			btfIndex := raw.data.(*btfDeclTag).ComponentIdx
			if uint64(btfIndex) > math.MaxInt {
				return nil, fmt.Errorf("type id %d: index exceeds int", id)
			}

			dt := &declTag{nil, name, int(int32(btfIndex))}
			fixup(raw.Type(), &dt.Type)
			typ = dt

			declTags = append(declTags, dt)

		case kindTypeTag:
			tt := &typeTag{nil, name}
			fixup(raw.Type(), &tt.Type)
			typ = tt

		case kindEnum64:
			rawvals := raw.data.([]btfEnum64)
			vals := make([]EnumValue, 0, len(rawvals))
			for i, btfVal := range rawvals {
				name, err := rawStrings.Lookup(btfVal.NameOff)
				if err != nil {
					return nil, fmt.Errorf("get name for enum64 value %d: %s", i, err)
				}
				value := (uint64(btfVal.ValHi32) << 32) | uint64(btfVal.ValLo32)
				vals = append(vals, EnumValue{name, value})
			}
			typ = &Enum{name, raw.Size(), raw.Signed(), vals}

		default:
			return nil, fmt.Errorf("type id %d: unknown kind: %v", id, raw.Kind())
		}

		types = append(types, typ)
	}

	for _, fixup := range fixups {
		i := int(fixup.id)
		if i >= len(types)+len(baseTypes) {
			return nil, fmt.Errorf("reference to invalid type id: %d", fixup.id)
		}
		if i < len(baseTypes) {
			return nil, fmt.Errorf("fixup for base type id %d is not expected", i)
		}

		*fixup.typ = types[i-len(baseTypes)]
	}

	for _, bitfieldFixup := range bitfieldFixups {
		if bitfieldFixup.id < TypeID(len(baseTypes)) {
			return nil, fmt.Errorf("bitfield fixup from split to base types is not expected")
		}

		data, ok := legacyBitfields[bitfieldFixup.id]
		if ok {
			// This is indeed a legacy bitfield, fix it up.
			bitfieldFixup.m.Offset += data[0]
			bitfieldFixup.m.BitfieldSize = data[1]
		}
	}

	for _, assertion := range assertions {
		if reflect.TypeOf(*assertion.typ) != assertion.want {
			return nil, fmt.Errorf("type ID %d: expected %s, got %T", assertion.id, assertion.want, *assertion.typ)
		}
	}

	for _, dt := range declTags {
		switch t := dt.Type.(type) {
		case *Var, *Typedef:
			if dt.Index != -1 {
				return nil, fmt.Errorf("type %s: index %d is not -1", dt, dt.Index)
			}

		case composite:
			if dt.Index >= len(t.members()) {
				return nil, fmt.Errorf("type %s: index %d exceeds members of %s", dt, dt.Index, t)
			}

		case *Func:
			if dt.Index >= len(t.Type.(*FuncProto).Params) {
				return nil, fmt.Errorf("type %s: index %d exceeds params of %s", dt, dt.Index, t)
			}

		default:
			return nil, fmt.Errorf("type %s: decl tag for type %s is not supported", dt, t)
		}
	}

	return types, nil
}

// essentialName represents the name of a BTF type stripped of any flavor
// suffixes after a ___ delimiter.
type essentialName string

// newEssentialName returns name without a ___ suffix.
//
// CO-RE has the concept of 'struct flavors', which are used to deal with
// changes in kernel data structures. Anything after three underscores
// in a type name is ignored for the purpose of finding a candidate type
// in the kernel's BTF.
func newEssentialName(name string) essentialName {
	if name == "" {
		return ""
	}
	lastIdx := strings.LastIndex(name, "___")
	if lastIdx > 0 {
		return essentialName(name[:lastIdx])
	}
	return essentialName(name)
}

// UnderlyingType skips qualifiers and Typedefs.
func UnderlyingType(typ Type) Type {
	result := typ
	for depth := 0; depth <= maxTypeDepth; depth++ {
		switch v := (result).(type) {
		case qualifier:
			result = v.qualify()
		case *Typedef:
			result = v.Type
		default:
			return result
		}
	}
	return &cycle{typ}
}

type formatState struct {
	fmt.State
	depth int
}

// formattableType is a subset of Type, to ease unit testing of formatType.
type formattableType interface {
	fmt.Formatter
	TypeName() string
}

// formatType formats a type in a canonical form.
//
// Handles cyclical types by only printing cycles up to a certain depth. Elements
// in extra are separated by spaces unless the preceding element is a string
// ending in '='.
func formatType(f fmt.State, verb rune, t formattableType, extra ...interface{}) {
	if verb != 'v' && verb != 's' {
		fmt.Fprintf(f, "{UNRECOGNIZED: %c}", verb)
		return
	}

	// This is the same as %T, but elides the package name. Assumes that
	// formattableType is implemented by a pointer receiver.
	goTypeName := reflect.TypeOf(t).Elem().Name()
	_, _ = io.WriteString(f, goTypeName)

	if name := t.TypeName(); name != "" {
		// Output BTF type name if present.
		fmt.Fprintf(f, ":%q", name)
	}

	if f.Flag('+') {
		// Output address if requested.
		fmt.Fprintf(f, ":%#p", t)
	}

	if verb == 's' {
		// %s omits details.
		return
	}

	var depth int
	if ps, ok := f.(*formatState); ok {
		depth = ps.depth
		f = ps.State
	}

	maxDepth, ok := f.Width()
	if !ok {
		maxDepth = 0
	}

	if depth > maxDepth {
		// We've reached the maximum depth. This avoids infinite recursion even
		// for cyclical types.
		return
	}

	if len(extra) == 0 {
		return
	}

	wantSpace := false
	_, _ = io.WriteString(f, "[")
	for _, arg := range extra {
		if wantSpace {
			_, _ = io.WriteString(f, " ")
		}

		switch v := arg.(type) {
		case string:
			_, _ = io.WriteString(f, v)
			wantSpace = len(v) > 0 && v[len(v)-1] != '='
			continue

		case formattableType:
			v.Format(&formatState{f, depth + 1}, verb)

		default:
			fmt.Fprint(f, arg)
		}

		wantSpace = true
	}
	_, _ = io.WriteString(f, "]")
}
