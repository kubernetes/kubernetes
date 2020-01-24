package btf

import (
	"math"

	"github.com/pkg/errors"
)

const maxTypeDepth = 32

// TypeID identifies a type in a BTF section.
type TypeID uint32

// ID implements part of the Type interface.
func (tid TypeID) ID() TypeID {
	return tid
}

// Type represents a type described by BTF.
type Type interface {
	ID() TypeID

	// Make a copy of the type, without copying Type members.
	copy() Type

	walk(*copyStack)
}

// Name identifies a type.
//
// Anonymous types have an empty name.
type Name string

func (n Name) name() string {
	return string(n)
}

// Void is the unit type of BTF.
type Void struct{}

func (v Void) ID() TypeID      { return 0 }
func (v Void) copy() Type      { return Void{} }
func (v Void) walk(*copyStack) {}

// Int is an integer of a given length.
type Int struct {
	TypeID
	Name

	// The size of the integer in bytes.
	Size uint32
}

func (i *Int) size() uint32    { return i.Size }
func (i *Int) walk(*copyStack) {}
func (i *Int) copy() Type {
	cpy := *i
	return &cpy
}

// Pointer is a pointer to another type.
type Pointer struct {
	TypeID
	Target Type
}

func (p *Pointer) size() uint32       { return 8 }
func (p *Pointer) walk(cs *copyStack) { cs.push(&p.Target) }
func (p *Pointer) copy() Type {
	cpy := *p
	return &cpy
}

// Array is an array with a fixed number of elements.
type Array struct {
	TypeID
	Type   Type
	Nelems uint32
}

func (arr *Array) walk(cs *copyStack) { cs.push(&arr.Type) }
func (arr *Array) copy() Type {
	cpy := *arr
	return &cpy
}

// Struct is a compound type of consecutive members.
type Struct struct {
	TypeID
	Name
	// The size of the struct including padding, in bytes
	Size    uint32
	Members []Member
}

func (s *Struct) size() uint32 { return s.Size }

func (s *Struct) walk(cs *copyStack) {
	for i := range s.Members {
		cs.push(&s.Members[i].Type)
	}
}

func (s *Struct) copy() Type {
	cpy := *s
	cpy.Members = copyMembers(cpy.Members)
	return &cpy
}

// Union is a compound type where members occupy the same memory.
type Union struct {
	TypeID
	Name
	// The size of the union including padding, in bytes.
	Size    uint32
	Members []Member
}

func (u *Union) size() uint32 { return u.Size }

func (u *Union) walk(cs *copyStack) {
	for i := range u.Members {
		cs.push(&u.Members[i].Type)
	}
}

func (u *Union) copy() Type {
	cpy := *u
	cpy.Members = copyMembers(cpy.Members)
	return &cpy
}

// Member is part of a Struct or Union.
//
// It is not a valid Type.
type Member struct {
	Name
	Type   Type
	Offset uint32
}

func copyMembers(in []Member) []Member {
	cpy := make([]Member, 0, len(in))
	for _, member := range in {
		cpy = append(cpy, member)
	}
	return cpy
}

// Enum lists possible values.
type Enum struct {
	TypeID
	Name
}

func (e *Enum) size() uint32    { return 4 }
func (e *Enum) walk(*copyStack) {}
func (e *Enum) copy() Type {
	cpy := *e
	return &cpy
}

// Fwd is a forward declaration of a Type.
type Fwd struct {
	TypeID
	Name
}

func (f *Fwd) walk(*copyStack) {}
func (f *Fwd) copy() Type {
	cpy := *f
	return &cpy
}

// Typedef is an alias of a Type.
type Typedef struct {
	TypeID
	Name
	Type Type
}

func (td *Typedef) walk(cs *copyStack) { cs.push(&td.Type) }
func (td *Typedef) copy() Type {
	cpy := *td
	return &cpy
}

// Volatile is a modifier.
type Volatile struct {
	TypeID
	Type Type
}

func (v *Volatile) walk(cs *copyStack) { cs.push(&v.Type) }
func (v *Volatile) copy() Type {
	cpy := *v
	return &cpy
}

// Const is a modifier.
type Const struct {
	TypeID
	Type Type
}

func (c *Const) walk(cs *copyStack) { cs.push(&c.Type) }
func (c *Const) copy() Type {
	cpy := *c
	return &cpy
}

// Restrict is a modifier.
type Restrict struct {
	TypeID
	Type Type
}

func (r *Restrict) walk(cs *copyStack) { cs.push(&r.Type) }
func (r *Restrict) copy() Type {
	cpy := *r
	return &cpy
}

// Func is a function definition.
type Func struct {
	TypeID
	Name
	Type Type
}

func (f *Func) walk(cs *copyStack) { cs.push(&f.Type) }
func (f *Func) copy() Type {
	cpy := *f
	return &cpy
}

// FuncProto is a function declaration.
type FuncProto struct {
	TypeID
	Return Type
	// Parameters not supported yet
}

func (fp *FuncProto) walk(cs *copyStack) { cs.push(&fp.Return) }
func (fp *FuncProto) copy() Type {
	cpy := *fp
	return &cpy
}

// Var is a global variable.
type Var struct {
	TypeID
	Name
	Type Type
}

func (v *Var) walk(cs *copyStack) { cs.push(&v.Type) }
func (v *Var) copy() Type {
	cpy := *v
	return &cpy
}

// Datasec is a global program section containing data.
type Datasec struct {
	TypeID
	Name
	Size uint32
}

func (ds *Datasec) size() uint32    { return ds.Size }
func (ds *Datasec) walk(*copyStack) {}
func (ds *Datasec) copy() Type {
	cpy := *ds
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
				return 0, errors.New("overflow")
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
		case *Volatile:
			typ = v.Type
			continue
		case *Const:
			typ = v.Type
			continue
		case *Restrict:
			typ = v.Type
			continue

		default:
			return 0, errors.Errorf("unrecognized type %T", typ)
		}

		if n > 0 && elem > math.MaxInt64/n {
			return 0, errors.New("overflow")
		}

		size := n * elem
		if int64(int(size)) != size {
			return 0, errors.New("overflow")
		}

		return int(size), nil
	}

	return 0, errors.New("exceeded type depth")
}

// copy a Type recursively.
//
// typ may form a cycle.
func copyType(typ Type) Type {
	var (
		copies = make(map[Type]Type)
		work   copyStack
	)

	for t := &typ; t != nil; t = work.pop() {
		// *t is the identity of the type.
		if cpy := copies[*t]; cpy != nil {
			*t = cpy
			continue
		}

		cpy := (*t).copy()
		copies[*t] = cpy
		*t = cpy

		// Mark any nested types for copying.
		cpy.walk(&work)
	}

	return typ
}

// copyStack keeps track of pointers to types which still
// need to be copied.
type copyStack []*Type

// push adds a type to the stack.
func (cs *copyStack) push(t *Type) {
	*cs = append(*cs, t)
}

// pop returns the topmost Type, or nil.
func (cs *copyStack) pop() *Type {
	n := len(*cs)
	if n == 0 {
		return nil
	}

	t := (*cs)[n-1]
	*cs = (*cs)[:n-1]
	return t
}

type namer interface {
	name() string
}

var _ namer = Name("")

// inflateRawTypes takes a list of raw btf types linked via type IDs, and turns
// it into a graph of Types connected via pointers.
//
// Returns a map of named types (so, where NameOff is non-zero). Since BTF ignores
// compilation units, multiple types may share the same name. A Type may form a
// cyclic graph by pointing at itself.
func inflateRawTypes(rawTypes []rawType, rawStrings stringTable) (namedTypes map[string][]Type, err error) {
	type fixup struct {
		id  TypeID
		typ *Type
	}

	var fixups []fixup
	convertMembers := func(raw []btfMember) ([]Member, error) {
		// NB: The fixup below relies on pre-allocating this array to
		// work, since otherwise append might re-allocate members.
		members := make([]Member, 0, len(raw))
		for i, btfMember := range raw {
			name, err := rawStrings.LookupName(btfMember.NameOff)
			if err != nil {
				return nil, errors.Wrapf(err, "can't get name for member %d", i)
			}
			members = append(members, Member{
				Name:   name,
				Offset: btfMember.Offset,
			})
		}
		for i := range members {
			fixups = append(fixups, fixup{raw[i].Type, &members[i].Type})
		}
		return members, nil
	}

	types := make([]Type, 0, len(rawTypes))
	types = append(types, Void{})
	namedTypes = make(map[string][]Type)

	for i, raw := range rawTypes {
		var (
			// Void is defined to always be type ID 0, and is thus
			// omitted from BTF.
			id  = TypeID(i + 1)
			typ Type
		)

		name, err := rawStrings.LookupName(raw.NameOff)
		if err != nil {
			return nil, errors.Wrapf(err, "can't get name for type id %d", id)
		}

		switch raw.Kind() {
		case kindInt:
			typ = &Int{id, name, raw.Size()}

		case kindPointer:
			ptr := &Pointer{id, nil}
			fixups = append(fixups, fixup{raw.Type(), &ptr.Target})
			typ = ptr

		case kindArray:
			btfArr := raw.data.(*btfArray)

			// IndexType is unused according to btf.rst.
			// Don't make it available right now.
			arr := &Array{id, nil, btfArr.Nelems}
			fixups = append(fixups, fixup{btfArr.Type, &arr.Type})
			typ = arr

		case kindStruct:
			members, err := convertMembers(raw.data.([]btfMember))
			if err != nil {
				return nil, errors.Wrapf(err, "struct %s (id %d)", name, id)
			}
			typ = &Struct{id, name, raw.Size(), members}

		case kindUnion:
			members, err := convertMembers(raw.data.([]btfMember))
			if err != nil {
				return nil, errors.Wrapf(err, "union %s (id %d)", name, id)
			}
			typ = &Union{id, name, raw.Size(), members}

		case kindEnum:
			typ = &Enum{id, name}

		case kindForward:
			typ = &Fwd{id, name}

		case kindTypedef:
			typedef := &Typedef{id, name, nil}
			fixups = append(fixups, fixup{raw.Type(), &typedef.Type})
			typ = typedef

		case kindVolatile:
			volatile := &Volatile{id, nil}
			fixups = append(fixups, fixup{raw.Type(), &volatile.Type})
			typ = volatile

		case kindConst:
			cnst := &Const{id, nil}
			fixups = append(fixups, fixup{raw.Type(), &cnst.Type})
			typ = cnst

		case kindRestrict:
			restrict := &Restrict{id, nil}
			fixups = append(fixups, fixup{raw.Type(), &restrict.Type})
			typ = restrict

		case kindFunc:
			fn := &Func{id, name, nil}
			fixups = append(fixups, fixup{raw.Type(), &fn.Type})
			typ = fn

		case kindFuncProto:
			fp := &FuncProto{id, nil}
			fixups = append(fixups, fixup{raw.Type(), &fp.Return})
			typ = fp

		case kindVar:
			v := &Var{id, name, nil}
			fixups = append(fixups, fixup{raw.Type(), &v.Type})
			typ = v

		case kindDatasec:
			typ = &Datasec{id, name, raw.SizeType}

		default:
			return nil, errors.Errorf("type id %d: unknown kind: %v", id, raw.Kind())
		}

		types = append(types, typ)

		if namer, ok := typ.(namer); ok {
			if name := namer.name(); name != "" {
				namedTypes[name] = append(namedTypes[name], typ)
			}
		}
	}

	for _, fixup := range fixups {
		i := int(fixup.id)
		if i >= len(types) {
			return nil, errors.Errorf("reference to invalid type id: %d", fixup.id)
		}

		*fixup.typ = types[i]
	}

	return namedTypes, nil
}
