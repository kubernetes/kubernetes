package btf

import (
	"errors"
	"fmt"
	"strings"
)

var errNestedTooDeep = errors.New("nested too deep")

// GoFormatter converts a Type to Go syntax.
//
// A zero GoFormatter is valid to use.
type GoFormatter struct {
	w strings.Builder

	// Types present in this map are referred to using the given name if they
	// are encountered when outputting another type.
	Names map[Type]string

	// Identifier is called for each field of struct-like types. By default the
	// field name is used as is.
	Identifier func(string) string

	// EnumIdentifier is called for each element of an enum. By default the
	// name of the enum type is concatenated with Identifier(element).
	EnumIdentifier func(name, element string) string
}

// TypeDeclaration generates a Go type declaration for a BTF type.
func (gf *GoFormatter) TypeDeclaration(name string, typ Type) (string, error) {
	gf.w.Reset()
	if err := gf.writeTypeDecl(name, typ); err != nil {
		return "", err
	}
	return gf.w.String(), nil
}

func (gf *GoFormatter) identifier(s string) string {
	if gf.Identifier != nil {
		return gf.Identifier(s)
	}

	return s
}

func (gf *GoFormatter) enumIdentifier(name, element string) string {
	if gf.EnumIdentifier != nil {
		return gf.EnumIdentifier(name, element)
	}

	return name + gf.identifier(element)
}

// writeTypeDecl outputs a declaration of the given type.
//
// It encodes https://golang.org/ref/spec#Type_declarations:
//
//     type foo struct { bar uint32; }
//     type bar int32
func (gf *GoFormatter) writeTypeDecl(name string, typ Type) error {
	if name == "" {
		return fmt.Errorf("need a name for type %s", typ)
	}

	switch v := skipQualifiers(typ).(type) {
	case *Enum:
		fmt.Fprintf(&gf.w, "type %s ", name)
		switch v.Size {
		case 1:
			gf.w.WriteString("int8")
		case 2:
			gf.w.WriteString("int16")
		case 4:
			gf.w.WriteString("int32")
		case 8:
			gf.w.WriteString("int64")
		default:
			return fmt.Errorf("%s: invalid enum size %d", typ, v.Size)
		}

		if len(v.Values) == 0 {
			return nil
		}

		gf.w.WriteString("; const ( ")
		for _, ev := range v.Values {
			id := gf.enumIdentifier(name, ev.Name)
			fmt.Fprintf(&gf.w, "%s %s = %d; ", id, name, ev.Value)
		}
		gf.w.WriteString(")")

		return nil

	default:
		fmt.Fprintf(&gf.w, "type %s ", name)
		return gf.writeTypeLit(v, 0)
	}
}

// writeType outputs the name of a named type or a literal describing the type.
//
// It encodes https://golang.org/ref/spec#Types.
//
//     foo                  (if foo is a named type)
//     uint32
func (gf *GoFormatter) writeType(typ Type, depth int) error {
	typ = skipQualifiers(typ)

	name := gf.Names[typ]
	if name != "" {
		gf.w.WriteString(name)
		return nil
	}

	return gf.writeTypeLit(typ, depth)
}

// writeTypeLit outputs a literal describing the type.
//
// The function ignores named types.
//
// It encodes https://golang.org/ref/spec#TypeLit.
//
//     struct { bar uint32; }
//     uint32
func (gf *GoFormatter) writeTypeLit(typ Type, depth int) error {
	depth++
	if depth > maxTypeDepth {
		return errNestedTooDeep
	}

	var err error
	switch v := skipQualifiers(typ).(type) {
	case *Int:
		gf.writeIntLit(v)

	case *Enum:
		gf.w.WriteString("int32")

	case *Typedef:
		err = gf.writeType(v.Type, depth)

	case *Array:
		fmt.Fprintf(&gf.w, "[%d]", v.Nelems)
		err = gf.writeType(v.Type, depth)

	case *Struct:
		err = gf.writeStructLit(v.Size, v.Members, depth)

	case *Union:
		// Always choose the first member to represent the union in Go.
		err = gf.writeStructLit(v.Size, v.Members[:1], depth)

	case *Datasec:
		err = gf.writeDatasecLit(v, depth)

	default:
		return fmt.Errorf("type %T: %w", v, ErrNotSupported)
	}

	if err != nil {
		return fmt.Errorf("%s: %w", typ, err)
	}

	return nil
}

func (gf *GoFormatter) writeIntLit(i *Int) {
	// NB: Encoding.IsChar is ignored.
	if i.Encoding.IsBool() && i.Size == 1 {
		gf.w.WriteString("bool")
		return
	}

	bits := i.Size * 8
	if i.Encoding.IsSigned() {
		fmt.Fprintf(&gf.w, "int%d", bits)
	} else {
		fmt.Fprintf(&gf.w, "uint%d", bits)
	}
}

func (gf *GoFormatter) writeStructLit(size uint32, members []Member, depth int) error {
	gf.w.WriteString("struct { ")

	prevOffset := uint32(0)
	skippedBitfield := false
	for i, m := range members {
		if m.BitfieldSize > 0 {
			skippedBitfield = true
			continue
		}

		offset := m.Offset.Bytes()
		if n := offset - prevOffset; skippedBitfield && n > 0 {
			fmt.Fprintf(&gf.w, "_ [%d]byte /* unsupported bitfield */; ", n)
		} else {
			gf.writePadding(n)
		}

		size, err := Sizeof(m.Type)
		if err != nil {
			return fmt.Errorf("field %d: %w", i, err)
		}
		prevOffset = offset + uint32(size)

		if err := gf.writeStructField(m, depth); err != nil {
			return fmt.Errorf("field %d: %w", i, err)
		}
	}

	gf.writePadding(size - prevOffset)
	gf.w.WriteString("}")
	return nil
}

func (gf *GoFormatter) writeStructField(m Member, depth int) error {
	if m.BitfieldSize > 0 {
		return fmt.Errorf("bitfields are not supported")
	}
	if m.Offset%8 != 0 {
		return fmt.Errorf("unsupported offset %d", m.Offset)
	}

	if m.Name == "" {
		// Special case a nested anonymous union like
		//     struct foo { union { int bar; int baz }; }
		// by replacing the whole union with its first member.
		union, ok := m.Type.(*Union)
		if !ok {
			return fmt.Errorf("anonymous fields are not supported")

		}

		if len(union.Members) == 0 {
			return errors.New("empty anonymous union")
		}

		depth++
		if depth > maxTypeDepth {
			return errNestedTooDeep
		}

		m := union.Members[0]
		size, err := Sizeof(m.Type)
		if err != nil {
			return err
		}

		if err := gf.writeStructField(m, depth); err != nil {
			return err
		}

		gf.writePadding(union.Size - uint32(size))
		return nil

	}

	fmt.Fprintf(&gf.w, "%s ", gf.identifier(m.Name))

	if err := gf.writeType(m.Type, depth); err != nil {
		return err
	}

	gf.w.WriteString("; ")
	return nil
}

func (gf *GoFormatter) writeDatasecLit(ds *Datasec, depth int) error {
	gf.w.WriteString("struct { ")

	prevOffset := uint32(0)
	for i, vsi := range ds.Vars {
		v := vsi.Type.(*Var)
		if v.Linkage != GlobalVar {
			// Ignore static, extern, etc. for now.
			continue
		}

		if v.Name == "" {
			return fmt.Errorf("variable %d: empty name", i)
		}

		gf.writePadding(vsi.Offset - prevOffset)
		prevOffset = vsi.Offset + vsi.Size

		fmt.Fprintf(&gf.w, "%s ", gf.identifier(v.Name))

		if err := gf.writeType(v.Type, depth); err != nil {
			return fmt.Errorf("variable %d: %w", i, err)
		}

		gf.w.WriteString("; ")
	}

	gf.writePadding(ds.Size - prevOffset)
	gf.w.WriteString("}")
	return nil
}

func (gf *GoFormatter) writePadding(bytes uint32) {
	if bytes > 0 {
		fmt.Fprintf(&gf.w, "_ [%d]byte; ", bytes)
	}
}

func skipQualifiers(typ Type) Type {
	result := typ
	for depth := 0; depth <= maxTypeDepth; depth++ {
		switch v := (result).(type) {
		case qualifier:
			result = v.qualify()
		default:
			return result
		}
	}
	return &cycle{typ}
}
