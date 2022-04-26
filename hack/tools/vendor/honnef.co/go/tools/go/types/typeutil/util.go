package typeutil

import (
	"bytes"
	"go/types"
	"sync"
)

var bufferPool = &sync.Pool{
	New: func() interface{} {
		buf := bytes.NewBuffer(nil)
		buf.Grow(64)
		return buf
	},
}

func FuncName(f *types.Func) string {
	buf := bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	if f.Type() != nil {
		sig := f.Type().(*types.Signature)
		if recv := sig.Recv(); recv != nil {
			buf.WriteByte('(')
			if _, ok := recv.Type().(*types.Interface); ok {
				// gcimporter creates abstract methods of
				// named interfaces using the interface type
				// (not the named type) as the receiver.
				// Don't print it in full.
				buf.WriteString("interface")
			} else {
				types.WriteType(buf, recv.Type(), nil)
			}
			buf.WriteByte(')')
			buf.WriteByte('.')
		} else if f.Pkg() != nil {
			writePackage(buf, f.Pkg())
		}
	}
	buf.WriteString(f.Name())
	s := buf.String()
	bufferPool.Put(buf)
	return s
}

func writePackage(buf *bytes.Buffer, pkg *types.Package) {
	if pkg == nil {
		return
	}
	s := pkg.Path()
	if s != "" {
		buf.WriteString(s)
		buf.WriteByte('.')
	}
}

// Dereference returns a pointer's element type; otherwise it returns
// T.
func Dereference(T types.Type) types.Type {
	if p, ok := T.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return T
}

// DereferenceR returns a pointer's element type; otherwise it returns
// T. If the element type is itself a pointer, DereferenceR will be
// applied recursively.
func DereferenceR(T types.Type) types.Type {
	if p, ok := T.Underlying().(*types.Pointer); ok {
		return DereferenceR(p.Elem())
	}
	return T
}

func IsObject(obj types.Object, name string) bool {
	var path string
	if pkg := obj.Pkg(); pkg != nil {
		path = pkg.Path() + "."
	}
	return path+obj.Name() == name
}

// OPT(dh): IsType is kind of expensive; should we really use it?
func IsType(T types.Type, name string) bool { return types.TypeString(T, nil) == name }

func IsPointerLike(T types.Type) bool {
	switch T := T.Underlying().(type) {
	case *types.Interface, *types.Chan, *types.Map, *types.Signature, *types.Pointer, *types.Slice:
		return true
	case *types.Basic:
		return T.Kind() == types.UnsafePointer
	}
	return false
}

type Field struct {
	Var  *types.Var
	Tag  string
	Path []int
}

// FlattenFields recursively flattens T and embedded structs,
// returning a list of fields. If multiple fields with the same name
// exist, all will be returned.
func FlattenFields(T *types.Struct) []Field {
	return flattenFields(T, nil, nil)
}

func flattenFields(T *types.Struct, path []int, seen map[types.Type]bool) []Field {
	if seen == nil {
		seen = map[types.Type]bool{}
	}
	if seen[T] {
		return nil
	}
	seen[T] = true
	var out []Field
	for i := 0; i < T.NumFields(); i++ {
		field := T.Field(i)
		tag := T.Tag(i)
		np := append(path[:len(path):len(path)], i)
		if field.Anonymous() {
			if s, ok := Dereference(field.Type()).Underlying().(*types.Struct); ok {
				out = append(out, flattenFields(s, np, seen)...)
			}
		} else {
			out = append(out, Field{field, tag, np})
		}
	}
	return out
}
