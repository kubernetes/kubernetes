package fakereflect

import (
	"fmt"
	"go/types"
	"reflect"
)

type TypeAndCanAddr struct {
	Type    types.Type
	canAddr bool
}

type StructField struct {
	Index     []int
	Name      string
	Anonymous bool
	Tag       reflect.StructTag
	f         *types.Var
	Type      TypeAndCanAddr
}

func (sf StructField) IsExported() bool { return sf.f.Exported() }

func (t TypeAndCanAddr) Field(i int) StructField {
	st := t.Type.Underlying().(*types.Struct)
	f := st.Field(i)
	return StructField{
		f:         f,
		Index:     []int{i},
		Name:      f.Name(),
		Anonymous: f.Anonymous(),
		Tag:       reflect.StructTag(st.Tag(i)),
		Type: TypeAndCanAddr{
			Type:    f.Type(),
			canAddr: t.canAddr,
		},
	}
}

func (t TypeAndCanAddr) FieldByIndex(index []int) StructField {
	f := t.Field(index[0])
	for _, idx := range index[1:] {
		f = f.Type.Field(idx)
	}
	f.Index = index
	return f
}

func PtrTo(t TypeAndCanAddr) TypeAndCanAddr {
	// Note that we don't care about canAddr here because it's irrelevant to all uses of PtrTo
	return TypeAndCanAddr{Type: types.NewPointer(t.Type)}
}

func (t TypeAndCanAddr) CanAddr() bool { return t.canAddr }

func (t TypeAndCanAddr) Implements(ityp *types.Interface) bool {
	return types.Implements(t.Type, ityp)
}

func (t TypeAndCanAddr) IsSlice() bool {
	_, ok := t.Type.Underlying().(*types.Slice)
	return ok
}

func (t TypeAndCanAddr) IsArray() bool {
	_, ok := t.Type.Underlying().(*types.Array)
	return ok
}

func (t TypeAndCanAddr) IsPtr() bool {
	_, ok := t.Type.Underlying().(*types.Pointer)
	return ok
}

func (t TypeAndCanAddr) IsInterface() bool {
	_, ok := t.Type.Underlying().(*types.Interface)
	return ok
}

func (t TypeAndCanAddr) IsStruct() bool {
	_, ok := t.Type.Underlying().(*types.Struct)
	return ok
}

func (t TypeAndCanAddr) Name() string {
	named, ok := t.Type.(*types.Named)
	if !ok {
		return ""
	}
	return named.Obj().Name()
}

func (t TypeAndCanAddr) NumField() int {
	return t.Type.Underlying().(*types.Struct).NumFields()
}

func (t TypeAndCanAddr) String() string {
	return t.Type.String()
}

func (t TypeAndCanAddr) Key() TypeAndCanAddr {
	return TypeAndCanAddr{Type: t.Type.Underlying().(*types.Map).Key()}
}

func (t TypeAndCanAddr) Elem() TypeAndCanAddr {
	switch typ := t.Type.Underlying().(type) {
	case *types.Pointer:
		return TypeAndCanAddr{
			Type:    typ.Elem(),
			canAddr: true,
		}
	case *types.Slice:
		return TypeAndCanAddr{
			Type:    typ.Elem(),
			canAddr: true,
		}
	case *types.Array:
		return TypeAndCanAddr{
			Type:    typ.Elem(),
			canAddr: t.canAddr,
		}
	case *types.Map:
		return TypeAndCanAddr{
			Type:    typ.Elem(),
			canAddr: false,
		}
	default:
		panic(fmt.Sprintf("unhandled type %T", typ))
	}
}
