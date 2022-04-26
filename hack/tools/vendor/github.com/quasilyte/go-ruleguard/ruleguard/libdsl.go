package ruleguard

import (
	"go/types"

	"github.com/quasilyte/go-ruleguard/internal/xtypes"
	"github.com/quasilyte/go-ruleguard/ruleguard/quasigo"
)

// This file implements `dsl/*` packages as native functions in quasigo.
//
// Every function and method defined in any `dsl/*` package should have
// associated Go function that implements it.
//
// In quasigo, it's impossible to have a pointer to an interface and
// non-pointer struct type. All interface type methods have FQN without `*` prefix
// while all struct type methods always begin with `*`.
//
// Fields are readonly.
// Field access is compiled into a method call that have a name identical to the field.
// For example, `foo.Bar` field access will be compiled as `foo.Bar()`.
// This may change in the future; benchmarks are needed to figure out
// what is more efficient: reflect-based field access or a function call.
//
// To keep this code organized, every type and package functions are represented
// as structs with methods. Then we bind a method value to quasigo symbol.
// The naming scheme is `dsl{$name}Package` for packages and `dsl{$pkg}{$name}` for types.

func initEnv(state *engineState, env *quasigo.Env) {
	nativeTypes := map[string]quasigoNative{
		`*github.com/quasilyte/go-ruleguard/dsl.VarFilterContext`: dslVarFilterContext{state: state},
		`github.com/quasilyte/go-ruleguard/dsl/types.Type`:        dslTypesType{},
		`*github.com/quasilyte/go-ruleguard/dsl/types.Interface`:  dslTypesInterface{},
		`*github.com/quasilyte/go-ruleguard/dsl/types.Pointer`:    dslTypesPointer{},
		`*github.com/quasilyte/go-ruleguard/dsl/types.Array`:      dslTypesArray{},
		`*github.com/quasilyte/go-ruleguard/dsl/types.Slice`:      dslTypesSlice{},
	}

	for qualifier, typ := range nativeTypes {
		for methodName, fn := range typ.funcs() {
			env.AddNativeMethod(qualifier, methodName, fn)
		}
	}

	nativePackages := map[string]quasigoNative{
		`github.com/quasilyte/go-ruleguard/dsl/types`: dslTypesPackage{},
	}

	for qualifier, pkg := range nativePackages {
		for funcName, fn := range pkg.funcs() {
			env.AddNativeMethod(qualifier, funcName, fn)
		}
	}
}

type quasigoNative interface {
	funcs() map[string]func(*quasigo.ValueStack)
}

type dslTypesType struct{}

func (native dslTypesType) funcs() map[string]func(*quasigo.ValueStack) {
	return map[string]func(*quasigo.ValueStack){
		"Underlying": native.Underlying,
		"String":     native.String,
	}
}

func (dslTypesType) Underlying(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(types.Type).Underlying())
}

func (dslTypesType) String(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(types.Type).String())
}

type dslTypesInterface struct{}

func (native dslTypesInterface) funcs() map[string]func(*quasigo.ValueStack) {
	return map[string]func(*quasigo.ValueStack){
		"Underlying": native.Underlying,
		"String":     native.String,
	}
}

func (dslTypesInterface) Underlying(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Interface).Underlying())
}

func (dslTypesInterface) String(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Interface).String())
}

type dslTypesSlice struct{}

func (native dslTypesSlice) funcs() map[string]func(*quasigo.ValueStack) {
	return map[string]func(*quasigo.ValueStack){
		"Underlying": native.Underlying,
		"String":     native.String,
		"Elem":       native.Elem,
	}
}

func (dslTypesSlice) Underlying(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Slice).Underlying())
}

func (dslTypesSlice) String(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Slice).String())
}

func (dslTypesSlice) Elem(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Slice).Elem())
}

type dslTypesArray struct{}

func (native dslTypesArray) funcs() map[string]func(*quasigo.ValueStack) {
	return map[string]func(*quasigo.ValueStack){
		"Underlying": native.Underlying,
		"String":     native.String,
		"Elem":       native.Elem,
		"Len":        native.Len,
	}
}

func (dslTypesArray) Underlying(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Array).Underlying())
}

func (dslTypesArray) String(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Array).String())
}

func (dslTypesArray) Elem(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Array).Elem())
}

func (dslTypesArray) Len(stack *quasigo.ValueStack) {
	stack.PushInt(int(stack.Pop().(*types.Array).Len()))
}

type dslTypesPointer struct{}

func (native dslTypesPointer) funcs() map[string]func(*quasigo.ValueStack) {
	return map[string]func(*quasigo.ValueStack){
		"Underlying": native.Underlying,
		"String":     native.String,
		"Elem":       native.Elem,
	}
}

func (dslTypesPointer) Underlying(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Pointer).Underlying())
}

func (dslTypesPointer) String(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Pointer).String())
}

func (dslTypesPointer) Elem(stack *quasigo.ValueStack) {
	stack.Push(stack.Pop().(*types.Pointer).Elem())
}

type dslTypesPackage struct{}

func (native dslTypesPackage) funcs() map[string]func(*quasigo.ValueStack) {
	return map[string]func(*quasigo.ValueStack){
		"Implements":  native.Implements,
		"Identical":   native.Identical,
		"NewArray":    native.NewArray,
		"NewSlice":    native.NewSlice,
		"NewPointer":  native.NewPointer,
		"AsArray":     native.AsArray,
		"AsSlice":     native.AsSlice,
		"AsPointer":   native.AsPointer,
		"AsInterface": native.AsInterface,
	}
}

func (dslTypesPackage) Implements(stack *quasigo.ValueStack) {
	iface := stack.Pop().(*types.Interface)
	typ := stack.Pop().(types.Type)
	stack.Push(xtypes.Implements(typ, iface))
}

func (dslTypesPackage) Identical(stack *quasigo.ValueStack) {
	y := stack.Pop().(types.Type)
	x := stack.Pop().(types.Type)
	stack.Push(xtypes.Identical(x, y))
}

func (dslTypesPackage) NewArray(stack *quasigo.ValueStack) {
	length := stack.PopInt()
	typ := stack.Pop().(types.Type)
	stack.Push(types.NewArray(typ, int64(length)))
}

func (dslTypesPackage) NewSlice(stack *quasigo.ValueStack) {
	typ := stack.Pop().(types.Type)
	stack.Push(types.NewSlice(typ))
}

func (dslTypesPackage) NewPointer(stack *quasigo.ValueStack) {
	typ := stack.Pop().(types.Type)
	stack.Push(types.NewPointer(typ))
}

func (dslTypesPackage) AsArray(stack *quasigo.ValueStack) {
	typ, _ := stack.Pop().(types.Type).(*types.Array)
	stack.Push(typ)
}

func (dslTypesPackage) AsSlice(stack *quasigo.ValueStack) {
	typ, _ := stack.Pop().(types.Type).(*types.Slice)
	stack.Push(typ)
}

func (dslTypesPackage) AsPointer(stack *quasigo.ValueStack) {
	typ, _ := stack.Pop().(types.Type).(*types.Pointer)
	stack.Push(typ)
}

func (dslTypesPackage) AsInterface(stack *quasigo.ValueStack) {
	typ, _ := stack.Pop().(types.Type).(*types.Interface)
	stack.Push(typ)
}

type dslVarFilterContext struct {
	state *engineState
}

func (native dslVarFilterContext) funcs() map[string]func(*quasigo.ValueStack) {
	return map[string]func(*quasigo.ValueStack){
		"Type":         native.Type,
		"SizeOf":       native.SizeOf,
		"GetType":      native.GetType,
		"GetInterface": native.GetInterface,
	}
}

func (dslVarFilterContext) Type(stack *quasigo.ValueStack) {
	params := stack.Pop().(*filterParams)
	typ := params.typeofNode(params.subExpr(params.varname))
	stack.Push(typ)
}

func (native dslVarFilterContext) SizeOf(stack *quasigo.ValueStack) {
	typ := stack.Pop().(types.Type)
	params := stack.Pop().(*filterParams)
	stack.PushInt(int(params.ctx.Sizes.Sizeof(typ)))
}

func (native dslVarFilterContext) GetType(stack *quasigo.ValueStack) {
	fqn := stack.Pop().(string)
	params := stack.Pop().(*filterParams)
	typ, err := native.state.FindType(params.importer, params.ctx.Pkg, fqn)
	if err != nil {
		panic(err)
	}
	stack.Push(typ)
}

func (native dslVarFilterContext) GetInterface(stack *quasigo.ValueStack) {
	fqn := stack.Pop().(string)
	params := stack.Pop().(*filterParams)
	typ, err := native.state.FindType(params.importer, params.ctx.Pkg, fqn)
	if err != nil {
		panic(err)
	}
	if ifaceType, ok := typ.Underlying().(*types.Interface); ok {
		stack.Push(ifaceType)
		return
	}
	stack.Push((*types.Interface)(nil)) // Not found or not an interface
}
