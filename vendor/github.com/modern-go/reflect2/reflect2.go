package reflect2

import (
	"github.com/modern-go/concurrent"
	"reflect"
	"unsafe"
)

type Type interface {
	Kind() reflect.Kind
	// New return pointer to data of this type
	New() interface{}
	// UnsafeNew return the allocated space pointed by unsafe.Pointer
	UnsafeNew() unsafe.Pointer
	// PackEFace cast a unsafe pointer to object represented pointer
	PackEFace(ptr unsafe.Pointer) interface{}
	// Indirect dereference object represented pointer to this type
	Indirect(obj interface{}) interface{}
	// UnsafeIndirect dereference pointer to this type
	UnsafeIndirect(ptr unsafe.Pointer) interface{}
	// Type1 returns reflect.Type
	Type1() reflect.Type
	Implements(thatType Type) bool
	String() string
	RType() uintptr
	// interface{} of this type has pointer like behavior
	LikePtr() bool
	IsNullable() bool
	IsNil(obj interface{}) bool
	UnsafeIsNil(ptr unsafe.Pointer) bool
	Set(obj interface{}, val interface{})
	UnsafeSet(ptr unsafe.Pointer, val unsafe.Pointer)
	AssignableTo(anotherType Type) bool
}

type ListType interface {
	Type
	Elem() Type
	SetIndex(obj interface{}, index int, elem interface{})
	UnsafeSetIndex(obj unsafe.Pointer, index int, elem unsafe.Pointer)
	GetIndex(obj interface{}, index int) interface{}
	UnsafeGetIndex(obj unsafe.Pointer, index int) unsafe.Pointer
}

type ArrayType interface {
	ListType
	Len() int
}

type SliceType interface {
	ListType
	MakeSlice(length int, cap int) interface{}
	UnsafeMakeSlice(length int, cap int) unsafe.Pointer
	Grow(obj interface{}, newLength int)
	UnsafeGrow(ptr unsafe.Pointer, newLength int)
	Append(obj interface{}, elem interface{})
	UnsafeAppend(obj unsafe.Pointer, elem unsafe.Pointer)
	LengthOf(obj interface{}) int
	UnsafeLengthOf(ptr unsafe.Pointer) int
	SetNil(obj interface{})
	UnsafeSetNil(ptr unsafe.Pointer)
	Cap(obj interface{}) int
	UnsafeCap(ptr unsafe.Pointer) int
}

type StructType interface {
	Type
	NumField() int
	Field(i int) StructField
	FieldByName(name string) StructField
	FieldByIndex(index []int) StructField
	FieldByNameFunc(match func(string) bool) StructField
}

type StructField interface {
	Offset() uintptr
	Name() string
	PkgPath() string
	Type() Type
	Tag() reflect.StructTag
	Index() []int
	Anonymous() bool
	Set(obj interface{}, value interface{})
	UnsafeSet(obj unsafe.Pointer, value unsafe.Pointer)
	Get(obj interface{}) interface{}
	UnsafeGet(obj unsafe.Pointer) unsafe.Pointer
}

type MapType interface {
	Type
	Key() Type
	Elem() Type
	MakeMap(cap int) interface{}
	UnsafeMakeMap(cap int) unsafe.Pointer
	SetIndex(obj interface{}, key interface{}, elem interface{})
	UnsafeSetIndex(obj unsafe.Pointer, key unsafe.Pointer, elem unsafe.Pointer)
	TryGetIndex(obj interface{}, key interface{}) (interface{}, bool)
	GetIndex(obj interface{}, key interface{}) interface{}
	UnsafeGetIndex(obj unsafe.Pointer, key unsafe.Pointer) unsafe.Pointer
	Iterate(obj interface{}) MapIterator
	UnsafeIterate(obj unsafe.Pointer) MapIterator
}

type MapIterator interface {
	HasNext() bool
	Next() (key interface{}, elem interface{})
	UnsafeNext() (key unsafe.Pointer, elem unsafe.Pointer)
}

type PtrType interface {
	Type
	Elem() Type
}

type InterfaceType interface {
	NumMethod() int
}

type Config struct {
	UseSafeImplementation bool
}

type API interface {
	TypeOf(obj interface{}) Type
	Type2(type1 reflect.Type) Type
}

var ConfigUnsafe = Config{UseSafeImplementation: false}.Froze()
var ConfigSafe = Config{UseSafeImplementation: true}.Froze()

type frozenConfig struct {
	useSafeImplementation bool
	cache                 *concurrent.Map
}

func (cfg Config) Froze() *frozenConfig {
	return &frozenConfig{
		useSafeImplementation: cfg.UseSafeImplementation,
		cache: concurrent.NewMap(),
	}
}

func (cfg *frozenConfig) TypeOf(obj interface{}) Type {
	cacheKey := uintptr(unpackEFace(obj).rtype)
	typeObj, found := cfg.cache.Load(cacheKey)
	if found {
		return typeObj.(Type)
	}
	return cfg.Type2(reflect.TypeOf(obj))
}

func (cfg *frozenConfig) Type2(type1 reflect.Type) Type {
	cacheKey := uintptr(unpackEFace(type1).data)
	typeObj, found := cfg.cache.Load(cacheKey)
	if found {
		return typeObj.(Type)
	}
	type2 := cfg.wrapType(type1)
	cfg.cache.Store(cacheKey, type2)
	return type2
}

func (cfg *frozenConfig) wrapType(type1 reflect.Type) Type {
	safeType := safeType{Type: type1, cfg: cfg}
	switch type1.Kind() {
	case reflect.Struct:
		if cfg.useSafeImplementation {
			return &safeStructType{safeType}
		}
		return newUnsafeStructType(cfg, type1)
	case reflect.Array:
		if cfg.useSafeImplementation {
			return &safeSliceType{safeType}
		}
		return newUnsafeArrayType(cfg, type1)
	case reflect.Slice:
		if cfg.useSafeImplementation {
			return &safeSliceType{safeType}
		}
		return newUnsafeSliceType(cfg, type1)
	case reflect.Map:
		if cfg.useSafeImplementation {
			return &safeMapType{safeType}
		}
		return newUnsafeMapType(cfg, type1)
	case reflect.Ptr, reflect.Chan, reflect.Func:
		if cfg.useSafeImplementation {
			return &safeMapType{safeType}
		}
		return newUnsafePtrType(cfg, type1)
	case reflect.Interface:
		if cfg.useSafeImplementation {
			return &safeMapType{safeType}
		}
		if type1.NumMethod() == 0 {
			return newUnsafeEFaceType(cfg, type1)
		}
		return newUnsafeIFaceType(cfg, type1)
	default:
		if cfg.useSafeImplementation {
			return &safeType
		}
		return newUnsafeType(cfg, type1)
	}
}

func TypeOf(obj interface{}) Type {
	return ConfigUnsafe.TypeOf(obj)
}

func TypeOfPtr(obj interface{}) PtrType {
	return TypeOf(obj).(PtrType)
}

func Type2(type1 reflect.Type) Type {
	if type1 == nil {
		return nil
	}
	return ConfigUnsafe.Type2(type1)
}

func PtrTo(typ Type) Type {
	return Type2(reflect.PtrTo(typ.Type1()))
}

func PtrOf(obj interface{}) unsafe.Pointer {
	return unpackEFace(obj).data
}

func RTypeOf(obj interface{}) uintptr {
	return uintptr(unpackEFace(obj).rtype)
}

func IsNil(obj interface{}) bool {
	if obj == nil {
		return true
	}
	return unpackEFace(obj).data == nil
}

func IsNullable(kind reflect.Kind) bool {
	switch kind {
	case reflect.Ptr, reflect.Map, reflect.Chan, reflect.Func, reflect.Slice, reflect.Interface:
		return true
	}
	return false
}

func likePtrKind(kind reflect.Kind) bool {
	switch kind {
	case reflect.Ptr, reflect.Map, reflect.Chan, reflect.Func:
		return true
	}
	return false
}

func likePtrType(typ reflect.Type) bool {
	if likePtrKind(typ.Kind()) {
		return true
	}
	if typ.Kind() == reflect.Struct {
		if typ.NumField() != 1 {
			return false
		}
		return likePtrType(typ.Field(0).Type)
	}
	if typ.Kind() == reflect.Array {
		if typ.Len() != 1 {
			return false
		}
		return likePtrType(typ.Elem())
	}
	return false
}

// NoEscape hides a pointer from escape analysis.  noescape is
// the identity function but escape analysis doesn't think the
// output depends on the input.  noescape is inlined and currently
// compiles down to zero instructions.
// USE CAREFULLY!
//go:nosplit
func NoEscape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}

func UnsafeCastString(str string) []byte {
	stringHeader := (*reflect.StringHeader)(unsafe.Pointer(&str))
	sliceHeader := &reflect.SliceHeader{
		Data: stringHeader.Data,
		Cap: stringHeader.Len,
		Len: stringHeader.Len,
	}
	return *(*[]byte)(unsafe.Pointer(sliceHeader))
}
