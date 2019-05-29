package jsoniter

import (
	"reflect"
	"unsafe"
)

type objectLazyAny struct {
	baseAny
	cfg *frozenConfig
	buf []byte
	err error
}

func (any *objectLazyAny) ValueType() ValueType {
	return ObjectValue
}

func (any *objectLazyAny) MustBeValid() Any {
	return any
}

func (any *objectLazyAny) LastError() error {
	return any.err
}

func (any *objectLazyAny) ToBool() bool {
	return true
}

func (any *objectLazyAny) ToInt() int {
	return 0
}

func (any *objectLazyAny) ToInt32() int32 {
	return 0
}

func (any *objectLazyAny) ToInt64() int64 {
	return 0
}

func (any *objectLazyAny) ToUint() uint {
	return 0
}

func (any *objectLazyAny) ToUint32() uint32 {
	return 0
}

func (any *objectLazyAny) ToUint64() uint64 {
	return 0
}

func (any *objectLazyAny) ToFloat32() float32 {
	return 0
}

func (any *objectLazyAny) ToFloat64() float64 {
	return 0
}

func (any *objectLazyAny) ToString() string {
	return *(*string)(unsafe.Pointer(&any.buf))
}

func (any *objectLazyAny) ToVal(obj interface{}) {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	iter.ReadVal(obj)
}

func (any *objectLazyAny) Get(path ...interface{}) Any {
	if len(path) == 0 {
		return any
	}
	switch firstPath := path[0].(type) {
	case string:
		iter := any.cfg.BorrowIterator(any.buf)
		defer any.cfg.ReturnIterator(iter)
		valueBytes := locateObjectField(iter, firstPath)
		if valueBytes == nil {
			return newInvalidAny(path)
		}
		iter.ResetBytes(valueBytes)
		return locatePath(iter, path[1:])
	case int32:
		if '*' == firstPath {
			mappedAll := map[string]Any{}
			iter := any.cfg.BorrowIterator(any.buf)
			defer any.cfg.ReturnIterator(iter)
			iter.ReadMapCB(func(iter *Iterator, field string) bool {
				mapped := locatePath(iter, path[1:])
				if mapped.ValueType() != InvalidValue {
					mappedAll[field] = mapped
				}
				return true
			})
			return wrapMap(mappedAll)
		}
		return newInvalidAny(path)
	default:
		return newInvalidAny(path)
	}
}

func (any *objectLazyAny) Keys() []string {
	keys := []string{}
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	iter.ReadMapCB(func(iter *Iterator, field string) bool {
		iter.Skip()
		keys = append(keys, field)
		return true
	})
	return keys
}

func (any *objectLazyAny) Size() int {
	size := 0
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	iter.ReadObjectCB(func(iter *Iterator, field string) bool {
		iter.Skip()
		size++
		return true
	})
	return size
}

func (any *objectLazyAny) WriteTo(stream *Stream) {
	stream.Write(any.buf)
}

func (any *objectLazyAny) GetInterface() interface{} {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	return iter.Read()
}

type objectAny struct {
	baseAny
	err error
	val reflect.Value
}

func wrapStruct(val interface{}) *objectAny {
	return &objectAny{baseAny{}, nil, reflect.ValueOf(val)}
}

func (any *objectAny) ValueType() ValueType {
	return ObjectValue
}

func (any *objectAny) MustBeValid() Any {
	return any
}

func (any *objectAny) Parse() *Iterator {
	return nil
}

func (any *objectAny) LastError() error {
	return any.err
}

func (any *objectAny) ToBool() bool {
	return any.val.NumField() != 0
}

func (any *objectAny) ToInt() int {
	return 0
}

func (any *objectAny) ToInt32() int32 {
	return 0
}

func (any *objectAny) ToInt64() int64 {
	return 0
}

func (any *objectAny) ToUint() uint {
	return 0
}

func (any *objectAny) ToUint32() uint32 {
	return 0
}

func (any *objectAny) ToUint64() uint64 {
	return 0
}

func (any *objectAny) ToFloat32() float32 {
	return 0
}

func (any *objectAny) ToFloat64() float64 {
	return 0
}

func (any *objectAny) ToString() string {
	str, err := MarshalToString(any.val.Interface())
	any.err = err
	return str
}

func (any *objectAny) Get(path ...interface{}) Any {
	if len(path) == 0 {
		return any
	}
	switch firstPath := path[0].(type) {
	case string:
		field := any.val.FieldByName(firstPath)
		if !field.IsValid() {
			return newInvalidAny(path)
		}
		return Wrap(field.Interface())
	case int32:
		if '*' == firstPath {
			mappedAll := map[string]Any{}
			for i := 0; i < any.val.NumField(); i++ {
				field := any.val.Field(i)
				if field.CanInterface() {
					mapped := Wrap(field.Interface()).Get(path[1:]...)
					if mapped.ValueType() != InvalidValue {
						mappedAll[any.val.Type().Field(i).Name] = mapped
					}
				}
			}
			return wrapMap(mappedAll)
		}
		return newInvalidAny(path)
	default:
		return newInvalidAny(path)
	}
}

func (any *objectAny) Keys() []string {
	keys := make([]string, 0, any.val.NumField())
	for i := 0; i < any.val.NumField(); i++ {
		keys = append(keys, any.val.Type().Field(i).Name)
	}
	return keys
}

func (any *objectAny) Size() int {
	return any.val.NumField()
}

func (any *objectAny) WriteTo(stream *Stream) {
	stream.WriteVal(any.val)
}

func (any *objectAny) GetInterface() interface{} {
	return any.val.Interface()
}

type mapAny struct {
	baseAny
	err error
	val reflect.Value
}

func wrapMap(val interface{}) *mapAny {
	return &mapAny{baseAny{}, nil, reflect.ValueOf(val)}
}

func (any *mapAny) ValueType() ValueType {
	return ObjectValue
}

func (any *mapAny) MustBeValid() Any {
	return any
}

func (any *mapAny) Parse() *Iterator {
	return nil
}

func (any *mapAny) LastError() error {
	return any.err
}

func (any *mapAny) ToBool() bool {
	return true
}

func (any *mapAny) ToInt() int {
	return 0
}

func (any *mapAny) ToInt32() int32 {
	return 0
}

func (any *mapAny) ToInt64() int64 {
	return 0
}

func (any *mapAny) ToUint() uint {
	return 0
}

func (any *mapAny) ToUint32() uint32 {
	return 0
}

func (any *mapAny) ToUint64() uint64 {
	return 0
}

func (any *mapAny) ToFloat32() float32 {
	return 0
}

func (any *mapAny) ToFloat64() float64 {
	return 0
}

func (any *mapAny) ToString() string {
	str, err := MarshalToString(any.val.Interface())
	any.err = err
	return str
}

func (any *mapAny) Get(path ...interface{}) Any {
	if len(path) == 0 {
		return any
	}
	switch firstPath := path[0].(type) {
	case int32:
		if '*' == firstPath {
			mappedAll := map[string]Any{}
			for _, key := range any.val.MapKeys() {
				keyAsStr := key.String()
				element := Wrap(any.val.MapIndex(key).Interface())
				mapped := element.Get(path[1:]...)
				if mapped.ValueType() != InvalidValue {
					mappedAll[keyAsStr] = mapped
				}
			}
			return wrapMap(mappedAll)
		}
		return newInvalidAny(path)
	default:
		value := any.val.MapIndex(reflect.ValueOf(firstPath))
		if !value.IsValid() {
			return newInvalidAny(path)
		}
		return Wrap(value.Interface())
	}
}

func (any *mapAny) Keys() []string {
	keys := make([]string, 0, any.val.Len())
	for _, key := range any.val.MapKeys() {
		keys = append(keys, key.String())
	}
	return keys
}

func (any *mapAny) Size() int {
	return any.val.Len()
}

func (any *mapAny) WriteTo(stream *Stream) {
	stream.WriteVal(any.val)
}

func (any *mapAny) GetInterface() interface{} {
	return any.val.Interface()
}
