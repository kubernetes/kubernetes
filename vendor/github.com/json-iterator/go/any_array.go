package jsoniter

import (
	"reflect"
	"unsafe"
)

type arrayLazyAny struct {
	baseAny
	cfg *frozenConfig
	buf []byte
	err error
}

func (any *arrayLazyAny) ValueType() ValueType {
	return ArrayValue
}

func (any *arrayLazyAny) MustBeValid() Any {
	return any
}

func (any *arrayLazyAny) LastError() error {
	return any.err
}

func (any *arrayLazyAny) ToBool() bool {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	return iter.ReadArray()
}

func (any *arrayLazyAny) ToInt() int {
	if any.ToBool() {
		return 1
	}
	return 0
}

func (any *arrayLazyAny) ToInt32() int32 {
	if any.ToBool() {
		return 1
	}
	return 0
}

func (any *arrayLazyAny) ToInt64() int64 {
	if any.ToBool() {
		return 1
	}
	return 0
}

func (any *arrayLazyAny) ToUint() uint {
	if any.ToBool() {
		return 1
	}
	return 0
}

func (any *arrayLazyAny) ToUint32() uint32 {
	if any.ToBool() {
		return 1
	}
	return 0
}

func (any *arrayLazyAny) ToUint64() uint64 {
	if any.ToBool() {
		return 1
	}
	return 0
}

func (any *arrayLazyAny) ToFloat32() float32 {
	if any.ToBool() {
		return 1
	}
	return 0
}

func (any *arrayLazyAny) ToFloat64() float64 {
	if any.ToBool() {
		return 1
	}
	return 0
}

func (any *arrayLazyAny) ToString() string {
	return *(*string)(unsafe.Pointer(&any.buf))
}

func (any *arrayLazyAny) ToVal(val interface{}) {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	iter.ReadVal(val)
}

func (any *arrayLazyAny) Get(path ...interface{}) Any {
	if len(path) == 0 {
		return any
	}
	switch firstPath := path[0].(type) {
	case int:
		iter := any.cfg.BorrowIterator(any.buf)
		defer any.cfg.ReturnIterator(iter)
		valueBytes := locateArrayElement(iter, firstPath)
		if valueBytes == nil {
			return newInvalidAny(path)
		}
		iter.ResetBytes(valueBytes)
		return locatePath(iter, path[1:])
	case int32:
		if '*' == firstPath {
			iter := any.cfg.BorrowIterator(any.buf)
			defer any.cfg.ReturnIterator(iter)
			arr := make([]Any, 0)
			iter.ReadArrayCB(func(iter *Iterator) bool {
				found := iter.readAny().Get(path[1:]...)
				if found.ValueType() != InvalidValue {
					arr = append(arr, found)
				}
				return true
			})
			return wrapArray(arr)
		}
		return newInvalidAny(path)
	default:
		return newInvalidAny(path)
	}
}

func (any *arrayLazyAny) Size() int {
	size := 0
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	iter.ReadArrayCB(func(iter *Iterator) bool {
		size++
		iter.Skip()
		return true
	})
	return size
}

func (any *arrayLazyAny) WriteTo(stream *Stream) {
	stream.Write(any.buf)
}

func (any *arrayLazyAny) GetInterface() interface{} {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	return iter.Read()
}

type arrayAny struct {
	baseAny
	val reflect.Value
}

func wrapArray(val interface{}) *arrayAny {
	return &arrayAny{baseAny{}, reflect.ValueOf(val)}
}

func (any *arrayAny) ValueType() ValueType {
	return ArrayValue
}

func (any *arrayAny) MustBeValid() Any {
	return any
}

func (any *arrayAny) LastError() error {
	return nil
}

func (any *arrayAny) ToBool() bool {
	return any.val.Len() != 0
}

func (any *arrayAny) ToInt() int {
	if any.val.Len() == 0 {
		return 0
	}
	return 1
}

func (any *arrayAny) ToInt32() int32 {
	if any.val.Len() == 0 {
		return 0
	}
	return 1
}

func (any *arrayAny) ToInt64() int64 {
	if any.val.Len() == 0 {
		return 0
	}
	return 1
}

func (any *arrayAny) ToUint() uint {
	if any.val.Len() == 0 {
		return 0
	}
	return 1
}

func (any *arrayAny) ToUint32() uint32 {
	if any.val.Len() == 0 {
		return 0
	}
	return 1
}

func (any *arrayAny) ToUint64() uint64 {
	if any.val.Len() == 0 {
		return 0
	}
	return 1
}

func (any *arrayAny) ToFloat32() float32 {
	if any.val.Len() == 0 {
		return 0
	}
	return 1
}

func (any *arrayAny) ToFloat64() float64 {
	if any.val.Len() == 0 {
		return 0
	}
	return 1
}

func (any *arrayAny) ToString() string {
	str, _ := MarshalToString(any.val.Interface())
	return str
}

func (any *arrayAny) Get(path ...interface{}) Any {
	if len(path) == 0 {
		return any
	}
	switch firstPath := path[0].(type) {
	case int:
		if firstPath < 0 || firstPath >= any.val.Len() {
			return newInvalidAny(path)
		}
		return Wrap(any.val.Index(firstPath).Interface())
	case int32:
		if '*' == firstPath {
			mappedAll := make([]Any, 0)
			for i := 0; i < any.val.Len(); i++ {
				mapped := Wrap(any.val.Index(i).Interface()).Get(path[1:]...)
				if mapped.ValueType() != InvalidValue {
					mappedAll = append(mappedAll, mapped)
				}
			}
			return wrapArray(mappedAll)
		}
		return newInvalidAny(path)
	default:
		return newInvalidAny(path)
	}
}

func (any *arrayAny) Size() int {
	return any.val.Len()
}

func (any *arrayAny) WriteTo(stream *Stream) {
	stream.WriteVal(any.val)
}

func (any *arrayAny) GetInterface() interface{} {
	return any.val.Interface()
}
