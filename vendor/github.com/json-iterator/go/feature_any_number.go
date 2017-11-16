package jsoniter

import "unsafe"

type numberLazyAny struct {
	baseAny
	cfg *frozenConfig
	buf []byte
	err error
}

func (any *numberLazyAny) ValueType() ValueType {
	return NumberValue
}

func (any *numberLazyAny) MustBeValid() Any {
	return any
}

func (any *numberLazyAny) LastError() error {
	return any.err
}

func (any *numberLazyAny) ToBool() bool {
	return any.ToFloat64() != 0
}

func (any *numberLazyAny) ToInt() int {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	val := iter.ReadInt()
	any.err = iter.Error
	return val
}

func (any *numberLazyAny) ToInt32() int32 {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	val := iter.ReadInt32()
	any.err = iter.Error
	return val
}

func (any *numberLazyAny) ToInt64() int64 {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	val := iter.ReadInt64()
	any.err = iter.Error
	return val
}

func (any *numberLazyAny) ToUint() uint {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	val := iter.ReadUint()
	any.err = iter.Error
	return val
}

func (any *numberLazyAny) ToUint32() uint32 {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	val := iter.ReadUint32()
	any.err = iter.Error
	return val
}

func (any *numberLazyAny) ToUint64() uint64 {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	val := iter.ReadUint64()
	any.err = iter.Error
	return val
}

func (any *numberLazyAny) ToFloat32() float32 {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	val := iter.ReadFloat32()
	any.err = iter.Error
	return val
}

func (any *numberLazyAny) ToFloat64() float64 {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	val := iter.ReadFloat64()
	any.err = iter.Error
	return val
}

func (any *numberLazyAny) ToString() string {
	return *(*string)(unsafe.Pointer(&any.buf))
}

func (any *numberLazyAny) WriteTo(stream *Stream) {
	stream.Write(any.buf)
}

func (any *numberLazyAny) GetInterface() interface{} {
	iter := any.cfg.BorrowIterator(any.buf)
	defer any.cfg.ReturnIterator(iter)
	return iter.Read()
}
