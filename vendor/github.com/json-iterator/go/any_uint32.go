package jsoniter

import (
	"strconv"
)

type uint32Any struct {
	baseAny
	val uint32
}

func (any *uint32Any) LastError() error {
	return nil
}

func (any *uint32Any) ValueType() ValueType {
	return NumberValue
}

func (any *uint32Any) MustBeValid() Any {
	return any
}

func (any *uint32Any) ToBool() bool {
	return any.val != 0
}

func (any *uint32Any) ToInt() int {
	return int(any.val)
}

func (any *uint32Any) ToInt32() int32 {
	return int32(any.val)
}

func (any *uint32Any) ToInt64() int64 {
	return int64(any.val)
}

func (any *uint32Any) ToUint() uint {
	return uint(any.val)
}

func (any *uint32Any) ToUint32() uint32 {
	return any.val
}

func (any *uint32Any) ToUint64() uint64 {
	return uint64(any.val)
}

func (any *uint32Any) ToFloat32() float32 {
	return float32(any.val)
}

func (any *uint32Any) ToFloat64() float64 {
	return float64(any.val)
}

func (any *uint32Any) ToString() string {
	return strconv.FormatInt(int64(any.val), 10)
}

func (any *uint32Any) WriteTo(stream *Stream) {
	stream.WriteUint32(any.val)
}

func (any *uint32Any) Parse() *Iterator {
	return nil
}

func (any *uint32Any) GetInterface() interface{} {
	return any.val
}
