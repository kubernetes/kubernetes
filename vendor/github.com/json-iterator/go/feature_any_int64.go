package jsoniter

import (
	"strconv"
)

type int64Any struct {
	baseAny
	val int64
}

func (any *int64Any) LastError() error {
	return nil
}

func (any *int64Any) ValueType() ValueType {
	return NumberValue
}

func (any *int64Any) MustBeValid() Any {
	return any
}

func (any *int64Any) ToBool() bool {
	return any.val != 0
}

func (any *int64Any) ToInt() int {
	return int(any.val)
}

func (any *int64Any) ToInt32() int32 {
	return int32(any.val)
}

func (any *int64Any) ToInt64() int64 {
	return any.val
}

func (any *int64Any) ToUint() uint {
	return uint(any.val)
}

func (any *int64Any) ToUint32() uint32 {
	return uint32(any.val)
}

func (any *int64Any) ToUint64() uint64 {
	return uint64(any.val)
}

func (any *int64Any) ToFloat32() float32 {
	return float32(any.val)
}

func (any *int64Any) ToFloat64() float64 {
	return float64(any.val)
}

func (any *int64Any) ToString() string {
	return strconv.FormatInt(any.val, 10)
}

func (any *int64Any) WriteTo(stream *Stream) {
	stream.WriteInt64(any.val)
}

func (any *int64Any) Parse() *Iterator {
	return nil
}

func (any *int64Any) GetInterface() interface{} {
	return any.val
}
