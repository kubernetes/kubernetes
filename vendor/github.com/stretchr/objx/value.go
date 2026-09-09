package objx

import (
	"fmt"
	"strconv"
)

// Value provides methods for extracting interface{} data in various
// types.
type Value struct {
	// data contains the raw data being managed by this Value
	data interface{}
}

// Data returns the raw data contained by this Value
func (v *Value) Data() interface{} {
	return v.data
}

// String returns the value always as a string
func (v *Value) String() string {
	switch {
	case v.IsNil():
		return ""
	case v.IsStr():
		return v.Str()
	case v.IsBool():
		return strconv.FormatBool(v.Bool())
	case v.IsFloat32():
		return strconv.FormatFloat(float64(v.Float32()), 'f', -1, 32)
	case v.IsFloat64():
		return strconv.FormatFloat(v.Float64(), 'f', -1, 64)
	case v.IsInt():
		return strconv.FormatInt(int64(v.Int()), 10)
	case v.IsInt8():
		return strconv.FormatInt(int64(v.Int8()), 10)
	case v.IsInt16():
		return strconv.FormatInt(int64(v.Int16()), 10)
	case v.IsInt32():
		return strconv.FormatInt(int64(v.Int32()), 10)
	case v.IsInt64():
		return strconv.FormatInt(v.Int64(), 10)
	case v.IsUint():
		return strconv.FormatUint(uint64(v.Uint()), 10)
	case v.IsUint8():
		return strconv.FormatUint(uint64(v.Uint8()), 10)
	case v.IsUint16():
		return strconv.FormatUint(uint64(v.Uint16()), 10)
	case v.IsUint32():
		return strconv.FormatUint(uint64(v.Uint32()), 10)
	case v.IsUint64():
		return strconv.FormatUint(v.Uint64(), 10)
	}
	return fmt.Sprintf("%#v", v.Data())
}

// StringSlice returns the value always as a []string
func (v *Value) StringSlice(optionalDefault ...[]string) []string {
	switch {
	case v.IsStrSlice():
		return v.MustStrSlice()
	case v.IsBoolSlice():
		slice := v.MustBoolSlice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatBool(iv)
		}
		return vals
	case v.IsFloat32Slice():
		slice := v.MustFloat32Slice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatFloat(float64(iv), 'f', -1, 32)
		}
		return vals
	case v.IsFloat64Slice():
		slice := v.MustFloat64Slice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatFloat(iv, 'f', -1, 64)
		}
		return vals
	case v.IsIntSlice():
		slice := v.MustIntSlice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatInt(int64(iv), 10)
		}
		return vals
	case v.IsInt8Slice():
		slice := v.MustInt8Slice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatInt(int64(iv), 10)
		}
		return vals
	case v.IsInt16Slice():
		slice := v.MustInt16Slice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatInt(int64(iv), 10)
		}
		return vals
	case v.IsInt32Slice():
		slice := v.MustInt32Slice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatInt(int64(iv), 10)
		}
		return vals
	case v.IsInt64Slice():
		slice := v.MustInt64Slice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatInt(iv, 10)
		}
		return vals
	case v.IsUintSlice():
		slice := v.MustUintSlice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatUint(uint64(iv), 10)
		}
		return vals
	case v.IsUint8Slice():
		slice := v.MustUint8Slice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatUint(uint64(iv), 10)
		}
		return vals
	case v.IsUint16Slice():
		slice := v.MustUint16Slice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatUint(uint64(iv), 10)
		}
		return vals
	case v.IsUint32Slice():
		slice := v.MustUint32Slice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatUint(uint64(iv), 10)
		}
		return vals
	case v.IsUint64Slice():
		slice := v.MustUint64Slice()
		vals := make([]string, len(slice))
		for i, iv := range slice {
			vals[i] = strconv.FormatUint(iv, 10)
		}
		return vals
	}
	if len(optionalDefault) == 1 {
		return optionalDefault[0]
	}

	return []string{}
}
