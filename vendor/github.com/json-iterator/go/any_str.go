package jsoniter

import (
	"fmt"
	"strconv"
)

type stringAny struct {
	baseAny
	val string
}

func (any *stringAny) Get(path ...interface{}) Any {
	if len(path) == 0 {
		return any
	}
	return &invalidAny{baseAny{}, fmt.Errorf("GetIndex %v from simple value", path)}
}

func (any *stringAny) Parse() *Iterator {
	return nil
}

func (any *stringAny) ValueType() ValueType {
	return StringValue
}

func (any *stringAny) MustBeValid() Any {
	return any
}

func (any *stringAny) LastError() error {
	return nil
}

func (any *stringAny) ToBool() bool {
	str := any.ToString()
	if str == "0" {
		return false
	}
	for _, c := range str {
		switch c {
		case ' ', '\n', '\r', '\t':
		default:
			return true
		}
	}
	return false
}

func (any *stringAny) ToInt() int {
	return int(any.ToInt64())

}

func (any *stringAny) ToInt32() int32 {
	return int32(any.ToInt64())
}

func (any *stringAny) ToInt64() int64 {
	if any.val == "" {
		return 0
	}

	flag := 1
	startPos := 0
	if any.val[0] == '+' || any.val[0] == '-' {
		startPos = 1
	}

	if any.val[0] == '-' {
		flag = -1
	}

	endPos := startPos
	for i := startPos; i < len(any.val); i++ {
		if any.val[i] >= '0' && any.val[i] <= '9' {
			endPos = i + 1
		} else {
			break
		}
	}
	parsed, _ := strconv.ParseInt(any.val[startPos:endPos], 10, 64)
	return int64(flag) * parsed
}

func (any *stringAny) ToUint() uint {
	return uint(any.ToUint64())
}

func (any *stringAny) ToUint32() uint32 {
	return uint32(any.ToUint64())
}

func (any *stringAny) ToUint64() uint64 {
	if any.val == "" {
		return 0
	}

	startPos := 0

	if any.val[0] == '-' {
		return 0
	}
	if any.val[0] == '+' {
		startPos = 1
	}

	endPos := startPos
	for i := startPos; i < len(any.val); i++ {
		if any.val[i] >= '0' && any.val[i] <= '9' {
			endPos = i + 1
		} else {
			break
		}
	}
	parsed, _ := strconv.ParseUint(any.val[startPos:endPos], 10, 64)
	return parsed
}

func (any *stringAny) ToFloat32() float32 {
	return float32(any.ToFloat64())
}

func (any *stringAny) ToFloat64() float64 {
	if len(any.val) == 0 {
		return 0
	}

	// first char invalid
	if any.val[0] != '+' && any.val[0] != '-' && (any.val[0] > '9' || any.val[0] < '0') {
		return 0
	}

	// extract valid num expression from string
	// eg 123true => 123, -12.12xxa => -12.12
	endPos := 1
	for i := 1; i < len(any.val); i++ {
		if any.val[i] == '.' || any.val[i] == 'e' || any.val[i] == 'E' || any.val[i] == '+' || any.val[i] == '-' {
			endPos = i + 1
			continue
		}

		// end position is the first char which is not digit
		if any.val[i] >= '0' && any.val[i] <= '9' {
			endPos = i + 1
		} else {
			endPos = i
			break
		}
	}
	parsed, _ := strconv.ParseFloat(any.val[:endPos], 64)
	return parsed
}

func (any *stringAny) ToString() string {
	return any.val
}

func (any *stringAny) WriteTo(stream *Stream) {
	stream.WriteString(any.val)
}

func (any *stringAny) GetInterface() interface{} {
	return any.val
}
