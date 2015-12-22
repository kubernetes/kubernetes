package tsm1

import "sync"

var (
	bufPool          sync.Pool
	float64ValuePool sync.Pool
	int64ValuePool   sync.Pool
	boolValuePool    sync.Pool
	stringValuePool  sync.Pool
)

// getBuf returns a buffer with length size from the buffer pool.
func getBuf(size int) []byte {
	x := bufPool.Get()
	if x == nil {
		return make([]byte, size)
	}
	buf := x.([]byte)
	if cap(buf) < size {
		return make([]byte, size)
	}
	return buf[:size]
}

// putBuf returns a buffer to the pool.
func putBuf(buf []byte) {
	bufPool.Put(buf)
}

// getBuf returns a buffer with length size from the buffer pool.
func getFloat64Values(size int) []Value {
	var buf []Value
	x := float64ValuePool.Get()
	if x == nil {
		buf = make([]Value, size)
	} else {
		buf = x.([]Value)
	}
	if cap(buf) < size {
		return make([]Value, size)
	}

	for i, v := range buf {
		if v == nil {
			buf[i] = &FloatValue{}
		}
	}
	return buf[:size]
}

// putBuf returns a buffer to the pool.
func putFloat64Values(buf []Value) {
	float64ValuePool.Put(buf)
}

// getBuf returns a buffer with length size from the buffer pool.
func getInt64Values(size int) []Value {
	var buf []Value
	x := int64ValuePool.Get()
	if x == nil {
		buf = make([]Value, size)
	} else {
		buf = x.([]Value)
	}
	if cap(buf) < size {
		return make([]Value, size)
	}

	for i, v := range buf {
		if v == nil {
			buf[i] = &Int64Value{}
		}
	}
	return buf[:size]
}

// putBuf returns a buffer to the pool.
func putInt64Values(buf []Value) {
	int64ValuePool.Put(buf)
}

// getBuf returns a buffer with length size from the buffer pool.
func getBoolValues(size int) []Value {
	var buf []Value
	x := boolValuePool.Get()
	if x == nil {
		buf = make([]Value, size)
	} else {
		buf = x.([]Value)
	}
	if cap(buf) < size {
		return make([]Value, size)
	}

	for i, v := range buf {
		if v == nil {
			buf[i] = &BoolValue{}
		}
	}
	return buf[:size]
}

// putBuf returns a buffer to the pool.
func putStringValues(buf []Value) {
	stringValuePool.Put(buf)
}

// getBuf returns a buffer with length size from the buffer pool.
func getStringValues(size int) []Value {
	var buf []Value
	x := stringValuePool.Get()
	if x == nil {
		buf = make([]Value, size)
	} else {
		buf = x.([]Value)
	}
	if cap(buf) < size {
		return make([]Value, size)
	}

	for i, v := range buf {
		if v == nil {
			buf[i] = &StringValue{}
		}
	}
	return buf[:size]
}

// putBuf returns a buffer to the pool.
func putBoolValues(buf []Value) {
	boolValuePool.Put(buf)
}
func putValue(buf []Value) {
	if len(buf) > 0 {
		switch buf[0].(type) {
		case *FloatValue:
			putFloat64Values(buf)
		case *Int64Value:
			putInt64Values(buf)
		case *BoolValue:
			putBoolValues(buf)
		case *StringValue:
			putBoolValues(buf)
		}
	}
}
