package jsoniter

import (
	"fmt"
	"math"
	"strconv"
)

var pow10 []uint64

func init() {
	pow10 = []uint64{1, 10, 100, 1000, 10000, 100000, 1000000}
}

// WriteFloat32 write float32 to stream
func (stream *Stream) WriteFloat32(val float32) {
	if math.IsInf(float64(val), 0) || math.IsNaN(float64(val)) {
		stream.Error = fmt.Errorf("unsupported value: %f", val)
		return
	}
	abs := math.Abs(float64(val))
	fmt := byte('f')
	// Note: Must use float32 comparisons for underlying float32 value to get precise cutoffs right.
	if abs != 0 {
		if float32(abs) < 1e-6 || float32(abs) >= 1e21 {
			fmt = 'e'
		}
	}
	stream.buf = strconv.AppendFloat(stream.buf, float64(val), fmt, -1, 32)
}

// WriteFloat32Lossy write float32 to stream with ONLY 6 digits precision although much much faster
func (stream *Stream) WriteFloat32Lossy(val float32) {
	if math.IsInf(float64(val), 0) || math.IsNaN(float64(val)) {
		stream.Error = fmt.Errorf("unsupported value: %f", val)
		return
	}
	if val < 0 {
		stream.writeByte('-')
		val = -val
	}
	if val > 0x4ffffff {
		stream.WriteFloat32(val)
		return
	}
	precision := 6
	exp := uint64(1000000) // 6
	lval := uint64(float64(val)*float64(exp) + 0.5)
	stream.WriteUint64(lval / exp)
	fval := lval % exp
	if fval == 0 {
		return
	}
	stream.writeByte('.')
	for p := precision - 1; p > 0 && fval < pow10[p]; p-- {
		stream.writeByte('0')
	}
	stream.WriteUint64(fval)
	for stream.buf[len(stream.buf)-1] == '0' {
		stream.buf = stream.buf[:len(stream.buf)-1]
	}
}

// WriteFloat64 write float64 to stream
func (stream *Stream) WriteFloat64(val float64) {
	if math.IsInf(val, 0) || math.IsNaN(val) {
		stream.Error = fmt.Errorf("unsupported value: %f", val)
		return
	}
	abs := math.Abs(val)
	fmt := byte('f')
	// Note: Must use float32 comparisons for underlying float32 value to get precise cutoffs right.
	if abs != 0 {
		if abs < 1e-6 || abs >= 1e21 {
			fmt = 'e'
		}
	}
	stream.buf = strconv.AppendFloat(stream.buf, float64(val), fmt, -1, 64)
}

// WriteFloat64Lossy write float64 to stream with ONLY 6 digits precision although much much faster
func (stream *Stream) WriteFloat64Lossy(val float64) {
	if math.IsInf(val, 0) || math.IsNaN(val) {
		stream.Error = fmt.Errorf("unsupported value: %f", val)
		return
	}
	if val < 0 {
		stream.writeByte('-')
		val = -val
	}
	if val > 0x4ffffff {
		stream.WriteFloat64(val)
		return
	}
	precision := 6
	exp := uint64(1000000) // 6
	lval := uint64(val*float64(exp) + 0.5)
	stream.WriteUint64(lval / exp)
	fval := lval % exp
	if fval == 0 {
		return
	}
	stream.writeByte('.')
	for p := precision - 1; p > 0 && fval < pow10[p]; p-- {
		stream.writeByte('0')
	}
	stream.WriteUint64(fval)
	for stream.buf[len(stream.buf)-1] == '0' {
		stream.buf = stream.buf[:len(stream.buf)-1]
	}
}
