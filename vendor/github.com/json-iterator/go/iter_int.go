package jsoniter

import (
	"math"
	"strconv"
)

var intDigits []int8

const uint32SafeToMultiply10 = uint32(0xffffffff)/10 - 1
const uint64SafeToMultiple10 = uint64(0xffffffffffffffff)/10 - 1
const maxFloat64 = 1<<53 - 1

func init() {
	intDigits = make([]int8, 256)
	for i := 0; i < len(intDigits); i++ {
		intDigits[i] = invalidCharForNumber
	}
	for i := int8('0'); i <= int8('9'); i++ {
		intDigits[i] = i - int8('0')
	}
}

// ReadUint read uint
func (iter *Iterator) ReadUint() uint {
	if strconv.IntSize == 32 {
		return uint(iter.ReadUint32())
	}
	return uint(iter.ReadUint64())
}

// ReadInt read int
func (iter *Iterator) ReadInt() int {
	if strconv.IntSize == 32 {
		return int(iter.ReadInt32())
	}
	return int(iter.ReadInt64())
}

// ReadInt8 read int8
func (iter *Iterator) ReadInt8() (ret int8) {
	c := iter.nextToken()
	if c == '-' {
		val := iter.readUint32(iter.readByte())
		if val > math.MaxInt8+1 {
			iter.ReportError("ReadInt8", "overflow: "+strconv.FormatInt(int64(val), 10))
			return
		}
		return -int8(val)
	}
	val := iter.readUint32(c)
	if val > math.MaxInt8 {
		iter.ReportError("ReadInt8", "overflow: "+strconv.FormatInt(int64(val), 10))
		return
	}
	return int8(val)
}

// ReadUint8 read uint8
func (iter *Iterator) ReadUint8() (ret uint8) {
	val := iter.readUint32(iter.nextToken())
	if val > math.MaxUint8 {
		iter.ReportError("ReadUint8", "overflow: "+strconv.FormatInt(int64(val), 10))
		return
	}
	return uint8(val)
}

// ReadInt16 read int16
func (iter *Iterator) ReadInt16() (ret int16) {
	c := iter.nextToken()
	if c == '-' {
		val := iter.readUint32(iter.readByte())
		if val > math.MaxInt16+1 {
			iter.ReportError("ReadInt16", "overflow: "+strconv.FormatInt(int64(val), 10))
			return
		}
		return -int16(val)
	}
	val := iter.readUint32(c)
	if val > math.MaxInt16 {
		iter.ReportError("ReadInt16", "overflow: "+strconv.FormatInt(int64(val), 10))
		return
	}
	return int16(val)
}

// ReadUint16 read uint16
func (iter *Iterator) ReadUint16() (ret uint16) {
	val := iter.readUint32(iter.nextToken())
	if val > math.MaxUint16 {
		iter.ReportError("ReadUint16", "overflow: "+strconv.FormatInt(int64(val), 10))
		return
	}
	return uint16(val)
}

// ReadInt32 read int32
func (iter *Iterator) ReadInt32() (ret int32) {
	c := iter.nextToken()
	if c == '-' {
		val := iter.readUint32(iter.readByte())
		if val > math.MaxInt32+1 {
			iter.ReportError("ReadInt32", "overflow: "+strconv.FormatInt(int64(val), 10))
			return
		}
		return -int32(val)
	}
	val := iter.readUint32(c)
	if val > math.MaxInt32 {
		iter.ReportError("ReadInt32", "overflow: "+strconv.FormatInt(int64(val), 10))
		return
	}
	return int32(val)
}

// ReadUint32 read uint32
func (iter *Iterator) ReadUint32() (ret uint32) {
	return iter.readUint32(iter.nextToken())
}

func (iter *Iterator) readUint32(c byte) (ret uint32) {
	ind := intDigits[c]
	if ind == 0 {
		iter.assertInteger()
		return 0 // single zero
	}
	if ind == invalidCharForNumber {
		iter.ReportError("readUint32", "unexpected character: "+string([]byte{byte(ind)}))
		return
	}
	value := uint32(ind)
	if iter.tail-iter.head > 10 {
		i := iter.head
		ind2 := intDigits[iter.buf[i]]
		if ind2 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value
		}
		i++
		ind3 := intDigits[iter.buf[i]]
		if ind3 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*10 + uint32(ind2)
		}
		//iter.head = i + 1
		//value = value * 100 + uint32(ind2) * 10 + uint32(ind3)
		i++
		ind4 := intDigits[iter.buf[i]]
		if ind4 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*100 + uint32(ind2)*10 + uint32(ind3)
		}
		i++
		ind5 := intDigits[iter.buf[i]]
		if ind5 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*1000 + uint32(ind2)*100 + uint32(ind3)*10 + uint32(ind4)
		}
		i++
		ind6 := intDigits[iter.buf[i]]
		if ind6 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*10000 + uint32(ind2)*1000 + uint32(ind3)*100 + uint32(ind4)*10 + uint32(ind5)
		}
		i++
		ind7 := intDigits[iter.buf[i]]
		if ind7 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*100000 + uint32(ind2)*10000 + uint32(ind3)*1000 + uint32(ind4)*100 + uint32(ind5)*10 + uint32(ind6)
		}
		i++
		ind8 := intDigits[iter.buf[i]]
		if ind8 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*1000000 + uint32(ind2)*100000 + uint32(ind3)*10000 + uint32(ind4)*1000 + uint32(ind5)*100 + uint32(ind6)*10 + uint32(ind7)
		}
		i++
		ind9 := intDigits[iter.buf[i]]
		value = value*10000000 + uint32(ind2)*1000000 + uint32(ind3)*100000 + uint32(ind4)*10000 + uint32(ind5)*1000 + uint32(ind6)*100 + uint32(ind7)*10 + uint32(ind8)
		iter.head = i
		if ind9 == invalidCharForNumber {
			iter.assertInteger()
			return value
		}
	}
	for {
		for i := iter.head; i < iter.tail; i++ {
			ind = intDigits[iter.buf[i]]
			if ind == invalidCharForNumber {
				iter.head = i
				iter.assertInteger()
				return value
			}
			if value > uint32SafeToMultiply10 {
				value2 := (value << 3) + (value << 1) + uint32(ind)
				if value2 < value {
					iter.ReportError("readUint32", "overflow")
					return
				}
				value = value2
				continue
			}
			value = (value << 3) + (value << 1) + uint32(ind)
		}
		if !iter.loadMore() {
			iter.assertInteger()
			return value
		}
	}
}

// ReadInt64 read int64
func (iter *Iterator) ReadInt64() (ret int64) {
	c := iter.nextToken()
	if c == '-' {
		val := iter.readUint64(iter.readByte())
		if val > math.MaxInt64+1 {
			iter.ReportError("ReadInt64", "overflow: "+strconv.FormatUint(uint64(val), 10))
			return
		}
		return -int64(val)
	}
	val := iter.readUint64(c)
	if val > math.MaxInt64 {
		iter.ReportError("ReadInt64", "overflow: "+strconv.FormatUint(uint64(val), 10))
		return
	}
	return int64(val)
}

// ReadUint64 read uint64
func (iter *Iterator) ReadUint64() uint64 {
	return iter.readUint64(iter.nextToken())
}

func (iter *Iterator) readUint64(c byte) (ret uint64) {
	ind := intDigits[c]
	if ind == 0 {
		iter.assertInteger()
		return 0 // single zero
	}
	if ind == invalidCharForNumber {
		iter.ReportError("readUint64", "unexpected character: "+string([]byte{byte(ind)}))
		return
	}
	value := uint64(ind)
	if iter.tail-iter.head > 10 {
		i := iter.head
		ind2 := intDigits[iter.buf[i]]
		if ind2 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value
		}
		i++
		ind3 := intDigits[iter.buf[i]]
		if ind3 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*10 + uint64(ind2)
		}
		//iter.head = i + 1
		//value = value * 100 + uint32(ind2) * 10 + uint32(ind3)
		i++
		ind4 := intDigits[iter.buf[i]]
		if ind4 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*100 + uint64(ind2)*10 + uint64(ind3)
		}
		i++
		ind5 := intDigits[iter.buf[i]]
		if ind5 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*1000 + uint64(ind2)*100 + uint64(ind3)*10 + uint64(ind4)
		}
		i++
		ind6 := intDigits[iter.buf[i]]
		if ind6 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*10000 + uint64(ind2)*1000 + uint64(ind3)*100 + uint64(ind4)*10 + uint64(ind5)
		}
		i++
		ind7 := intDigits[iter.buf[i]]
		if ind7 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*100000 + uint64(ind2)*10000 + uint64(ind3)*1000 + uint64(ind4)*100 + uint64(ind5)*10 + uint64(ind6)
		}
		i++
		ind8 := intDigits[iter.buf[i]]
		if ind8 == invalidCharForNumber {
			iter.head = i
			iter.assertInteger()
			return value*1000000 + uint64(ind2)*100000 + uint64(ind3)*10000 + uint64(ind4)*1000 + uint64(ind5)*100 + uint64(ind6)*10 + uint64(ind7)
		}
		i++
		ind9 := intDigits[iter.buf[i]]
		value = value*10000000 + uint64(ind2)*1000000 + uint64(ind3)*100000 + uint64(ind4)*10000 + uint64(ind5)*1000 + uint64(ind6)*100 + uint64(ind7)*10 + uint64(ind8)
		iter.head = i
		if ind9 == invalidCharForNumber {
			iter.assertInteger()
			return value
		}
	}
	for {
		for i := iter.head; i < iter.tail; i++ {
			ind = intDigits[iter.buf[i]]
			if ind == invalidCharForNumber {
				iter.head = i
				iter.assertInteger()
				return value
			}
			if value > uint64SafeToMultiple10 {
				value2 := (value << 3) + (value << 1) + uint64(ind)
				if value2 < value {
					iter.ReportError("readUint64", "overflow")
					return
				}
				value = value2
				continue
			}
			value = (value << 3) + (value << 1) + uint64(ind)
		}
		if !iter.loadMore() {
			iter.assertInteger()
			return value
		}
	}
}

func (iter *Iterator) assertInteger() {
	if iter.head < iter.tail && iter.buf[iter.head] == '.' {
		iter.ReportError("assertInteger", "can not decode float as int")
	}
}
