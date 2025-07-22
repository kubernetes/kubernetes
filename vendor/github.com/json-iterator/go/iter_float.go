package jsoniter

import (
	"encoding/json"
	"io"
	"math/big"
	"strconv"
	"strings"
	"unsafe"
)

var floatDigits []int8

const invalidCharForNumber = int8(-1)
const endOfNumber = int8(-2)
const dotInNumber = int8(-3)

func init() {
	floatDigits = make([]int8, 256)
	for i := 0; i < len(floatDigits); i++ {
		floatDigits[i] = invalidCharForNumber
	}
	for i := int8('0'); i <= int8('9'); i++ {
		floatDigits[i] = i - int8('0')
	}
	floatDigits[','] = endOfNumber
	floatDigits[']'] = endOfNumber
	floatDigits['}'] = endOfNumber
	floatDigits[' '] = endOfNumber
	floatDigits['\t'] = endOfNumber
	floatDigits['\n'] = endOfNumber
	floatDigits['.'] = dotInNumber
}

// ReadBigFloat read big.Float
func (iter *Iterator) ReadBigFloat() (ret *big.Float) {
	str := iter.readNumberAsString()
	if iter.Error != nil && iter.Error != io.EOF {
		return nil
	}
	prec := 64
	if len(str) > prec {
		prec = len(str)
	}
	val, _, err := big.ParseFloat(str, 10, uint(prec), big.ToZero)
	if err != nil {
		iter.Error = err
		return nil
	}
	return val
}

// ReadBigInt read big.Int
func (iter *Iterator) ReadBigInt() (ret *big.Int) {
	str := iter.readNumberAsString()
	if iter.Error != nil && iter.Error != io.EOF {
		return nil
	}
	ret = big.NewInt(0)
	var success bool
	ret, success = ret.SetString(str, 10)
	if !success {
		iter.ReportError("ReadBigInt", "invalid big int")
		return nil
	}
	return ret
}

//ReadFloat32 read float32
func (iter *Iterator) ReadFloat32() (ret float32) {
	c := iter.nextToken()
	if c == '-' {
		return -iter.readPositiveFloat32()
	}
	iter.unreadByte()
	return iter.readPositiveFloat32()
}

func (iter *Iterator) readPositiveFloat32() (ret float32) {
	i := iter.head
	// first char
	if i == iter.tail {
		return iter.readFloat32SlowPath()
	}
	c := iter.buf[i]
	i++
	ind := floatDigits[c]
	switch ind {
	case invalidCharForNumber:
		return iter.readFloat32SlowPath()
	case endOfNumber:
		iter.ReportError("readFloat32", "empty number")
		return
	case dotInNumber:
		iter.ReportError("readFloat32", "leading dot is invalid")
		return
	case 0:
		if i == iter.tail {
			return iter.readFloat32SlowPath()
		}
		c = iter.buf[i]
		switch c {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			iter.ReportError("readFloat32", "leading zero is invalid")
			return
		}
	}
	value := uint64(ind)
	// chars before dot
non_decimal_loop:
	for ; i < iter.tail; i++ {
		c = iter.buf[i]
		ind := floatDigits[c]
		switch ind {
		case invalidCharForNumber:
			return iter.readFloat32SlowPath()
		case endOfNumber:
			iter.head = i
			return float32(value)
		case dotInNumber:
			break non_decimal_loop
		}
		if value > uint64SafeToMultiple10 {
			return iter.readFloat32SlowPath()
		}
		value = (value << 3) + (value << 1) + uint64(ind) // value = value * 10 + ind;
	}
	// chars after dot
	if c == '.' {
		i++
		decimalPlaces := 0
		if i == iter.tail {
			return iter.readFloat32SlowPath()
		}
		for ; i < iter.tail; i++ {
			c = iter.buf[i]
			ind := floatDigits[c]
			switch ind {
			case endOfNumber:
				if decimalPlaces > 0 && decimalPlaces < len(pow10) {
					iter.head = i
					return float32(float64(value) / float64(pow10[decimalPlaces]))
				}
				// too many decimal places
				return iter.readFloat32SlowPath()
			case invalidCharForNumber, dotInNumber:
				return iter.readFloat32SlowPath()
			}
			decimalPlaces++
			if value > uint64SafeToMultiple10 {
				return iter.readFloat32SlowPath()
			}
			value = (value << 3) + (value << 1) + uint64(ind)
		}
	}
	return iter.readFloat32SlowPath()
}

func (iter *Iterator) readNumberAsString() (ret string) {
	strBuf := [16]byte{}
	str := strBuf[0:0]
load_loop:
	for {
		for i := iter.head; i < iter.tail; i++ {
			c := iter.buf[i]
			switch c {
			case '+', '-', '.', 'e', 'E', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
				str = append(str, c)
				continue
			default:
				iter.head = i
				break load_loop
			}
		}
		if !iter.loadMore() {
			break
		}
	}
	if iter.Error != nil && iter.Error != io.EOF {
		return
	}
	if len(str) == 0 {
		iter.ReportError("readNumberAsString", "invalid number")
	}
	return *(*string)(unsafe.Pointer(&str))
}

func (iter *Iterator) readFloat32SlowPath() (ret float32) {
	str := iter.readNumberAsString()
	if iter.Error != nil && iter.Error != io.EOF {
		return
	}
	errMsg := validateFloat(str)
	if errMsg != "" {
		iter.ReportError("readFloat32SlowPath", errMsg)
		return
	}
	val, err := strconv.ParseFloat(str, 32)
	if err != nil {
		iter.Error = err
		return
	}
	return float32(val)
}

// ReadFloat64 read float64
func (iter *Iterator) ReadFloat64() (ret float64) {
	c := iter.nextToken()
	if c == '-' {
		return -iter.readPositiveFloat64()
	}
	iter.unreadByte()
	return iter.readPositiveFloat64()
}

func (iter *Iterator) readPositiveFloat64() (ret float64) {
	i := iter.head
	// first char
	if i == iter.tail {
		return iter.readFloat64SlowPath()
	}
	c := iter.buf[i]
	i++
	ind := floatDigits[c]
	switch ind {
	case invalidCharForNumber:
		return iter.readFloat64SlowPath()
	case endOfNumber:
		iter.ReportError("readFloat64", "empty number")
		return
	case dotInNumber:
		iter.ReportError("readFloat64", "leading dot is invalid")
		return
	case 0:
		if i == iter.tail {
			return iter.readFloat64SlowPath()
		}
		c = iter.buf[i]
		switch c {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			iter.ReportError("readFloat64", "leading zero is invalid")
			return
		}
	}
	value := uint64(ind)
	// chars before dot
non_decimal_loop:
	for ; i < iter.tail; i++ {
		c = iter.buf[i]
		ind := floatDigits[c]
		switch ind {
		case invalidCharForNumber:
			return iter.readFloat64SlowPath()
		case endOfNumber:
			iter.head = i
			return float64(value)
		case dotInNumber:
			break non_decimal_loop
		}
		if value > uint64SafeToMultiple10 {
			return iter.readFloat64SlowPath()
		}
		value = (value << 3) + (value << 1) + uint64(ind) // value = value * 10 + ind;
	}
	// chars after dot
	if c == '.' {
		i++
		decimalPlaces := 0
		if i == iter.tail {
			return iter.readFloat64SlowPath()
		}
		for ; i < iter.tail; i++ {
			c = iter.buf[i]
			ind := floatDigits[c]
			switch ind {
			case endOfNumber:
				if decimalPlaces > 0 && decimalPlaces < len(pow10) {
					iter.head = i
					return float64(value) / float64(pow10[decimalPlaces])
				}
				// too many decimal places
				return iter.readFloat64SlowPath()
			case invalidCharForNumber, dotInNumber:
				return iter.readFloat64SlowPath()
			}
			decimalPlaces++
			if value > uint64SafeToMultiple10 {
				return iter.readFloat64SlowPath()
			}
			value = (value << 3) + (value << 1) + uint64(ind)
			if value > maxFloat64 {
				return iter.readFloat64SlowPath()
			}
		}
	}
	return iter.readFloat64SlowPath()
}

func (iter *Iterator) readFloat64SlowPath() (ret float64) {
	str := iter.readNumberAsString()
	if iter.Error != nil && iter.Error != io.EOF {
		return
	}
	errMsg := validateFloat(str)
	if errMsg != "" {
		iter.ReportError("readFloat64SlowPath", errMsg)
		return
	}
	val, err := strconv.ParseFloat(str, 64)
	if err != nil {
		iter.Error = err
		return
	}
	return val
}

func validateFloat(str string) string {
	// strconv.ParseFloat is not validating `1.` or `1.e1`
	if len(str) == 0 {
		return "empty number"
	}
	if str[0] == '-' {
		return "-- is not valid"
	}
	dotPos := strings.IndexByte(str, '.')
	if dotPos != -1 {
		if dotPos == len(str)-1 {
			return "dot can not be last character"
		}
		switch str[dotPos+1] {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		default:
			return "missing digit after dot"
		}
	}
	return ""
}

// ReadNumber read json.Number
func (iter *Iterator) ReadNumber() (ret json.Number) {
	return json.Number(iter.readNumberAsString())
}
