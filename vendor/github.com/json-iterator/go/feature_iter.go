package jsoniter

import (
	"encoding/json"
	"fmt"
	"io"
)

// ValueType the type for JSON element
type ValueType int

const (
	// InvalidValue invalid JSON element
	InvalidValue ValueType = iota
	// StringValue JSON element "string"
	StringValue
	// NumberValue JSON element 100 or 0.10
	NumberValue
	// NilValue JSON element null
	NilValue
	// BoolValue JSON element true or false
	BoolValue
	// ArrayValue JSON element []
	ArrayValue
	// ObjectValue JSON element {}
	ObjectValue
)

var hexDigits []byte
var valueTypes []ValueType

func init() {
	hexDigits = make([]byte, 256)
	for i := 0; i < len(hexDigits); i++ {
		hexDigits[i] = 255
	}
	for i := '0'; i <= '9'; i++ {
		hexDigits[i] = byte(i - '0')
	}
	for i := 'a'; i <= 'f'; i++ {
		hexDigits[i] = byte((i - 'a') + 10)
	}
	for i := 'A'; i <= 'F'; i++ {
		hexDigits[i] = byte((i - 'A') + 10)
	}
	valueTypes = make([]ValueType, 256)
	for i := 0; i < len(valueTypes); i++ {
		valueTypes[i] = InvalidValue
	}
	valueTypes['"'] = StringValue
	valueTypes['-'] = NumberValue
	valueTypes['0'] = NumberValue
	valueTypes['1'] = NumberValue
	valueTypes['2'] = NumberValue
	valueTypes['3'] = NumberValue
	valueTypes['4'] = NumberValue
	valueTypes['5'] = NumberValue
	valueTypes['6'] = NumberValue
	valueTypes['7'] = NumberValue
	valueTypes['8'] = NumberValue
	valueTypes['9'] = NumberValue
	valueTypes['t'] = BoolValue
	valueTypes['f'] = BoolValue
	valueTypes['n'] = NilValue
	valueTypes['['] = ArrayValue
	valueTypes['{'] = ObjectValue
}

// Iterator is a io.Reader like object, with JSON specific read functions.
// Error is not returned as return value, but stored as Error member on this iterator instance.
type Iterator struct {
	cfg              *frozenConfig
	reader           io.Reader
	buf              []byte
	head             int
	tail             int
	captureStartedAt int
	captured         []byte
	Error            error
}

// NewIterator creates an empty Iterator instance
func NewIterator(cfg API) *Iterator {
	return &Iterator{
		cfg:    cfg.(*frozenConfig),
		reader: nil,
		buf:    nil,
		head:   0,
		tail:   0,
	}
}

// Parse creates an Iterator instance from io.Reader
func Parse(cfg API, reader io.Reader, bufSize int) *Iterator {
	return &Iterator{
		cfg:    cfg.(*frozenConfig),
		reader: reader,
		buf:    make([]byte, bufSize),
		head:   0,
		tail:   0,
	}
}

// ParseBytes creates an Iterator instance from byte array
func ParseBytes(cfg API, input []byte) *Iterator {
	return &Iterator{
		cfg:    cfg.(*frozenConfig),
		reader: nil,
		buf:    input,
		head:   0,
		tail:   len(input),
	}
}

// ParseString creates an Iterator instance from string
func ParseString(cfg API, input string) *Iterator {
	return ParseBytes(cfg, []byte(input))
}

// Pool returns a pool can provide more iterator with same configuration
func (iter *Iterator) Pool() IteratorPool {
	return iter.cfg
}

// Reset reuse iterator instance by specifying another reader
func (iter *Iterator) Reset(reader io.Reader) *Iterator {
	iter.reader = reader
	iter.head = 0
	iter.tail = 0
	return iter
}

// ResetBytes reuse iterator instance by specifying another byte array as input
func (iter *Iterator) ResetBytes(input []byte) *Iterator {
	iter.reader = nil
	iter.buf = input
	iter.head = 0
	iter.tail = len(input)
	return iter
}

// WhatIsNext gets ValueType of relatively next json element
func (iter *Iterator) WhatIsNext() ValueType {
	valueType := valueTypes[iter.nextToken()]
	iter.unreadByte()
	return valueType
}

func (iter *Iterator) skipWhitespacesWithoutLoadMore() bool {
	for i := iter.head; i < iter.tail; i++ {
		c := iter.buf[i]
		switch c {
		case ' ', '\n', '\t', '\r':
			continue
		}
		iter.head = i
		return false
	}
	return true
}

func (iter *Iterator) isObjectEnd() bool {
	c := iter.nextToken()
	if c == ',' {
		return false
	}
	if c == '}' {
		return true
	}
	iter.ReportError("isObjectEnd", "object ended prematurely")
	return true
}

func (iter *Iterator) nextToken() byte {
	// a variation of skip whitespaces, returning the next non-whitespace token
	for {
		for i := iter.head; i < iter.tail; i++ {
			c := iter.buf[i]
			switch c {
			case ' ', '\n', '\t', '\r':
				continue
			}
			iter.head = i + 1
			return c
		}
		if !iter.loadMore() {
			return 0
		}
	}
}

// ReportError record a error in iterator instance with current position.
func (iter *Iterator) ReportError(operation string, msg string) {
	if iter.Error != nil {
		if iter.Error != io.EOF {
			return
		}
	}
	peekStart := iter.head - 10
	if peekStart < 0 {
		peekStart = 0
	}
	iter.Error = fmt.Errorf("%s: %s, parsing %v ...%s... at %s", operation, msg, iter.head,
		string(iter.buf[peekStart:iter.head]), string(iter.buf[0:iter.tail]))
}

// CurrentBuffer gets current buffer as string for debugging purpose
func (iter *Iterator) CurrentBuffer() string {
	peekStart := iter.head - 10
	if peekStart < 0 {
		peekStart = 0
	}
	return fmt.Sprintf("parsing %v ...|%s|... at %s", iter.head,
		string(iter.buf[peekStart:iter.head]), string(iter.buf[0:iter.tail]))
}

func (iter *Iterator) readByte() (ret byte) {
	if iter.head == iter.tail {
		if iter.loadMore() {
			ret = iter.buf[iter.head]
			iter.head++
			return ret
		}
		return 0
	}
	ret = iter.buf[iter.head]
	iter.head++
	return ret
}

func (iter *Iterator) loadMore() bool {
	if iter.reader == nil {
		if iter.Error == nil {
			iter.head = iter.tail
			iter.Error = io.EOF
		}
		return false
	}
	if iter.captured != nil {
		iter.captured = append(iter.captured,
			iter.buf[iter.captureStartedAt:iter.tail]...)
		iter.captureStartedAt = 0
	}
	for {
		n, err := iter.reader.Read(iter.buf)
		if n == 0 {
			if err != nil {
				if iter.Error == nil {
					iter.Error = err
				}
				return false
			}
		} else {
			iter.head = 0
			iter.tail = n
			return true
		}
	}
}

func (iter *Iterator) unreadByte() {
	if iter.Error != nil {
		return
	}
	iter.head--
	return
}

// Read read the next JSON element as generic interface{}.
func (iter *Iterator) Read() interface{} {
	valueType := iter.WhatIsNext()
	switch valueType {
	case StringValue:
		return iter.ReadString()
	case NumberValue:
		if iter.cfg.configBeforeFrozen.UseNumber {
			return json.Number(iter.readNumberAsString())
		}
		return iter.ReadFloat64()
	case NilValue:
		iter.skipFourBytes('n', 'u', 'l', 'l')
		return nil
	case BoolValue:
		return iter.ReadBool()
	case ArrayValue:
		arr := []interface{}{}
		iter.ReadArrayCB(func(iter *Iterator) bool {
			var elem interface{}
			iter.ReadVal(&elem)
			arr = append(arr, elem)
			return true
		})
		return arr
	case ObjectValue:
		obj := map[string]interface{}{}
		iter.ReadMapCB(func(Iter *Iterator, field string) bool {
			var elem interface{}
			iter.ReadVal(&elem)
			obj[field] = elem
			return true
		})
		return obj
	default:
		iter.ReportError("Read", fmt.Sprintf("unexpected value type: %v", valueType))
		return nil
	}
}
