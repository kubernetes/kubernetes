package jsoniter

import (
	"fmt"
	"unicode"
	"unsafe"
)

// ReadObject read one field from object.
// If object ended, returns empty string.
// Otherwise, returns the field name.
func (iter *Iterator) ReadObject() (ret string) {
	c := iter.nextToken()
	switch c {
	case 'n':
		iter.skipThreeBytes('u', 'l', 'l')
		return "" // null
	case '{':
		c = iter.nextToken()
		if c == '"' {
			iter.unreadByte()
			return string(iter.readObjectFieldAsBytes())
		}
		if c == '}' {
			return "" // end of object
		}
		iter.ReportError("ReadObject", `expect " after {`)
		return
	case ',':
		return string(iter.readObjectFieldAsBytes())
	case '}':
		return "" // end of object
	default:
		iter.ReportError("ReadObject", fmt.Sprintf(`expect { or , or } or n, but found %s`, string([]byte{c})))
		return
	}
}

func (iter *Iterator) readFieldHash() int32 {
	hash := int64(0x811c9dc5)
	c := iter.nextToken()
	if c == '"' {
		for {
			for i := iter.head; i < iter.tail; i++ {
				// require ascii string and no escape
				b := iter.buf[i]
				if 'A' <= b && b <= 'Z' {
					b += 'a' - 'A'
				}
				if b == '"' {
					iter.head = i + 1
					c = iter.nextToken()
					if c != ':' {
						iter.ReportError("readFieldHash", `expect :, but found `+string([]byte{c}))
					}
					return int32(hash)
				}
				hash ^= int64(b)
				hash *= 0x1000193
			}
			if !iter.loadMore() {
				iter.ReportError("readFieldHash", `incomplete field name`)
				return 0
			}
		}
	}
	iter.ReportError("readFieldHash", `expect ", but found `+string([]byte{c}))
	return 0
}

func calcHash(str string) int32 {
	hash := int64(0x811c9dc5)
	for _, b := range str {
		hash ^= int64(unicode.ToLower(b))
		hash *= 0x1000193
	}
	return int32(hash)
}

// ReadObjectCB read object with callback, the key is ascii only and field name not copied
func (iter *Iterator) ReadObjectCB(callback func(*Iterator, string) bool) bool {
	c := iter.nextToken()
	if c == '{' {
		c = iter.nextToken()
		if c == '"' {
			iter.unreadByte()
			field := iter.readObjectFieldAsBytes()
			if !callback(iter, *(*string)(unsafe.Pointer(&field))) {
				return false
			}
			c = iter.nextToken()
			for c == ',' {
				field = iter.readObjectFieldAsBytes()
				if !callback(iter, *(*string)(unsafe.Pointer(&field))) {
					return false
				}
				c = iter.nextToken()
			}
			if c != '}' {
				iter.ReportError("ReadObjectCB", `object not ended with }`)
				return false
			}
			return true
		}
		if c == '}' {
			return true
		}
		iter.ReportError("ReadObjectCB", `expect " after }`)
		return false
	}
	if c == 'n' {
		iter.skipThreeBytes('u', 'l', 'l')
		return true // null
	}
	iter.ReportError("ReadObjectCB", `expect { or n`)
	return false
}

// ReadMapCB read map with callback, the key can be any string
func (iter *Iterator) ReadMapCB(callback func(*Iterator, string) bool) bool {
	c := iter.nextToken()
	if c == '{' {
		c = iter.nextToken()
		if c == '"' {
			iter.unreadByte()
			field := iter.ReadString()
			if iter.nextToken() != ':' {
				iter.ReportError("ReadMapCB", "expect : after object field")
				return false
			}
			if !callback(iter, field) {
				return false
			}
			c = iter.nextToken()
			for c == ',' {
				field = iter.ReadString()
				if iter.nextToken() != ':' {
					iter.ReportError("ReadMapCB", "expect : after object field")
					return false
				}
				if !callback(iter, field) {
					return false
				}
				c = iter.nextToken()
			}
			if c != '}' {
				iter.ReportError("ReadMapCB", `object not ended with }`)
				return false
			}
			return true
		}
		if c == '}' {
			return true
		}
		iter.ReportError("ReadMapCB", `expect " after }`)
		return false
	}
	if c == 'n' {
		iter.skipThreeBytes('u', 'l', 'l')
		return true // null
	}
	iter.ReportError("ReadMapCB", `expect { or n`)
	return false
}

func (iter *Iterator) readObjectStart() bool {
	c := iter.nextToken()
	if c == '{' {
		c = iter.nextToken()
		if c == '}' {
			return false
		}
		iter.unreadByte()
		return true
	} else if c == 'n' {
		iter.skipThreeBytes('u', 'l', 'l')
		return false
	}
	iter.ReportError("readObjectStart", "expect { or n")
	return false
}

func (iter *Iterator) readObjectFieldAsBytes() (ret []byte) {
	str := iter.ReadStringAsSlice()
	if iter.skipWhitespacesWithoutLoadMore() {
		if ret == nil {
			ret = make([]byte, len(str))
			copy(ret, str)
		}
		if !iter.loadMore() {
			return
		}
	}
	if iter.buf[iter.head] != ':' {
		iter.ReportError("readObjectFieldAsBytes", "expect : after object field")
		return
	}
	iter.head++
	if iter.skipWhitespacesWithoutLoadMore() {
		if ret == nil {
			ret = make([]byte, len(str))
			copy(ret, str)
		}
		if !iter.loadMore() {
			return
		}
	}
	if ret == nil {
		return str
	}
	return ret
}
