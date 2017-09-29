//+build !jsoniter-sloppy

package jsoniter

import "fmt"

func (iter *Iterator) skipNumber() {
	if !iter.trySkipNumber() {
		iter.unreadByte()
		iter.ReadFloat32()
	}
}

func (iter *Iterator) trySkipNumber() bool {
	dotFound := false
	for i := iter.head; i < iter.tail; i++ {
		c := iter.buf[i]
		switch c {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		case '.':
			if dotFound {
				iter.ReportError("validateNumber", `more than one dot found in number`)
				return true // already failed
			}
			if i+1 == iter.tail {
				return false
			}
			c = iter.buf[i+1]
			switch c {
			case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			default:
				iter.ReportError("validateNumber", `missing digit after dot`)
				return true // already failed
			}
			dotFound = true
		default:
			switch c {
			case ',', ']', '}', ' ', '\t', '\n', '\r':
				if iter.head == i {
					return false // if - without following digits
				}
				iter.head = i
				return true // must be valid
			}
			return false // may be invalid
		}
	}
	return false
}

func (iter *Iterator) skipString() {
	if !iter.trySkipString() {
		iter.unreadByte()
		iter.ReadString()
	}
}

func (iter *Iterator) trySkipString() bool {
	for i := iter.head; i < iter.tail; i++ {
		c := iter.buf[i]
		if c == '"' {
			iter.head = i + 1
			return true // valid
		} else if c == '\\' {
			return false
		} else if c < ' ' {
			iter.ReportError("ReadString",
				fmt.Sprintf(`invalid control character found: %d`, c))
			return true // already failed
		}
	}
	return false
}

func (iter *Iterator) skipObject() {
	iter.unreadByte()
	iter.ReadObjectCB(func(iter *Iterator, field string) bool {
		iter.Skip()
		return true
	})
}

func (iter *Iterator) skipArray() {
	iter.unreadByte()
	iter.ReadArrayCB(func(iter *Iterator) bool {
		iter.Skip()
		return true
	})
}
