package jsoniter

import "fmt"

// ReadNil reads a json object as nil and
// returns whether it's a nil or not
func (iter *Iterator) ReadNil() (ret bool) {
	c := iter.nextToken()
	if c == 'n' {
		iter.skipThreeBytes('u', 'l', 'l') // null
		return true
	}
	iter.unreadByte()
	return false
}

// ReadBool reads a json object as BoolValue
func (iter *Iterator) ReadBool() (ret bool) {
	c := iter.nextToken()
	if c == 't' {
		iter.skipThreeBytes('r', 'u', 'e')
		return true
	}
	if c == 'f' {
		iter.skipFourBytes('a', 'l', 's', 'e')
		return false
	}
	iter.ReportError("ReadBool", "expect t or f")
	return
}

// SkipAndReturnBytes skip next JSON element, and return its content as []byte.
// The []byte can be kept, it is a copy of data.
func (iter *Iterator) SkipAndReturnBytes() []byte {
	iter.startCapture(iter.head)
	iter.Skip()
	return iter.stopCapture()
}

type captureBuffer struct {
	startedAt int
	captured  []byte
}

func (iter *Iterator) startCapture(captureStartedAt int) {
	if iter.captured != nil {
		panic("already in capture mode")
	}
	iter.captureStartedAt = captureStartedAt
	iter.captured = make([]byte, 0, 32)
}

func (iter *Iterator) stopCapture() []byte {
	if iter.captured == nil {
		panic("not in capture mode")
	}
	captured := iter.captured
	remaining := iter.buf[iter.captureStartedAt:iter.head]
	iter.captureStartedAt = -1
	iter.captured = nil
	if len(captured) == 0 {
		return remaining
	}
	captured = append(captured, remaining...)
	return captured
}

// Skip skips a json object and positions to relatively the next json object
func (iter *Iterator) Skip() {
	c := iter.nextToken()
	switch c {
	case '"':
		iter.skipString()
	case 'n':
		iter.skipThreeBytes('u', 'l', 'l') // null
	case 't':
		iter.skipThreeBytes('r', 'u', 'e') // true
	case 'f':
		iter.skipFourBytes('a', 'l', 's', 'e') // false
	case '0':
		iter.unreadByte()
		iter.ReadFloat32()
	case '-', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		iter.skipNumber()
	case '[':
		iter.skipArray()
	case '{':
		iter.skipObject()
	default:
		iter.ReportError("Skip", fmt.Sprintf("do not know how to skip: %v", c))
		return
	}
}

func (iter *Iterator) skipFourBytes(b1, b2, b3, b4 byte) {
	if iter.readByte() != b1 {
		iter.ReportError("skipFourBytes", fmt.Sprintf("expect %s", string([]byte{b1, b2, b3, b4})))
		return
	}
	if iter.readByte() != b2 {
		iter.ReportError("skipFourBytes", fmt.Sprintf("expect %s", string([]byte{b1, b2, b3, b4})))
		return
	}
	if iter.readByte() != b3 {
		iter.ReportError("skipFourBytes", fmt.Sprintf("expect %s", string([]byte{b1, b2, b3, b4})))
		return
	}
	if iter.readByte() != b4 {
		iter.ReportError("skipFourBytes", fmt.Sprintf("expect %s", string([]byte{b1, b2, b3, b4})))
		return
	}
}

func (iter *Iterator) skipThreeBytes(b1, b2, b3 byte) {
	if iter.readByte() != b1 {
		iter.ReportError("skipThreeBytes", fmt.Sprintf("expect %s", string([]byte{b1, b2, b3})))
		return
	}
	if iter.readByte() != b2 {
		iter.ReportError("skipThreeBytes", fmt.Sprintf("expect %s", string([]byte{b1, b2, b3})))
		return
	}
	if iter.readByte() != b3 {
		iter.ReportError("skipThreeBytes", fmt.Sprintf("expect %s", string([]byte{b1, b2, b3})))
		return
	}
}
