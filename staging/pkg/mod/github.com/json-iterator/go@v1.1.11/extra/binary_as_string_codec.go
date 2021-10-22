package extra

import (
	"github.com/json-iterator/go"
	"github.com/modern-go/reflect2"
	"unicode/utf8"
	"unsafe"
)

// safeSet holds the value true if the ASCII character with the given array
// position can be represented inside a JSON string without any further
// escaping.
//
// All values are true except for the ASCII control characters (0-31), the
// double quote ("), and the backslash character ("\").
var safeSet = [utf8.RuneSelf]bool{
	' ':      true,
	'!':      true,
	'"':      false,
	'#':      true,
	'$':      true,
	'%':      true,
	'&':      true,
	'\'':     true,
	'(':      true,
	')':      true,
	'*':      true,
	'+':      true,
	',':      true,
	'-':      true,
	'.':      true,
	'/':      true,
	'0':      true,
	'1':      true,
	'2':      true,
	'3':      true,
	'4':      true,
	'5':      true,
	'6':      true,
	'7':      true,
	'8':      true,
	'9':      true,
	':':      true,
	';':      true,
	'<':      true,
	'=':      true,
	'>':      true,
	'?':      true,
	'@':      true,
	'A':      true,
	'B':      true,
	'C':      true,
	'D':      true,
	'E':      true,
	'F':      true,
	'G':      true,
	'H':      true,
	'I':      true,
	'J':      true,
	'K':      true,
	'L':      true,
	'M':      true,
	'N':      true,
	'O':      true,
	'P':      true,
	'Q':      true,
	'R':      true,
	'S':      true,
	'T':      true,
	'U':      true,
	'V':      true,
	'W':      true,
	'X':      true,
	'Y':      true,
	'Z':      true,
	'[':      true,
	'\\':     false,
	']':      true,
	'^':      true,
	'_':      true,
	'`':      true,
	'a':      true,
	'b':      true,
	'c':      true,
	'd':      true,
	'e':      true,
	'f':      true,
	'g':      true,
	'h':      true,
	'i':      true,
	'j':      true,
	'k':      true,
	'l':      true,
	'm':      true,
	'n':      true,
	'o':      true,
	'p':      true,
	'q':      true,
	'r':      true,
	's':      true,
	't':      true,
	'u':      true,
	'v':      true,
	'w':      true,
	'x':      true,
	'y':      true,
	'z':      true,
	'{':      true,
	'|':      true,
	'}':      true,
	'~':      true,
	'\u007f': true,
}

var binaryType = reflect2.TypeOfPtr((*[]byte)(nil)).Elem()

type BinaryAsStringExtension struct {
	jsoniter.DummyExtension
}

func (extension *BinaryAsStringExtension) CreateEncoder(typ reflect2.Type) jsoniter.ValEncoder {
	if typ == binaryType {
		return &binaryAsStringCodec{}
	}
	return nil
}

func (extension *BinaryAsStringExtension) CreateDecoder(typ reflect2.Type) jsoniter.ValDecoder {
	if typ == binaryType {
		return &binaryAsStringCodec{}
	}
	return nil
}

type binaryAsStringCodec struct {
}

func (codec *binaryAsStringCodec) Decode(ptr unsafe.Pointer, iter *jsoniter.Iterator) {
	rawBytes := iter.ReadStringAsSlice()
	bytes := make([]byte, 0, len(rawBytes))
	for i := 0; i < len(rawBytes); i++ {
		b := rawBytes[i]
		if b == '\\' {
			b2 := rawBytes[i+1]
			if b2 != '\\' {
				iter.ReportError("decode binary as string", `\\x is only supported escape`)
				return
			}
			b3 := rawBytes[i+2]
			if b3 != 'x' {
				iter.ReportError("decode binary as string", `\\x is only supported escape`)
				return
			}
			b4 := rawBytes[i+3]
			b5 := rawBytes[i+4]
			i += 4
			b = readHex(iter, b4, b5)
		}
		bytes = append(bytes, b)
	}
	*(*[]byte)(ptr) = bytes
}
func (codec *binaryAsStringCodec) IsEmpty(ptr unsafe.Pointer) bool {
	return len(*((*[]byte)(ptr))) == 0
}
func (codec *binaryAsStringCodec) Encode(ptr unsafe.Pointer, stream *jsoniter.Stream) {
	newBuffer := writeBytes(stream.Buffer(), *(*[]byte)(ptr))
	stream.SetBuffer(newBuffer)
}

func readHex(iter *jsoniter.Iterator, b1, b2 byte) byte {
	var ret byte
	if b1 >= '0' && b1 <= '9' {
		ret = b1 - '0'
	} else if b1 >= 'a' && b1 <= 'f' {
		ret = b1 - 'a' + 10
	} else {
		iter.ReportError("read hex", "expects 0~9 or a~f, but found "+string([]byte{b1}))
		return 0
	}
	ret *= 16
	if b2 >= '0' && b2 <= '9' {
		ret = b2 - '0'
	} else if b2 >= 'a' && b2 <= 'f' {
		ret = b2 - 'a' + 10
	} else {
		iter.ReportError("read hex", "expects 0~9 or a~f, but found "+string([]byte{b2}))
		return 0
	}
	return ret
}

var hex = "0123456789abcdef"

func writeBytes(space []byte, s []byte) []byte {
	space = append(space, '"')
	// write string, the fast path, without utf8 and escape support
	var i int
	var c byte
	for i, c = range s {
		if c < utf8.RuneSelf && safeSet[c] {
			space = append(space, c)
		} else {
			break
		}
	}
	if i == len(s)-1 {
		space = append(space, '"')
		return space
	}
	return writeBytesSlowPath(space, s[i:])
}

func writeBytesSlowPath(space []byte, s []byte) []byte {
	start := 0
	// for the remaining parts, we process them char by char
	var i int
	var b byte
	for i, b = range s {
		if b >= utf8.RuneSelf {
			space = append(space, '\\', '\\', 'x', hex[b>>4], hex[b&0xF])
			start = i + 1
			continue
		}
		if safeSet[b] {
			continue
		}
		if start < i {
			space = append(space, s[start:i]...)
		}
		space = append(space, '\\', '\\', 'x', hex[b>>4], hex[b&0xF])
		start = i + 1
	}
	if start < len(s) {
		space = append(space, s[start:]...)
	}
	return append(space, '"')
}
