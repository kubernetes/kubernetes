package ber

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// MaxPacketLengthBytes specifies the maximum allowed packet size when calling ReadPacket or DecodePacket. Set to 0 for
// no limit.
var MaxPacketLengthBytes int64 = math.MaxInt32

type Packet struct {
	Identifier
	Value       interface{}
	ByteValue   []byte
	Data        *bytes.Buffer
	Children    []*Packet
	Description string
}

type Identifier struct {
	ClassType Class
	TagType   Type
	Tag       Tag
}

type Tag uint64

const (
	TagEOC              Tag = 0x00
	TagBoolean          Tag = 0x01
	TagInteger          Tag = 0x02
	TagBitString        Tag = 0x03
	TagOctetString      Tag = 0x04
	TagNULL             Tag = 0x05
	TagObjectIdentifier Tag = 0x06
	TagObjectDescriptor Tag = 0x07
	TagExternal         Tag = 0x08
	TagRealFloat        Tag = 0x09
	TagEnumerated       Tag = 0x0a
	TagEmbeddedPDV      Tag = 0x0b
	TagUTF8String       Tag = 0x0c
	TagRelativeOID      Tag = 0x0d
	TagSequence         Tag = 0x10
	TagSet              Tag = 0x11
	TagNumericString    Tag = 0x12
	TagPrintableString  Tag = 0x13
	TagT61String        Tag = 0x14
	TagVideotexString   Tag = 0x15
	TagIA5String        Tag = 0x16
	TagUTCTime          Tag = 0x17
	TagGeneralizedTime  Tag = 0x18
	TagGraphicString    Tag = 0x19
	TagVisibleString    Tag = 0x1a
	TagGeneralString    Tag = 0x1b
	TagUniversalString  Tag = 0x1c
	TagCharacterString  Tag = 0x1d
	TagBMPString        Tag = 0x1e
	TagBitmask          Tag = 0x1f // xxx11111b

	// HighTag indicates the start of a high-tag byte sequence
	HighTag Tag = 0x1f // xxx11111b
	// HighTagContinueBitmask indicates the high-tag byte sequence should continue
	HighTagContinueBitmask Tag = 0x80 // 10000000b
	// HighTagValueBitmask obtains the tag value from a high-tag byte sequence byte
	HighTagValueBitmask Tag = 0x7f // 01111111b
)

const (
	// LengthLongFormBitmask is the mask to apply to the length byte to see if a long-form byte sequence is used
	LengthLongFormBitmask = 0x80
	// LengthValueBitmask is the mask to apply to the length byte to get the number of bytes in the long-form byte sequence
	LengthValueBitmask = 0x7f

	// LengthIndefinite is returned from readLength to indicate an indefinite length
	LengthIndefinite = -1
)

var tagMap = map[Tag]string{
	TagEOC:              "EOC (End-of-Content)",
	TagBoolean:          "Boolean",
	TagInteger:          "Integer",
	TagBitString:        "Bit String",
	TagOctetString:      "Octet String",
	TagNULL:             "NULL",
	TagObjectIdentifier: "Object Identifier",
	TagObjectDescriptor: "Object Descriptor",
	TagExternal:         "External",
	TagRealFloat:        "Real (float)",
	TagEnumerated:       "Enumerated",
	TagEmbeddedPDV:      "Embedded PDV",
	TagUTF8String:       "UTF8 String",
	TagRelativeOID:      "Relative-OID",
	TagSequence:         "Sequence and Sequence of",
	TagSet:              "Set and Set OF",
	TagNumericString:    "Numeric String",
	TagPrintableString:  "Printable String",
	TagT61String:        "T61 String",
	TagVideotexString:   "Videotex String",
	TagIA5String:        "IA5 String",
	TagUTCTime:          "UTC Time",
	TagGeneralizedTime:  "Generalized Time",
	TagGraphicString:    "Graphic String",
	TagVisibleString:    "Visible String",
	TagGeneralString:    "General String",
	TagUniversalString:  "Universal String",
	TagCharacterString:  "Character String",
	TagBMPString:        "BMP String",
}

type Class uint8

const (
	ClassUniversal   Class = 0   // 00xxxxxxb
	ClassApplication Class = 64  // 01xxxxxxb
	ClassContext     Class = 128 // 10xxxxxxb
	ClassPrivate     Class = 192 // 11xxxxxxb
	ClassBitmask     Class = 192 // 11xxxxxxb
)

var ClassMap = map[Class]string{
	ClassUniversal:   "Universal",
	ClassApplication: "Application",
	ClassContext:     "Context",
	ClassPrivate:     "Private",
}

type Type uint8

const (
	TypePrimitive   Type = 0  // xx0xxxxxb
	TypeConstructed Type = 32 // xx1xxxxxb
	TypeBitmask     Type = 32 // xx1xxxxxb
)

var TypeMap = map[Type]string{
	TypePrimitive:   "Primitive",
	TypeConstructed: "Constructed",
}

var Debug = false

func PrintBytes(out io.Writer, buf []byte, indent string) {
	dataLines := make([]string, (len(buf)/30)+1)
	numLines := make([]string, (len(buf)/30)+1)

	for i, b := range buf {
		dataLines[i/30] += fmt.Sprintf("%02x ", b)
		numLines[i/30] += fmt.Sprintf("%02d ", (i+1)%100)
	}

	for i := 0; i < len(dataLines); i++ {
		_, _ = out.Write([]byte(indent + dataLines[i] + "\n"))
		_, _ = out.Write([]byte(indent + numLines[i] + "\n\n"))
	}
}

func WritePacket(out io.Writer, p *Packet) {
	printPacket(out, p, 0, false)
}

func PrintPacket(p *Packet) {
	printPacket(os.Stdout, p, 0, false)
}

// Return a string describing packet content. This is not recursive,
// If the packet is a sequence, use `printPacket()`, or browse
// sequence yourself.
func DescribePacket(p *Packet) string {

	classStr := ClassMap[p.ClassType]

	tagTypeStr := TypeMap[p.TagType]

	tagStr := fmt.Sprintf("0x%02X", p.Tag)

	if p.ClassType == ClassUniversal {
		tagStr = tagMap[p.Tag]
	}

	value := fmt.Sprint(p.Value)
	description := ""

	if p.Description != "" {
		description = p.Description + ": "
	}

	return fmt.Sprintf("%s(%s, %s, %s) Len=%d %q", description, classStr, tagTypeStr, tagStr, p.Data.Len(), value)
}

func printPacket(out io.Writer, p *Packet, indent int, printBytes bool) {
	indentStr := ""

	for len(indentStr) != indent {
		indentStr += " "
	}

	_, _ = fmt.Fprintf(out, "%s%s\n", indentStr, DescribePacket(p))

	if printBytes {
		PrintBytes(out, p.Bytes(), indentStr)
	}

	for _, child := range p.Children {
		printPacket(out, child, indent+1, printBytes)
	}
}

// ReadPacket reads a single Packet from the reader.
func ReadPacket(reader io.Reader) (*Packet, error) {
	p, _, err := readPacket(reader)
	if err != nil {
		return nil, err
	}
	return p, nil
}

func DecodeString(data []byte) string {
	return string(data)
}

func ParseInt64(bytes []byte) (ret int64, err error) {
	if len(bytes) > 8 {
		// We'll overflow an int64 in this case.
		err = fmt.Errorf("integer too large")
		return
	}
	for bytesRead := 0; bytesRead < len(bytes); bytesRead++ {
		ret <<= 8
		ret |= int64(bytes[bytesRead])
	}

	// Shift up and down in order to sign extend the result.
	ret <<= 64 - uint8(len(bytes))*8
	ret >>= 64 - uint8(len(bytes))*8
	return
}

func encodeInteger(i int64) []byte {
	n := int64Length(i)
	out := make([]byte, n)

	var j int
	for ; n > 0; n-- {
		out[j] = byte(i >> uint((n-1)*8))
		j++
	}

	return out
}

func int64Length(i int64) (numBytes int) {
	numBytes = 1

	for i > 127 {
		numBytes++
		i >>= 8
	}

	for i < -128 {
		numBytes++
		i >>= 8
	}

	return
}

// DecodePacket decodes the given bytes into a single Packet
// If a decode error is encountered, nil is returned.
func DecodePacket(data []byte) *Packet {
	p, _, _ := readPacket(bytes.NewBuffer(data))

	return p
}

// DecodePacketErr decodes the given bytes into a single Packet
// If a decode error is encountered, nil is returned.
func DecodePacketErr(data []byte) (*Packet, error) {
	p, _, err := readPacket(bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	return p, nil
}

// readPacket reads a single Packet from the reader, returning the number of bytes read.
func readPacket(reader io.Reader) (*Packet, int, error) {
	identifier, length, read, err := readHeader(reader)
	if err != nil {
		return nil, read, err
	}

	p := &Packet{
		Identifier: identifier,
	}

	p.Data = new(bytes.Buffer)
	p.Children = make([]*Packet, 0, 2)
	p.Value = nil

	if p.TagType == TypeConstructed {
		// TODO: if universal, ensure tag type is allowed to be constructed

		// Track how much content we've read
		contentRead := 0
		for {
			if length != LengthIndefinite {
				// End if we've read what we've been told to
				if contentRead == length {
					break
				}
				// Detect if a packet boundary didn't fall on the expected length
				if contentRead > length {
					return nil, read, fmt.Errorf("expected to read %d bytes, read %d", length, contentRead)
				}
			}

			// Read the next packet
			child, r, err := readPacket(reader)
			if err != nil {
				return nil, read, unexpectedEOF(err)
			}
			contentRead += r
			read += r

			// Test is this is the EOC marker for our packet
			if isEOCPacket(child) {
				if length == LengthIndefinite {
					break
				}
				return nil, read, errors.New("eoc child not allowed with definite length")
			}

			// Append and continue
			p.AppendChild(child)
		}
		return p, read, nil
	}

	if length == LengthIndefinite {
		return nil, read, errors.New("indefinite length used with primitive type")
	}

	// Read definite-length content
	if MaxPacketLengthBytes > 0 && int64(length) > MaxPacketLengthBytes {
		return nil, read, fmt.Errorf("length %d greater than maximum %d", length, MaxPacketLengthBytes)
	}

	var content []byte
	if length > 0 {
		// Read the content and limit it to the parsed length.
		// If the content is less than the length, we return an EOF error.
		content, err = ioutil.ReadAll(io.LimitReader(reader, int64(length)))
		if err == nil && len(content) < int(length) {
			err = io.EOF
		}
		if err != nil {
			return nil, read, unexpectedEOF(err)
		}
		read += len(content)
	} else {
		// If length == 0, we set the ByteValue to an empty slice
		content = make([]byte, 0)
	}

	if p.ClassType == ClassUniversal {
		p.Data.Write(content)
		p.ByteValue = content

		switch p.Tag {
		case TagEOC:
		case TagBoolean:
			val, _ := ParseInt64(content)

			p.Value = val != 0
		case TagInteger:
			p.Value, _ = ParseInt64(content)
		case TagBitString:
		case TagOctetString:
			// the actual string encoding is not known here
			// (e.g. for LDAP content is already an UTF8-encoded
			// string). Return the data without further processing
			p.Value = DecodeString(content)
		case TagNULL:
		case TagObjectIdentifier:
			oid, err := parseObjectIdentifier(content)
			if err == nil {
				p.Value = OIDToString(oid)
			}
		case TagObjectDescriptor:
		case TagExternal:
		case TagRealFloat:
			p.Value, err = ParseReal(content)
		case TagEnumerated:
			p.Value, _ = ParseInt64(content)
		case TagEmbeddedPDV:
		case TagUTF8String:
			val := DecodeString(content)
			if !utf8.Valid([]byte(val)) {
				err = errors.New("invalid UTF-8 string")
			} else {
				p.Value = val
			}
		case TagRelativeOID:
			oid, err := parseRelativeObjectIdentifier(content)
			if err == nil {
				p.Value = OIDToString(oid)
			}
		case TagSequence:
		case TagSet:
		case TagNumericString:
		case TagPrintableString:
			val := DecodeString(content)
			if err = isPrintableString(val); err == nil {
				p.Value = val
			}
		case TagT61String:
		case TagVideotexString:
		case TagIA5String:
			val := DecodeString(content)
			for i, c := range val {
				if c >= 0x7F {
					err = fmt.Errorf("invalid character for IA5String at pos %d: %c", i, c)
					break
				}
			}
			if err == nil {
				p.Value = val
			}
		case TagUTCTime:
		case TagGeneralizedTime:
			p.Value, err = ParseGeneralizedTime(content)
		case TagGraphicString:
		case TagVisibleString:
		case TagGeneralString:
		case TagUniversalString:
		case TagCharacterString:
		case TagBMPString:
		}
	} else {
		p.Data.Write(content)
	}

	return p, read, err
}

func isPrintableString(val string) error {
	for i, c := range val {
		switch {
		case c >= 'a' && c <= 'z':
		case c >= 'A' && c <= 'Z':
		case c >= '0' && c <= '9':
		default:
			switch c {
			case '\'', '(', ')', '+', ',', '-', '.', '=', '/', ':', '?', ' ':
			default:
				return fmt.Errorf("invalid character in position %d", i)
			}
		}
	}
	return nil
}

func (p *Packet) Bytes() []byte {
	var out bytes.Buffer

	out.Write(encodeIdentifier(p.Identifier))
	out.Write(encodeLength(p.Data.Len()))
	out.Write(p.Data.Bytes())

	return out.Bytes()
}

func (p *Packet) AppendChild(child *Packet) {
	p.Data.Write(child.Bytes())
	p.Children = append(p.Children, child)
}

func Encode(classType Class, tagType Type, tag Tag, value interface{}, description string) *Packet {
	p := new(Packet)

	p.ClassType = classType
	p.TagType = tagType
	p.Tag = tag
	p.Data = new(bytes.Buffer)

	p.Children = make([]*Packet, 0, 2)

	p.Value = value
	p.Description = description

	if value != nil {
		v := reflect.ValueOf(value)

		if classType == ClassUniversal {
			switch tag {
			case TagOctetString:
				sv, ok := v.Interface().(string)

				if ok {
					p.Data.Write([]byte(sv))
				}
			case TagEnumerated:
				bv, ok := v.Interface().([]byte)
				if ok {
					p.Data.Write(bv)
				}
			case TagEmbeddedPDV:
				bv, ok := v.Interface().([]byte)
				if ok {
					p.Data.Write(bv)
				}
			}
		} else if classType == ClassContext {
			switch tag {
			case TagEnumerated:
				bv, ok := v.Interface().([]byte)
				if ok {
					p.Data.Write(bv)
				}
			case TagEmbeddedPDV:
				bv, ok := v.Interface().([]byte)
				if ok {
					p.Data.Write(bv)
				}
			}
		}
	}
	return p
}

func NewSequence(description string) *Packet {
	return Encode(ClassUniversal, TypeConstructed, TagSequence, nil, description)
}

func NewBoolean(classType Class, tagType Type, tag Tag, value bool, description string) *Packet {
	intValue := int64(0)

	if value {
		intValue = 1
	}

	p := Encode(classType, tagType, tag, nil, description)

	p.Value = value
	p.Data.Write(encodeInteger(intValue))

	return p
}

// NewLDAPBoolean returns a RFC 4511-compliant Boolean packet.
func NewLDAPBoolean(classType Class, tagType Type, tag Tag, value bool, description string) *Packet {
	p := Encode(classType, tagType, tag, nil, description)

	p.Value = value
	if value {
		p.Data.Write([]byte{255})
	} else {
		p.Data.Write([]byte{0})
	}

	return p
}

func NewInteger(classType Class, tagType Type, tag Tag, value interface{}, description string) *Packet {
	p := Encode(classType, tagType, tag, nil, description)

	p.Value = value
	switch v := value.(type) {
	case int:
		p.Data.Write(encodeInteger(int64(v)))
	case uint:
		p.Data.Write(encodeInteger(int64(v)))
	case int64:
		p.Data.Write(encodeInteger(v))
	case uint64:
		// TODO : check range or add encodeUInt...
		p.Data.Write(encodeInteger(int64(v)))
	case int32:
		p.Data.Write(encodeInteger(int64(v)))
	case uint32:
		p.Data.Write(encodeInteger(int64(v)))
	case int16:
		p.Data.Write(encodeInteger(int64(v)))
	case uint16:
		p.Data.Write(encodeInteger(int64(v)))
	case int8:
		p.Data.Write(encodeInteger(int64(v)))
	case uint8:
		p.Data.Write(encodeInteger(int64(v)))
	default:
		// TODO : add support for big.Int ?
		panic(fmt.Sprintf("Invalid type %T, expected {u|}int{64|32|16|8}", v))
	}

	return p
}

func NewString(classType Class, tagType Type, tag Tag, value, description string) *Packet {
	p := Encode(classType, tagType, tag, nil, description)

	p.Value = value
	p.Data.Write([]byte(value))

	return p
}

func NewGeneralizedTime(classType Class, tagType Type, tag Tag, value time.Time, description string) *Packet {
	p := Encode(classType, tagType, tag, nil, description)
	var s string
	if value.Nanosecond() != 0 {
		s = value.Format(`20060102150405.000000000Z`)
	} else {
		s = value.Format(`20060102150405Z`)
	}
	p.Value = s
	p.Data.Write([]byte(s))
	return p
}

func NewReal(classType Class, tagType Type, tag Tag, value interface{}, description string) *Packet {
	p := Encode(classType, tagType, tag, nil, description)

	switch v := value.(type) {
	case float64:
		p.Data.Write(encodeFloat(v))
	case float32:
		p.Data.Write(encodeFloat(float64(v)))
	default:
		panic(fmt.Sprintf("Invalid type %T, expected float{64|32}", v))
	}
	return p
}

func NewOID(classType Class, tagType Type, tag Tag, value interface{}, description string) *Packet {
	p := Encode(classType, tagType, tag, nil, description)

	switch v := value.(type) {
	case string:
		encoded, err := encodeOID(v)
		if err != nil {
			fmt.Printf("failed writing %v", err)
			return nil
		}
		p.Value = v
		p.Data.Write(encoded)
		// TODO: support []int already ?
	default:
		panic(fmt.Sprintf("Invalid type %T, expected float{64|32}", v))
	}
	return p
}

func NewRelativeOID(classType Class, tagType Type, tag Tag, value interface{}, description string) *Packet {
	p := Encode(classType, tagType, tag, nil, description)

	switch v := value.(type) {
	case string:
		encoded, err := encodeRelativeOID(v)
		if err != nil {
			fmt.Printf("failed writing %v", err)
			return nil
		}
		p.Value = v
		p.Data.Write(encoded)
		// TODO: support []int already ?
	default:
		panic(fmt.Sprintf("Invalid type %T, expected float{64|32}", v))
	}
	return p
}

// encodeOID takes a string representation of an OID and returns its DER-encoded byte slice along with any error.
func encodeOID(oidString string) ([]byte, error) {
	// Convert the string representation to an asn1.ObjectIdentifier
	parts := strings.Split(oidString, ".")
	oid := make([]int, len(parts))
	for i, part := range parts {
		var val int
		if _, err := fmt.Sscanf(part, "%d", &val); err != nil {
			return nil, fmt.Errorf("invalid OID part '%s': %w", part, err)
		}
		oid[i] = val
	}
	if len(oid) < 2 || oid[0] > 2 || (oid[0] < 2 && oid[1] >= 40) {
		panic(fmt.Sprintf("invalid object identifier % d", oid)) // TODO: not elegant
	}
	encoded := make([]byte, 0)

	encoded = appendBase128Int(encoded[:0], int64(oid[0]*40+oid[1]))
	for i := 2; i < len(oid); i++ {
		encoded = appendBase128Int(encoded, int64(oid[i]))
	}

	return encoded, nil
}

func encodeRelativeOID(oidString string) ([]byte, error) {
	parts := strings.Split(oidString, ".")
	oid := make([]int, len(parts))
	for i, part := range parts {
		var val int
		if _, err := fmt.Sscanf(part, "%d", &val); err != nil {
			return nil, fmt.Errorf("invalid RELATIVE OID part '%s': %w", part, err)
		}
		oid[i] = val
	}

	encoded := make([]byte, 0)

	for i := 0; i < len(oid); i++ {
		encoded = appendBase128Int(encoded, int64(oid[i]))
	}

	return encoded, nil
}

func appendBase128Int(dst []byte, n int64) []byte {
	l := base128IntLength(n)

	for i := l - 1; i >= 0; i-- {
		o := byte(n >> uint(i*7))
		o &= 0x7f
		if i != 0 {
			o |= 0x80
		}

		dst = append(dst, o)
	}

	return dst
}
func base128IntLength(n int64) int {
	if n == 0 {
		return 1
	}

	l := 0
	for i := n; i > 0; i >>= 7 {
		l++
	}

	return l
}

func OIDToString(oi []int) string {
	var s strings.Builder
	s.Grow(32)

	buf := make([]byte, 0, 19)
	for i, v := range oi {
		if i > 0 {
			s.WriteByte('.')
		}
		s.Write(strconv.AppendInt(buf, int64(v), 10))
	}

	return s.String()
}

// parseObjectIdentifier parses an OBJECT IDENTIFIER from the given bytes and
// returns it. An object identifier is a sequence of variable length integers
// that are assigned in a hierarchy.
func parseObjectIdentifier(bytes []byte) (s []int, err error) {
	if len(bytes) == 0 {
		err = fmt.Errorf("zero length OBJECT IDENTIFIER")
		return
	}

	// In the worst case, we get two elements from the first byte (which is
	// encoded differently) and then every varint is a single byte long.
	s = make([]int, len(bytes)+1)

	// The first varint is 40*value1 + value2:
	// According to this packing, value1 can take the values 0, 1 and 2 only.
	// When value1 = 0 or value1 = 1, then value2 is <= 39. When value1 = 2,
	// then there are no restrictions on value2.
	v, offset, err := parseBase128Int(bytes, 0)
	if err != nil {
		return
	}
	if v < 80 {
		s[0] = v / 40
		s[1] = v % 40
	} else {
		s[0] = 2
		s[1] = v - 80
	}

	i := 2
	for ; offset < len(bytes); i++ {
		v, offset, err = parseBase128Int(bytes, offset)
		if err != nil {
			return
		}
		s[i] = v
	}
	s = s[0:i]
	return
}

func parseRelativeObjectIdentifier(bytes []byte) (s []int, err error) {
	if len(bytes) == 0 {
		err = fmt.Errorf("zero length RELATIVE OBJECT IDENTIFIER")
		return
	}

	s = make([]int, len(bytes)+1)

	var v, offset int
	i := 0
	for ; offset < len(bytes); i++ {
		v, offset, err = parseBase128Int(bytes, offset)
		if err != nil {
			return
		}
		s[i] = v
	}
	s = s[0:i]
	return
}

// parseBase128Int parses a base-128 encoded int from the given offset in the
// given byte slice. It returns the value and the new offset.
func parseBase128Int(bytes []byte, initOffset int) (ret, offset int, err error) {
	offset = initOffset
	var ret64 int64
	for shifted := 0; offset < len(bytes); shifted++ {
		// 5 * 7 bits per byte == 35 bits of data
		// Thus the representation is either non-minimal or too large for an int32
		if shifted == 5 {
			err = fmt.Errorf("base 128 integer too large")
			return
		}
		ret64 <<= 7
		b := bytes[offset]
		// integers should be minimally encoded, so the leading octet should
		// never be 0x80
		if shifted == 0 && b == 0x80 {
			err = fmt.Errorf("integer is not minimally encoded")
			return
		}
		ret64 |= int64(b & 0x7f)
		offset++
		if b&0x80 == 0 {
			ret = int(ret64)
			// Ensure that the returned value fits in an int on all platforms
			if ret64 > math.MaxInt32 {
				err = fmt.Errorf("base 128 integer too large")
			}
			return
		}
	}
	err = fmt.Errorf("truncated base 128 integer")
	return
}
