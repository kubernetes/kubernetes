package ber

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
)

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

var Debug bool = false

func PrintBytes(out io.Writer, buf []byte, indent string) {
	data_lines := make([]string, (len(buf)/30)+1)
	num_lines := make([]string, (len(buf)/30)+1)

	for i, b := range buf {
		data_lines[i/30] += fmt.Sprintf("%02x ", b)
		num_lines[i/30] += fmt.Sprintf("%02d ", (i+1)%100)
	}

	for i := 0; i < len(data_lines); i++ {
		out.Write([]byte(indent + data_lines[i] + "\n"))
		out.Write([]byte(indent + num_lines[i] + "\n\n"))
	}
}

func PrintPacket(p *Packet) {
	printPacket(os.Stdout, p, 0, false)
}

func printPacket(out io.Writer, p *Packet, indent int, printBytes bool) {
	indent_str := ""

	for len(indent_str) != indent {
		indent_str += " "
	}

	class_str := ClassMap[p.ClassType]

	tagtype_str := TypeMap[p.TagType]

	tag_str := fmt.Sprintf("0x%02X", p.Tag)

	if p.ClassType == ClassUniversal {
		tag_str = tagMap[p.Tag]
	}

	value := fmt.Sprint(p.Value)
	description := ""

	if p.Description != "" {
		description = p.Description + ": "
	}

	fmt.Fprintf(out, "%s%s(%s, %s, %s) Len=%d %q\n", indent_str, description, class_str, tagtype_str, tag_str, p.Data.Len(), value)

	if printBytes {
		PrintBytes(out, p.Bytes(), indent_str)
	}

	for _, child := range p.Children {
		printPacket(out, child, indent+1, printBytes)
	}
}

// ReadPacket reads a single Packet from the reader
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

func parseInt64(bytes []byte) (ret int64, err error) {
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
		out[j] = (byte(i >> uint((n-1)*8)))
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
// If a decode error is encountered, nil is returned
func DecodePacketErr(data []byte) (*Packet, error) {
	p, _, err := readPacket(bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	return p, nil
}

// readPacket reads a single Packet from the reader, returning the number of bytes read
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
				return nil, read, err
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
	content := make([]byte, length, length)
	if length > 0 {
		_, err := io.ReadFull(reader, content)
		if err != nil {
			if err == io.EOF {
				return nil, read, io.ErrUnexpectedEOF
			}
			return nil, read, err
		}
		read += length
	}

	if p.ClassType == ClassUniversal {
		p.Data.Write(content)
		p.ByteValue = content

		switch p.Tag {
		case TagEOC:
		case TagBoolean:
			val, _ := parseInt64(content)

			p.Value = val != 0
		case TagInteger:
			p.Value, _ = parseInt64(content)
		case TagBitString:
		case TagOctetString:
			// the actual string encoding is not known here
			// (e.g. for LDAP content is already an UTF8-encoded
			// string). Return the data without further processing
			p.Value = DecodeString(content)
		case TagNULL:
		case TagObjectIdentifier:
		case TagObjectDescriptor:
		case TagExternal:
		case TagRealFloat:
		case TagEnumerated:
			p.Value, _ = parseInt64(content)
		case TagEmbeddedPDV:
		case TagUTF8String:
			p.Value = DecodeString(content)
		case TagRelativeOID:
		case TagSequence:
		case TagSet:
		case TagNumericString:
		case TagPrintableString:
			p.Value = DecodeString(content)
		case TagT61String:
		case TagVideotexString:
		case TagIA5String:
		case TagUTCTime:
		case TagGeneralizedTime:
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

	return p, read, nil
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

func Encode(ClassType Class, TagType Type, Tag Tag, Value interface{}, Description string) *Packet {
	p := new(Packet)

	p.ClassType = ClassType
	p.TagType = TagType
	p.Tag = Tag
	p.Data = new(bytes.Buffer)

	p.Children = make([]*Packet, 0, 2)

	p.Value = Value
	p.Description = Description

	if Value != nil {
		v := reflect.ValueOf(Value)

		if ClassType == ClassUniversal {
			switch Tag {
			case TagOctetString:
				sv, ok := v.Interface().(string)

				if ok {
					p.Data.Write([]byte(sv))
				}
			}
		}
	}

	return p
}

func NewSequence(Description string) *Packet {
	return Encode(ClassUniversal, TypeConstructed, TagSequence, nil, Description)
}

func NewBoolean(ClassType Class, TagType Type, Tag Tag, Value bool, Description string) *Packet {
	intValue := int64(0)

	if Value {
		intValue = 1
	}

	p := Encode(ClassType, TagType, Tag, nil, Description)

	p.Value = Value
	p.Data.Write(encodeInteger(intValue))

	return p
}

func NewInteger(ClassType Class, TagType Type, Tag Tag, Value interface{}, Description string) *Packet {
	p := Encode(ClassType, TagType, Tag, nil, Description)

	p.Value = Value
	switch v := Value.(type) {
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

func NewString(ClassType Class, TagType Type, Tag Tag, Value, Description string) *Packet {
	p := Encode(ClassType, TagType, Tag, nil, Description)

	p.Value = Value
	p.Data.Write([]byte(Value))

	return p
}
