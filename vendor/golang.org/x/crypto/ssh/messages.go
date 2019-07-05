// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/big"
	"reflect"
	"strconv"
	"strings"
)

// These are SSH message type numbers. They are scattered around several
// documents but many were taken from [SSH-PARAMETERS].
const (
	msgIgnore        = 2
	msgUnimplemented = 3
	msgDebug         = 4
	msgNewKeys       = 21
)

// SSH messages:
//
// These structures mirror the wire format of the corresponding SSH messages.
// They are marshaled using reflection with the marshal and unmarshal functions
// in this file. The only wrinkle is that a final member of type []byte with a
// ssh tag of "rest" receives the remainder of a packet when unmarshaling.

// See RFC 4253, section 11.1.
const msgDisconnect = 1

// disconnectMsg is the message that signals a disconnect. It is also
// the error type returned from mux.Wait()
type disconnectMsg struct {
	Reason   uint32 `sshtype:"1"`
	Message  string
	Language string
}

func (d *disconnectMsg) Error() string {
	return fmt.Sprintf("ssh: disconnect, reason %d: %s", d.Reason, d.Message)
}

// See RFC 4253, section 7.1.
const msgKexInit = 20

type kexInitMsg struct {
	Cookie                  [16]byte `sshtype:"20"`
	KexAlgos                []string
	ServerHostKeyAlgos      []string
	CiphersClientServer     []string
	CiphersServerClient     []string
	MACsClientServer        []string
	MACsServerClient        []string
	CompressionClientServer []string
	CompressionServerClient []string
	LanguagesClientServer   []string
	LanguagesServerClient   []string
	FirstKexFollows         bool
	Reserved                uint32
}

// See RFC 4253, section 8.

// Diffie-Helman
const msgKexDHInit = 30

type kexDHInitMsg struct {
	X *big.Int `sshtype:"30"`
}

const msgKexECDHInit = 30

type kexECDHInitMsg struct {
	ClientPubKey []byte `sshtype:"30"`
}

const msgKexECDHReply = 31

type kexECDHReplyMsg struct {
	HostKey         []byte `sshtype:"31"`
	EphemeralPubKey []byte
	Signature       []byte
}

const msgKexDHReply = 31

type kexDHReplyMsg struct {
	HostKey   []byte `sshtype:"31"`
	Y         *big.Int
	Signature []byte
}

// See RFC 4253, section 10.
const msgServiceRequest = 5

type serviceRequestMsg struct {
	Service string `sshtype:"5"`
}

// See RFC 4253, section 10.
const msgServiceAccept = 6

type serviceAcceptMsg struct {
	Service string `sshtype:"6"`
}

// See RFC 4252, section 5.
const msgUserAuthRequest = 50

type userAuthRequestMsg struct {
	User    string `sshtype:"50"`
	Service string
	Method  string
	Payload []byte `ssh:"rest"`
}

// Used for debug printouts of packets.
type userAuthSuccessMsg struct {
}

// See RFC 4252, section 5.1
const msgUserAuthFailure = 51

type userAuthFailureMsg struct {
	Methods        []string `sshtype:"51"`
	PartialSuccess bool
}

// See RFC 4252, section 5.1
const msgUserAuthSuccess = 52

// See RFC 4252, section 5.4
const msgUserAuthBanner = 53

type userAuthBannerMsg struct {
	Message string `sshtype:"53"`
	// unused, but required to allow message parsing
	Language string
}

// See RFC 4256, section 3.2
const msgUserAuthInfoRequest = 60
const msgUserAuthInfoResponse = 61

type userAuthInfoRequestMsg struct {
	User               string `sshtype:"60"`
	Instruction        string
	DeprecatedLanguage string
	NumPrompts         uint32
	Prompts            []byte `ssh:"rest"`
}

// See RFC 4254, section 5.1.
const msgChannelOpen = 90

type channelOpenMsg struct {
	ChanType         string `sshtype:"90"`
	PeersID          uint32
	PeersWindow      uint32
	MaxPacketSize    uint32
	TypeSpecificData []byte `ssh:"rest"`
}

const msgChannelExtendedData = 95
const msgChannelData = 94

// Used for debug print outs of packets.
type channelDataMsg struct {
	PeersID uint32 `sshtype:"94"`
	Length  uint32
	Rest    []byte `ssh:"rest"`
}

// See RFC 4254, section 5.1.
const msgChannelOpenConfirm = 91

type channelOpenConfirmMsg struct {
	PeersID          uint32 `sshtype:"91"`
	MyID             uint32
	MyWindow         uint32
	MaxPacketSize    uint32
	TypeSpecificData []byte `ssh:"rest"`
}

// See RFC 4254, section 5.1.
const msgChannelOpenFailure = 92

type channelOpenFailureMsg struct {
	PeersID  uint32 `sshtype:"92"`
	Reason   RejectionReason
	Message  string
	Language string
}

const msgChannelRequest = 98

type channelRequestMsg struct {
	PeersID             uint32 `sshtype:"98"`
	Request             string
	WantReply           bool
	RequestSpecificData []byte `ssh:"rest"`
}

// See RFC 4254, section 5.4.
const msgChannelSuccess = 99

type channelRequestSuccessMsg struct {
	PeersID uint32 `sshtype:"99"`
}

// See RFC 4254, section 5.4.
const msgChannelFailure = 100

type channelRequestFailureMsg struct {
	PeersID uint32 `sshtype:"100"`
}

// See RFC 4254, section 5.3
const msgChannelClose = 97

type channelCloseMsg struct {
	PeersID uint32 `sshtype:"97"`
}

// See RFC 4254, section 5.3
const msgChannelEOF = 96

type channelEOFMsg struct {
	PeersID uint32 `sshtype:"96"`
}

// See RFC 4254, section 4
const msgGlobalRequest = 80

type globalRequestMsg struct {
	Type      string `sshtype:"80"`
	WantReply bool
	Data      []byte `ssh:"rest"`
}

// See RFC 4254, section 4
const msgRequestSuccess = 81

type globalRequestSuccessMsg struct {
	Data []byte `ssh:"rest" sshtype:"81"`
}

// See RFC 4254, section 4
const msgRequestFailure = 82

type globalRequestFailureMsg struct {
	Data []byte `ssh:"rest" sshtype:"82"`
}

// See RFC 4254, section 5.2
const msgChannelWindowAdjust = 93

type windowAdjustMsg struct {
	PeersID         uint32 `sshtype:"93"`
	AdditionalBytes uint32
}

// See RFC 4252, section 7
const msgUserAuthPubKeyOk = 60

type userAuthPubKeyOkMsg struct {
	Algo   string `sshtype:"60"`
	PubKey []byte
}

// typeTags returns the possible type bytes for the given reflect.Type, which
// should be a struct. The possible values are separated by a '|' character.
func typeTags(structType reflect.Type) (tags []byte) {
	tagStr := structType.Field(0).Tag.Get("sshtype")

	for _, tag := range strings.Split(tagStr, "|") {
		i, err := strconv.Atoi(tag)
		if err == nil {
			tags = append(tags, byte(i))
		}
	}

	return tags
}

func fieldError(t reflect.Type, field int, problem string) error {
	if problem != "" {
		problem = ": " + problem
	}
	return fmt.Errorf("ssh: unmarshal error for field %s of type %s%s", t.Field(field).Name, t.Name(), problem)
}

var errShortRead = errors.New("ssh: short read")

// Unmarshal parses data in SSH wire format into a structure. The out
// argument should be a pointer to struct. If the first member of the
// struct has the "sshtype" tag set to a '|'-separated set of numbers
// in decimal, the packet must start with one of those numbers. In
// case of error, Unmarshal returns a ParseError or
// UnexpectedMessageError.
func Unmarshal(data []byte, out interface{}) error {
	v := reflect.ValueOf(out).Elem()
	structType := v.Type()
	expectedTypes := typeTags(structType)

	var expectedType byte
	if len(expectedTypes) > 0 {
		expectedType = expectedTypes[0]
	}

	if len(data) == 0 {
		return parseError(expectedType)
	}

	if len(expectedTypes) > 0 {
		goodType := false
		for _, e := range expectedTypes {
			if e > 0 && data[0] == e {
				goodType = true
				break
			}
		}
		if !goodType {
			return fmt.Errorf("ssh: unexpected message type %d (expected one of %v)", data[0], expectedTypes)
		}
		data = data[1:]
	}

	var ok bool
	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		t := field.Type()
		switch t.Kind() {
		case reflect.Bool:
			if len(data) < 1 {
				return errShortRead
			}
			field.SetBool(data[0] != 0)
			data = data[1:]
		case reflect.Array:
			if t.Elem().Kind() != reflect.Uint8 {
				return fieldError(structType, i, "array of unsupported type")
			}
			if len(data) < t.Len() {
				return errShortRead
			}
			for j, n := 0, t.Len(); j < n; j++ {
				field.Index(j).Set(reflect.ValueOf(data[j]))
			}
			data = data[t.Len():]
		case reflect.Uint64:
			var u64 uint64
			if u64, data, ok = parseUint64(data); !ok {
				return errShortRead
			}
			field.SetUint(u64)
		case reflect.Uint32:
			var u32 uint32
			if u32, data, ok = parseUint32(data); !ok {
				return errShortRead
			}
			field.SetUint(uint64(u32))
		case reflect.Uint8:
			if len(data) < 1 {
				return errShortRead
			}
			field.SetUint(uint64(data[0]))
			data = data[1:]
		case reflect.String:
			var s []byte
			if s, data, ok = parseString(data); !ok {
				return fieldError(structType, i, "")
			}
			field.SetString(string(s))
		case reflect.Slice:
			switch t.Elem().Kind() {
			case reflect.Uint8:
				if structType.Field(i).Tag.Get("ssh") == "rest" {
					field.Set(reflect.ValueOf(data))
					data = nil
				} else {
					var s []byte
					if s, data, ok = parseString(data); !ok {
						return errShortRead
					}
					field.Set(reflect.ValueOf(s))
				}
			case reflect.String:
				var nl []string
				if nl, data, ok = parseNameList(data); !ok {
					return errShortRead
				}
				field.Set(reflect.ValueOf(nl))
			default:
				return fieldError(structType, i, "slice of unsupported type")
			}
		case reflect.Ptr:
			if t == bigIntType {
				var n *big.Int
				if n, data, ok = parseInt(data); !ok {
					return errShortRead
				}
				field.Set(reflect.ValueOf(n))
			} else {
				return fieldError(structType, i, "pointer to unsupported type")
			}
		default:
			return fieldError(structType, i, fmt.Sprintf("unsupported type: %v", t))
		}
	}

	if len(data) != 0 {
		return parseError(expectedType)
	}

	return nil
}

// Marshal serializes the message in msg to SSH wire format.  The msg
// argument should be a struct or pointer to struct. If the first
// member has the "sshtype" tag set to a number in decimal, that
// number is prepended to the result. If the last of member has the
// "ssh" tag set to "rest", its contents are appended to the output.
func Marshal(msg interface{}) []byte {
	out := make([]byte, 0, 64)
	return marshalStruct(out, msg)
}

func marshalStruct(out []byte, msg interface{}) []byte {
	v := reflect.Indirect(reflect.ValueOf(msg))
	msgTypes := typeTags(v.Type())
	if len(msgTypes) > 0 {
		out = append(out, msgTypes[0])
	}

	for i, n := 0, v.NumField(); i < n; i++ {
		field := v.Field(i)
		switch t := field.Type(); t.Kind() {
		case reflect.Bool:
			var v uint8
			if field.Bool() {
				v = 1
			}
			out = append(out, v)
		case reflect.Array:
			if t.Elem().Kind() != reflect.Uint8 {
				panic(fmt.Sprintf("array of non-uint8 in field %d: %T", i, field.Interface()))
			}
			for j, l := 0, t.Len(); j < l; j++ {
				out = append(out, uint8(field.Index(j).Uint()))
			}
		case reflect.Uint32:
			out = appendU32(out, uint32(field.Uint()))
		case reflect.Uint64:
			out = appendU64(out, uint64(field.Uint()))
		case reflect.Uint8:
			out = append(out, uint8(field.Uint()))
		case reflect.String:
			s := field.String()
			out = appendInt(out, len(s))
			out = append(out, s...)
		case reflect.Slice:
			switch t.Elem().Kind() {
			case reflect.Uint8:
				if v.Type().Field(i).Tag.Get("ssh") != "rest" {
					out = appendInt(out, field.Len())
				}
				out = append(out, field.Bytes()...)
			case reflect.String:
				offset := len(out)
				out = appendU32(out, 0)
				if n := field.Len(); n > 0 {
					for j := 0; j < n; j++ {
						f := field.Index(j)
						if j != 0 {
							out = append(out, ',')
						}
						out = append(out, f.String()...)
					}
					// overwrite length value
					binary.BigEndian.PutUint32(out[offset:], uint32(len(out)-offset-4))
				}
			default:
				panic(fmt.Sprintf("slice of unknown type in field %d: %T", i, field.Interface()))
			}
		case reflect.Ptr:
			if t == bigIntType {
				var n *big.Int
				nValue := reflect.ValueOf(&n)
				nValue.Elem().Set(field)
				needed := intLength(n)
				oldLength := len(out)

				if cap(out)-len(out) < needed {
					newOut := make([]byte, len(out), 2*(len(out)+needed))
					copy(newOut, out)
					out = newOut
				}
				out = out[:oldLength+needed]
				marshalInt(out[oldLength:], n)
			} else {
				panic(fmt.Sprintf("pointer to unknown type in field %d: %T", i, field.Interface()))
			}
		}
	}

	return out
}

var bigOne = big.NewInt(1)

func parseString(in []byte) (out, rest []byte, ok bool) {
	if len(in) < 4 {
		return
	}
	length := binary.BigEndian.Uint32(in)
	in = in[4:]
	if uint32(len(in)) < length {
		return
	}
	out = in[:length]
	rest = in[length:]
	ok = true
	return
}

var (
	comma         = []byte{','}
	emptyNameList = []string{}
)

func parseNameList(in []byte) (out []string, rest []byte, ok bool) {
	contents, rest, ok := parseString(in)
	if !ok {
		return
	}
	if len(contents) == 0 {
		out = emptyNameList
		return
	}
	parts := bytes.Split(contents, comma)
	out = make([]string, len(parts))
	for i, part := range parts {
		out[i] = string(part)
	}
	return
}

func parseInt(in []byte) (out *big.Int, rest []byte, ok bool) {
	contents, rest, ok := parseString(in)
	if !ok {
		return
	}
	out = new(big.Int)

	if len(contents) > 0 && contents[0]&0x80 == 0x80 {
		// This is a negative number
		notBytes := make([]byte, len(contents))
		for i := range notBytes {
			notBytes[i] = ^contents[i]
		}
		out.SetBytes(notBytes)
		out.Add(out, bigOne)
		out.Neg(out)
	} else {
		// Positive number
		out.SetBytes(contents)
	}
	ok = true
	return
}

func parseUint32(in []byte) (uint32, []byte, bool) {
	if len(in) < 4 {
		return 0, nil, false
	}
	return binary.BigEndian.Uint32(in), in[4:], true
}

func parseUint64(in []byte) (uint64, []byte, bool) {
	if len(in) < 8 {
		return 0, nil, false
	}
	return binary.BigEndian.Uint64(in), in[8:], true
}

func intLength(n *big.Int) int {
	length := 4 /* length bytes */
	if n.Sign() < 0 {
		nMinus1 := new(big.Int).Neg(n)
		nMinus1.Sub(nMinus1, bigOne)
		bitLen := nMinus1.BitLen()
		if bitLen%8 == 0 {
			// The number will need 0xff padding
			length++
		}
		length += (bitLen + 7) / 8
	} else if n.Sign() == 0 {
		// A zero is the zero length string
	} else {
		bitLen := n.BitLen()
		if bitLen%8 == 0 {
			// The number will need 0x00 padding
			length++
		}
		length += (bitLen + 7) / 8
	}

	return length
}

func marshalUint32(to []byte, n uint32) []byte {
	binary.BigEndian.PutUint32(to, n)
	return to[4:]
}

func marshalUint64(to []byte, n uint64) []byte {
	binary.BigEndian.PutUint64(to, n)
	return to[8:]
}

func marshalInt(to []byte, n *big.Int) []byte {
	lengthBytes := to
	to = to[4:]
	length := 0

	if n.Sign() < 0 {
		// A negative number has to be converted to two's-complement
		// form. So we'll subtract 1 and invert. If the
		// most-significant-bit isn't set then we'll need to pad the
		// beginning with 0xff in order to keep the number negative.
		nMinus1 := new(big.Int).Neg(n)
		nMinus1.Sub(nMinus1, bigOne)
		bytes := nMinus1.Bytes()
		for i := range bytes {
			bytes[i] ^= 0xff
		}
		if len(bytes) == 0 || bytes[0]&0x80 == 0 {
			to[0] = 0xff
			to = to[1:]
			length++
		}
		nBytes := copy(to, bytes)
		to = to[nBytes:]
		length += nBytes
	} else if n.Sign() == 0 {
		// A zero is the zero length string
	} else {
		bytes := n.Bytes()
		if len(bytes) > 0 && bytes[0]&0x80 != 0 {
			// We'll have to pad this with a 0x00 in order to
			// stop it looking like a negative number.
			to[0] = 0
			to = to[1:]
			length++
		}
		nBytes := copy(to, bytes)
		to = to[nBytes:]
		length += nBytes
	}

	lengthBytes[0] = byte(length >> 24)
	lengthBytes[1] = byte(length >> 16)
	lengthBytes[2] = byte(length >> 8)
	lengthBytes[3] = byte(length)
	return to
}

func writeInt(w io.Writer, n *big.Int) {
	length := intLength(n)
	buf := make([]byte, length)
	marshalInt(buf, n)
	w.Write(buf)
}

func writeString(w io.Writer, s []byte) {
	var lengthBytes [4]byte
	lengthBytes[0] = byte(len(s) >> 24)
	lengthBytes[1] = byte(len(s) >> 16)
	lengthBytes[2] = byte(len(s) >> 8)
	lengthBytes[3] = byte(len(s))
	w.Write(lengthBytes[:])
	w.Write(s)
}

func stringLength(n int) int {
	return 4 + n
}

func marshalString(to []byte, s []byte) []byte {
	to[0] = byte(len(s) >> 24)
	to[1] = byte(len(s) >> 16)
	to[2] = byte(len(s) >> 8)
	to[3] = byte(len(s))
	to = to[4:]
	copy(to, s)
	return to[len(s):]
}

var bigIntType = reflect.TypeOf((*big.Int)(nil))

// Decode a packet into its corresponding message.
func decode(packet []byte) (interface{}, error) {
	var msg interface{}
	switch packet[0] {
	case msgDisconnect:
		msg = new(disconnectMsg)
	case msgServiceRequest:
		msg = new(serviceRequestMsg)
	case msgServiceAccept:
		msg = new(serviceAcceptMsg)
	case msgKexInit:
		msg = new(kexInitMsg)
	case msgKexDHInit:
		msg = new(kexDHInitMsg)
	case msgKexDHReply:
		msg = new(kexDHReplyMsg)
	case msgUserAuthRequest:
		msg = new(userAuthRequestMsg)
	case msgUserAuthSuccess:
		return new(userAuthSuccessMsg), nil
	case msgUserAuthFailure:
		msg = new(userAuthFailureMsg)
	case msgUserAuthPubKeyOk:
		msg = new(userAuthPubKeyOkMsg)
	case msgGlobalRequest:
		msg = new(globalRequestMsg)
	case msgRequestSuccess:
		msg = new(globalRequestSuccessMsg)
	case msgRequestFailure:
		msg = new(globalRequestFailureMsg)
	case msgChannelOpen:
		msg = new(channelOpenMsg)
	case msgChannelData:
		msg = new(channelDataMsg)
	case msgChannelOpenConfirm:
		msg = new(channelOpenConfirmMsg)
	case msgChannelOpenFailure:
		msg = new(channelOpenFailureMsg)
	case msgChannelWindowAdjust:
		msg = new(windowAdjustMsg)
	case msgChannelEOF:
		msg = new(channelEOFMsg)
	case msgChannelClose:
		msg = new(channelCloseMsg)
	case msgChannelRequest:
		msg = new(channelRequestMsg)
	case msgChannelSuccess:
		msg = new(channelRequestSuccessMsg)
	case msgChannelFailure:
		msg = new(channelRequestFailureMsg)
	default:
		return nil, unexpectedMessageError(0, packet[0])
	}
	if err := Unmarshal(packet, msg); err != nil {
		return nil, err
	}
	return msg, nil
}
