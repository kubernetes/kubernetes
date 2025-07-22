// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"bytes"
	"encoding/base32"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/x448/float16"
)

// DiagMode is the main interface for CBOR diagnostic notation.
type DiagMode interface {
	// Diagnose returns extended diagnostic notation (EDN) of CBOR data items using this DiagMode.
	Diagnose([]byte) (string, error)

	// DiagnoseFirst returns extended diagnostic notation (EDN) of the first CBOR data item using the DiagMode. Any remaining bytes are returned in rest.
	DiagnoseFirst([]byte) (string, []byte, error)

	// DiagOptions returns user specified options used to create this DiagMode.
	DiagOptions() DiagOptions
}

// ByteStringEncoding specifies the base encoding that byte strings are notated.
type ByteStringEncoding uint8

const (
	// ByteStringBase16Encoding encodes byte strings in base16, without padding.
	ByteStringBase16Encoding ByteStringEncoding = iota

	// ByteStringBase32Encoding encodes byte strings in base32, without padding.
	ByteStringBase32Encoding

	// ByteStringBase32HexEncoding encodes byte strings in base32hex, without padding.
	ByteStringBase32HexEncoding

	// ByteStringBase64Encoding encodes byte strings in base64url, without padding.
	ByteStringBase64Encoding

	maxByteStringEncoding
)

func (bse ByteStringEncoding) valid() error {
	if bse >= maxByteStringEncoding {
		return errors.New("cbor: invalid ByteStringEncoding " + strconv.Itoa(int(bse)))
	}
	return nil
}

// DiagOptions specifies Diag options.
type DiagOptions struct {
	// ByteStringEncoding specifies the base encoding that byte strings are notated.
	// Default is ByteStringBase16Encoding.
	ByteStringEncoding ByteStringEncoding

	// ByteStringHexWhitespace specifies notating with whitespace in byte string
	// when ByteStringEncoding is ByteStringBase16Encoding.
	ByteStringHexWhitespace bool

	// ByteStringText specifies notating with text in byte string
	// if it is a valid UTF-8 text.
	ByteStringText bool

	// ByteStringEmbeddedCBOR specifies notating embedded CBOR in byte string
	// if it is a valid CBOR bytes.
	ByteStringEmbeddedCBOR bool

	// CBORSequence specifies notating CBOR sequences.
	// otherwise, it returns an error if there are more bytes after the first CBOR.
	CBORSequence bool

	// FloatPrecisionIndicator specifies appending a suffix to indicate float precision.
	// Refer to https://www.rfc-editor.org/rfc/rfc8949.html#name-encoding-indicators.
	FloatPrecisionIndicator bool

	// MaxNestedLevels specifies the max nested levels allowed for any combination of CBOR array, maps, and tags.
	// Default is 32 levels and it can be set to [4, 65535]. Note that higher maximum levels of nesting can
	// require larger amounts of stack to deserialize. Don't increase this higher than you require.
	MaxNestedLevels int

	// MaxArrayElements specifies the max number of elements for CBOR arrays.
	// Default is 128*1024=131072 and it can be set to [16, 2147483647]
	MaxArrayElements int

	// MaxMapPairs specifies the max number of key-value pairs for CBOR maps.
	// Default is 128*1024=131072 and it can be set to [16, 2147483647]
	MaxMapPairs int
}

// DiagMode returns a DiagMode with immutable options.
func (opts DiagOptions) DiagMode() (DiagMode, error) {
	return opts.diagMode()
}

func (opts DiagOptions) diagMode() (*diagMode, error) {
	if err := opts.ByteStringEncoding.valid(); err != nil {
		return nil, err
	}

	decMode, err := DecOptions{
		MaxNestedLevels:  opts.MaxNestedLevels,
		MaxArrayElements: opts.MaxArrayElements,
		MaxMapPairs:      opts.MaxMapPairs,
	}.decMode()
	if err != nil {
		return nil, err
	}

	return &diagMode{
		byteStringEncoding:      opts.ByteStringEncoding,
		byteStringHexWhitespace: opts.ByteStringHexWhitespace,
		byteStringText:          opts.ByteStringText,
		byteStringEmbeddedCBOR:  opts.ByteStringEmbeddedCBOR,
		cborSequence:            opts.CBORSequence,
		floatPrecisionIndicator: opts.FloatPrecisionIndicator,
		decMode:                 decMode,
	}, nil
}

type diagMode struct {
	byteStringEncoding      ByteStringEncoding
	byteStringHexWhitespace bool
	byteStringText          bool
	byteStringEmbeddedCBOR  bool
	cborSequence            bool
	floatPrecisionIndicator bool
	decMode                 *decMode
}

// DiagOptions returns user specified options used to create this DiagMode.
func (dm *diagMode) DiagOptions() DiagOptions {
	return DiagOptions{
		ByteStringEncoding:      dm.byteStringEncoding,
		ByteStringHexWhitespace: dm.byteStringHexWhitespace,
		ByteStringText:          dm.byteStringText,
		ByteStringEmbeddedCBOR:  dm.byteStringEmbeddedCBOR,
		CBORSequence:            dm.cborSequence,
		FloatPrecisionIndicator: dm.floatPrecisionIndicator,
		MaxNestedLevels:         dm.decMode.maxNestedLevels,
		MaxArrayElements:        dm.decMode.maxArrayElements,
		MaxMapPairs:             dm.decMode.maxMapPairs,
	}
}

// Diagnose returns extended diagnostic notation (EDN) of CBOR data items using the DiagMode.
func (dm *diagMode) Diagnose(data []byte) (string, error) {
	return newDiagnose(data, dm.decMode, dm).diag(dm.cborSequence)
}

// DiagnoseFirst returns extended diagnostic notation (EDN) of the first CBOR data item using the DiagMode. Any remaining bytes are returned in rest.
func (dm *diagMode) DiagnoseFirst(data []byte) (diagNotation string, rest []byte, err error) {
	return newDiagnose(data, dm.decMode, dm).diagFirst()
}

var defaultDiagMode, _ = DiagOptions{}.diagMode()

// Diagnose returns extended diagnostic notation (EDN) of CBOR data items
// using the default diagnostic mode.
//
// Refer to https://www.rfc-editor.org/rfc/rfc8949.html#name-diagnostic-notation.
func Diagnose(data []byte) (string, error) {
	return defaultDiagMode.Diagnose(data)
}

// Diagnose returns extended diagnostic notation (EDN) of the first CBOR data item using the DiagMode. Any remaining bytes are returned in rest.
func DiagnoseFirst(data []byte) (diagNotation string, rest []byte, err error) {
	return defaultDiagMode.DiagnoseFirst(data)
}

type diagnose struct {
	dm *diagMode
	d  *decoder
	w  *bytes.Buffer
}

func newDiagnose(data []byte, decm *decMode, diagm *diagMode) *diagnose {
	return &diagnose{
		dm: diagm,
		d:  &decoder{data: data, dm: decm},
		w:  &bytes.Buffer{},
	}
}

func (di *diagnose) diag(cborSequence bool) (string, error) {
	// CBOR Sequence
	firstItem := true
	for {
		switch err := di.wellformed(cborSequence); err {
		case nil:
			if !firstItem {
				di.w.WriteString(", ")
			}
			firstItem = false
			if itemErr := di.item(); itemErr != nil {
				return di.w.String(), itemErr
			}

		case io.EOF:
			if firstItem {
				return di.w.String(), err
			}
			return di.w.String(), nil

		default:
			return di.w.String(), err
		}
	}
}

func (di *diagnose) diagFirst() (diagNotation string, rest []byte, err error) {
	err = di.wellformed(true)
	if err == nil {
		err = di.item()
	}

	if err == nil {
		// Return EDN and the rest of the data slice (which might be len 0)
		return di.w.String(), di.d.data[di.d.off:], nil
	}

	return di.w.String(), nil, err
}

func (di *diagnose) wellformed(allowExtraData bool) error {
	off := di.d.off
	err := di.d.wellformed(allowExtraData, false)
	di.d.off = off
	return err
}

func (di *diagnose) item() error { //nolint:gocyclo
	initialByte := di.d.data[di.d.off]
	switch initialByte {
	case cborByteStringWithIndefiniteLengthHead,
		cborTextStringWithIndefiniteLengthHead: // indefinite-length byte/text string
		di.d.off++
		if isBreakFlag(di.d.data[di.d.off]) {
			di.d.off++
			switch initialByte {
			case cborByteStringWithIndefiniteLengthHead:
				// indefinite-length bytes with no chunks.
				di.w.WriteString(`''_`)
				return nil
			case cborTextStringWithIndefiniteLengthHead:
				// indefinite-length text with no chunks.
				di.w.WriteString(`""_`)
				return nil
			}
		}

		di.w.WriteString("(_ ")

		i := 0
		for !di.d.foundBreak() {
			if i > 0 {
				di.w.WriteString(", ")
			}

			i++
			// wellformedIndefiniteString() already checked that the next item is a byte/text string.
			if err := di.item(); err != nil {
				return err
			}
		}

		di.w.WriteByte(')')
		return nil

	case cborArrayWithIndefiniteLengthHead: // indefinite-length array
		di.d.off++
		di.w.WriteString("[_ ")

		i := 0
		for !di.d.foundBreak() {
			if i > 0 {
				di.w.WriteString(", ")
			}

			i++
			if err := di.item(); err != nil {
				return err
			}
		}

		di.w.WriteByte(']')
		return nil

	case cborMapWithIndefiniteLengthHead: // indefinite-length map
		di.d.off++
		di.w.WriteString("{_ ")

		i := 0
		for !di.d.foundBreak() {
			if i > 0 {
				di.w.WriteString(", ")
			}

			i++
			// key
			if err := di.item(); err != nil {
				return err
			}

			di.w.WriteString(": ")

			// value
			if err := di.item(); err != nil {
				return err
			}
		}

		di.w.WriteByte('}')
		return nil
	}

	t := di.d.nextCBORType()
	switch t {
	case cborTypePositiveInt:
		_, _, val := di.d.getHead()
		di.w.WriteString(strconv.FormatUint(val, 10))
		return nil

	case cborTypeNegativeInt:
		_, _, val := di.d.getHead()
		if val > math.MaxInt64 {
			// CBOR negative integer overflows int64, use big.Int to store value.
			bi := new(big.Int)
			bi.SetUint64(val)
			bi.Add(bi, big.NewInt(1))
			bi.Neg(bi)
			di.w.WriteString(bi.String())
			return nil
		}

		nValue := int64(-1) ^ int64(val)
		di.w.WriteString(strconv.FormatInt(nValue, 10))
		return nil

	case cborTypeByteString:
		b, _ := di.d.parseByteString()
		return di.encodeByteString(b)

	case cborTypeTextString:
		b, err := di.d.parseTextString()
		if err != nil {
			return err
		}
		return di.encodeTextString(string(b), '"')

	case cborTypeArray:
		_, _, val := di.d.getHead()
		count := int(val)
		di.w.WriteByte('[')

		for i := 0; i < count; i++ {
			if i > 0 {
				di.w.WriteString(", ")
			}
			if err := di.item(); err != nil {
				return err
			}
		}
		di.w.WriteByte(']')
		return nil

	case cborTypeMap:
		_, _, val := di.d.getHead()
		count := int(val)
		di.w.WriteByte('{')

		for i := 0; i < count; i++ {
			if i > 0 {
				di.w.WriteString(", ")
			}
			// key
			if err := di.item(); err != nil {
				return err
			}
			di.w.WriteString(": ")
			// value
			if err := di.item(); err != nil {
				return err
			}
		}
		di.w.WriteByte('}')
		return nil

	case cborTypeTag:
		_, _, tagNum := di.d.getHead()
		switch tagNum {
		case tagNumUnsignedBignum:
			if nt := di.d.nextCBORType(); nt != cborTypeByteString {
				return newInadmissibleTagContentTypeError(
					tagNumUnsignedBignum,
					"byte string",
					nt.String())
			}

			b, _ := di.d.parseByteString()
			bi := new(big.Int).SetBytes(b)
			di.w.WriteString(bi.String())
			return nil

		case tagNumNegativeBignum:
			if nt := di.d.nextCBORType(); nt != cborTypeByteString {
				return newInadmissibleTagContentTypeError(
					tagNumNegativeBignum,
					"byte string",
					nt.String(),
				)
			}

			b, _ := di.d.parseByteString()
			bi := new(big.Int).SetBytes(b)
			bi.Add(bi, big.NewInt(1))
			bi.Neg(bi)
			di.w.WriteString(bi.String())
			return nil

		default:
			di.w.WriteString(strconv.FormatUint(tagNum, 10))
			di.w.WriteByte('(')
			if err := di.item(); err != nil {
				return err
			}
			di.w.WriteByte(')')
			return nil
		}

	case cborTypePrimitives:
		_, ai, val := di.d.getHead()
		switch ai {
		case additionalInformationAsFalse:
			di.w.WriteString("false")
			return nil

		case additionalInformationAsTrue:
			di.w.WriteString("true")
			return nil

		case additionalInformationAsNull:
			di.w.WriteString("null")
			return nil

		case additionalInformationAsUndefined:
			di.w.WriteString("undefined")
			return nil

		case additionalInformationAsFloat16,
			additionalInformationAsFloat32,
			additionalInformationAsFloat64:
			return di.encodeFloat(ai, val)

		default:
			di.w.WriteString("simple(")
			di.w.WriteString(strconv.FormatUint(val, 10))
			di.w.WriteByte(')')
			return nil
		}
	}

	return nil
}

// writeU16 format a rune as "\uxxxx"
func (di *diagnose) writeU16(val rune) {
	di.w.WriteString("\\u")
	var in [2]byte
	in[0] = byte(val >> 8)
	in[1] = byte(val)
	sz := hex.EncodedLen(len(in))
	di.w.Grow(sz)
	dst := di.w.Bytes()[di.w.Len() : di.w.Len()+sz]
	hex.Encode(dst, in[:])
	di.w.Write(dst)
}

var rawBase32Encoding = base32.StdEncoding.WithPadding(base32.NoPadding)
var rawBase32HexEncoding = base32.HexEncoding.WithPadding(base32.NoPadding)

func (di *diagnose) encodeByteString(val []byte) error {
	if len(val) > 0 {
		if di.dm.byteStringText && utf8.Valid(val) {
			return di.encodeTextString(string(val), '\'')
		}

		if di.dm.byteStringEmbeddedCBOR {
			di2 := newDiagnose(val, di.dm.decMode, di.dm)
			// should always notating embedded CBOR sequence.
			if str, err := di2.diag(true); err == nil {
				di.w.WriteString("<<")
				di.w.WriteString(str)
				di.w.WriteString(">>")
				return nil
			}
		}
	}

	switch di.dm.byteStringEncoding {
	case ByteStringBase16Encoding:
		di.w.WriteString("h'")
		if di.dm.byteStringHexWhitespace {
			sz := hex.EncodedLen(len(val))
			if len(val) > 0 {
				sz += len(val) - 1
			}
			di.w.Grow(sz)

			dst := di.w.Bytes()[di.w.Len():]
			for i := range val {
				if i > 0 {
					dst = append(dst, ' ')
				}
				hex.Encode(dst[len(dst):len(dst)+2], val[i:i+1])
				dst = dst[:len(dst)+2]
			}
			di.w.Write(dst)
		} else {
			sz := hex.EncodedLen(len(val))
			di.w.Grow(sz)
			dst := di.w.Bytes()[di.w.Len() : di.w.Len()+sz]
			hex.Encode(dst, val)
			di.w.Write(dst)
		}
		di.w.WriteByte('\'')
		return nil

	case ByteStringBase32Encoding:
		di.w.WriteString("b32'")
		sz := rawBase32Encoding.EncodedLen(len(val))
		di.w.Grow(sz)
		dst := di.w.Bytes()[di.w.Len() : di.w.Len()+sz]
		rawBase32Encoding.Encode(dst, val)
		di.w.Write(dst)
		di.w.WriteByte('\'')
		return nil

	case ByteStringBase32HexEncoding:
		di.w.WriteString("h32'")
		sz := rawBase32HexEncoding.EncodedLen(len(val))
		di.w.Grow(sz)
		dst := di.w.Bytes()[di.w.Len() : di.w.Len()+sz]
		rawBase32HexEncoding.Encode(dst, val)
		di.w.Write(dst)
		di.w.WriteByte('\'')
		return nil

	case ByteStringBase64Encoding:
		di.w.WriteString("b64'")
		sz := base64.RawURLEncoding.EncodedLen(len(val))
		di.w.Grow(sz)
		dst := di.w.Bytes()[di.w.Len() : di.w.Len()+sz]
		base64.RawURLEncoding.Encode(dst, val)
		di.w.Write(dst)
		di.w.WriteByte('\'')
		return nil

	default:
		// It should not be possible for users to construct a *diagMode with an invalid byte
		// string encoding.
		panic(fmt.Sprintf("diagmode has invalid ByteStringEncoding %v", di.dm.byteStringEncoding))
	}
}

const utf16SurrSelf = rune(0x10000)

// quote should be either `'` or `"`
func (di *diagnose) encodeTextString(val string, quote byte) error {
	di.w.WriteByte(quote)

	for i := 0; i < len(val); {
		if b := val[i]; b < utf8.RuneSelf {
			switch {
			case b == '\t', b == '\n', b == '\r', b == '\\', b == quote:
				di.w.WriteByte('\\')

				switch b {
				case '\t':
					b = 't'
				case '\n':
					b = 'n'
				case '\r':
					b = 'r'
				}
				di.w.WriteByte(b)

			case b >= ' ' && b <= '~':
				di.w.WriteByte(b)

			default:
				di.writeU16(rune(b))
			}

			i++
			continue
		}

		c, size := utf8.DecodeRuneInString(val[i:])
		switch {
		case c == utf8.RuneError:
			return &SemanticError{"cbor: invalid UTF-8 string"}

		case c < utf16SurrSelf:
			di.writeU16(c)

		default:
			c1, c2 := utf16.EncodeRune(c)
			di.writeU16(c1)
			di.writeU16(c2)
		}

		i += size
	}

	di.w.WriteByte(quote)
	return nil
}

func (di *diagnose) encodeFloat(ai byte, val uint64) error {
	f64 := float64(0)
	switch ai {
	case additionalInformationAsFloat16:
		f16 := float16.Frombits(uint16(val))
		switch {
		case f16.IsNaN():
			di.w.WriteString("NaN")
			return nil
		case f16.IsInf(1):
			di.w.WriteString("Infinity")
			return nil
		case f16.IsInf(-1):
			di.w.WriteString("-Infinity")
			return nil
		default:
			f64 = float64(f16.Float32())
		}

	case additionalInformationAsFloat32:
		f32 := math.Float32frombits(uint32(val))
		switch {
		case f32 != f32:
			di.w.WriteString("NaN")
			return nil
		case f32 > math.MaxFloat32:
			di.w.WriteString("Infinity")
			return nil
		case f32 < -math.MaxFloat32:
			di.w.WriteString("-Infinity")
			return nil
		default:
			f64 = float64(f32)
		}

	case additionalInformationAsFloat64:
		f64 = math.Float64frombits(val)
		switch {
		case f64 != f64:
			di.w.WriteString("NaN")
			return nil
		case f64 > math.MaxFloat64:
			di.w.WriteString("Infinity")
			return nil
		case f64 < -math.MaxFloat64:
			di.w.WriteString("-Infinity")
			return nil
		}
	}
	// Use ES6 number to string conversion which should match most JSON generators.
	// Inspired by https://github.com/golang/go/blob/4df10fba1687a6d4f51d7238a403f8f2298f6a16/src/encoding/json/encode.go#L585
	const bitSize = 64
	b := make([]byte, 0, 32)
	if abs := math.Abs(f64); abs != 0 && (abs < 1e-6 || abs >= 1e21) {
		b = strconv.AppendFloat(b, f64, 'e', -1, bitSize)
		// clean up e-09 to e-9
		n := len(b)
		if n >= 4 && string(b[n-4:n-1]) == "e-0" {
			b = append(b[:n-2], b[n-1])
		}
	} else {
		b = strconv.AppendFloat(b, f64, 'f', -1, bitSize)
	}

	// add decimal point and trailing zero if needed
	if bytes.IndexByte(b, '.') < 0 {
		if i := bytes.IndexByte(b, 'e'); i < 0 {
			b = append(b, '.', '0')
		} else {
			b = append(b[:i+2], b[i:]...)
			b[i] = '.'
			b[i+1] = '0'
		}
	}

	di.w.WriteString(string(b))

	if di.dm.floatPrecisionIndicator {
		switch ai {
		case additionalInformationAsFloat16:
			di.w.WriteString("_1")
			return nil

		case additionalInformationAsFloat32:
			di.w.WriteString("_2")
			return nil

		case additionalInformationAsFloat64:
			di.w.WriteString("_3")
			return nil
		}
	}

	return nil
}
