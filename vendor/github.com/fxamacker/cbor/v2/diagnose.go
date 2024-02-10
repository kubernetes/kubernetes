// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"bytes"
	"encoding/base32"
	"encoding/base64"
	"encoding/hex"
	"errors"
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
func (dm *diagMode) DiagnoseFirst(data []byte) (string, []byte, error) {
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
func DiagnoseFirst(data []byte) (string, []byte, error) {
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
				if err = di.writeString(", "); err != nil {
					return di.w.String(), err
				}
			}
			firstItem = false
			if err = di.item(); err != nil {
				return di.w.String(), err
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

func (di *diagnose) diagFirst() (string, []byte, error) {
	err := di.wellformed(true)
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
	err := di.d.wellformed(allowExtraData)
	di.d.off = off
	return err
}

func (di *diagnose) item() error { //nolint:gocyclo
	initialByte := di.d.data[di.d.off]
	switch initialByte {
	case 0x5f, 0x7f: // indefinite-length byte/text string
		di.d.off++
		if di.d.data[di.d.off] == 0xff {
			di.d.off++
			switch initialByte {
			case 0x5f:
				// indefinite-length bytes with no chunks.
				return di.writeString(`''_`)
			case 0x7f:
				// indefinite-length text with no chunks.
				return di.writeString(`""_`)
			}
		}

		if err := di.writeString("(_ "); err != nil {
			return err
		}

		i := 0
		for !di.d.foundBreak() {
			if i > 0 {
				if err := di.writeString(", "); err != nil {
					return err
				}
			}

			i++
			// wellformedIndefiniteString() already checked that the next item is a byte/text string.
			if err := di.item(); err != nil {
				return err
			}
		}

		return di.writeByte(')')

	case 0x9f: // indefinite-length array
		di.d.off++
		if err := di.writeString("[_ "); err != nil {
			return err
		}

		i := 0
		for !di.d.foundBreak() {
			if i > 0 {
				if err := di.writeString(", "); err != nil {
					return err
				}
			}

			i++
			if err := di.item(); err != nil {
				return err
			}
		}

		return di.writeByte(']')

	case 0xbf: // indefinite-length map
		di.d.off++
		if err := di.writeString("{_ "); err != nil {
			return err
		}

		i := 0
		for !di.d.foundBreak() {
			if i > 0 {
				if err := di.writeString(", "); err != nil {
					return err
				}
			}

			i++
			// key
			if err := di.item(); err != nil {
				return err
			}

			if err := di.writeString(": "); err != nil {
				return err
			}

			// value
			if err := di.item(); err != nil {
				return err
			}
		}

		return di.writeByte('}')
	}

	t := di.d.nextCBORType()
	switch t {
	case cborTypePositiveInt:
		_, _, val := di.d.getHead()
		return di.writeString(strconv.FormatUint(val, 10))

	case cborTypeNegativeInt:
		_, _, val := di.d.getHead()
		if val > math.MaxInt64 {
			// CBOR negative integer overflows int64, use big.Int to store value.
			bi := new(big.Int)
			bi.SetUint64(val)
			bi.Add(bi, big.NewInt(1))
			bi.Neg(bi)
			return di.writeString(bi.String())
		}

		nValue := int64(-1) ^ int64(val)
		return di.writeString(strconv.FormatInt(nValue, 10))

	case cborTypeByteString:
		b := di.d.parseByteString()
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
		if err := di.writeByte('['); err != nil {
			return err
		}

		for i := 0; i < count; i++ {
			if i > 0 {
				if err := di.writeString(", "); err != nil {
					return err
				}
			}
			if err := di.item(); err != nil {
				return err
			}
		}
		return di.writeByte(']')

	case cborTypeMap:
		_, _, val := di.d.getHead()
		count := int(val)
		if err := di.writeByte('{'); err != nil {
			return err
		}

		for i := 0; i < count; i++ {
			if i > 0 {
				if err := di.writeString(", "); err != nil {
					return err
				}
			}
			// key
			if err := di.item(); err != nil {
				return err
			}
			if err := di.writeString(": "); err != nil {
				return err
			}
			// value
			if err := di.item(); err != nil {
				return err
			}
		}
		return di.writeByte('}')

	case cborTypeTag:
		_, _, tagNum := di.d.getHead()
		switch tagNum {
		case 2:
			if nt := di.d.nextCBORType(); nt != cborTypeByteString {
				return errors.New("cbor: tag number 2 must be followed by byte string, got " + nt.String())
			}

			b := di.d.parseByteString()
			bi := new(big.Int).SetBytes(b)
			return di.writeString(bi.String())

		case 3:
			if nt := di.d.nextCBORType(); nt != cborTypeByteString {
				return errors.New("cbor: tag number 3 must be followed by byte string, got " + nt.String())
			}

			b := di.d.parseByteString()
			bi := new(big.Int).SetBytes(b)
			bi.Add(bi, big.NewInt(1))
			bi.Neg(bi)
			return di.writeString(bi.String())

		default:
			if err := di.writeString(strconv.FormatUint(tagNum, 10)); err != nil {
				return err
			}
			if err := di.writeByte('('); err != nil {
				return err
			}
			if err := di.item(); err != nil {
				return err
			}
			return di.writeByte(')')
		}

	case cborTypePrimitives:
		_, ai, val := di.d.getHead()
		switch ai {
		case 20:
			return di.writeString("false")

		case 21:
			return di.writeString("true")

		case 22:
			return di.writeString("null")

		case 23:
			return di.writeString("undefined")

		case 25, 26, 27:
			return di.encodeFloat(ai, val)

		default:
			if err := di.writeString("simple("); err != nil {
				return err
			}
			if err := di.writeString(strconv.FormatUint(val, 10)); err != nil {
				return err
			}
			return di.writeByte(')')
		}
	}

	return nil
}

func (di *diagnose) writeByte(val byte) error {
	return di.w.WriteByte(val)
}

func (di *diagnose) writeString(val string) error {
	_, err := di.w.WriteString(val)
	return err
}

// writeU16 format a rune as "\uxxxx"
func (di *diagnose) writeU16(val rune) error {
	if err := di.writeString("\\u"); err != nil {
		return err
	}
	b := make([]byte, 2)
	b[0] = byte(val >> 8)
	b[1] = byte(val)
	return di.writeString(hex.EncodeToString(b))
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
				if err := di.writeString("<<"); err != nil {
					return err
				}
				if err := di.writeString(str); err != nil {
					return err
				}
				return di.writeString(">>")
			}
		}
	}

	switch di.dm.byteStringEncoding {
	case ByteStringBase16Encoding:
		if err := di.writeString("h'"); err != nil {
			return err
		}

		encoder := hex.NewEncoder(di.w)
		if di.dm.byteStringHexWhitespace {
			for i, b := range val {
				if i > 0 {
					if err := di.writeByte(' '); err != nil {
						return err
					}
				}
				if _, err := encoder.Write([]byte{b}); err != nil {
					return err
				}
			}
		} else {
			if _, err := encoder.Write(val); err != nil {
				return err
			}
		}
		return di.writeByte('\'')

	case ByteStringBase32Encoding:
		if err := di.writeString("b32'"); err != nil {
			return err
		}
		encoder := base32.NewEncoder(rawBase32Encoding, di.w)
		if _, err := encoder.Write(val); err != nil {
			return err
		}
		encoder.Close()
		return di.writeByte('\'')

	case ByteStringBase32HexEncoding:
		if err := di.writeString("h32'"); err != nil {
			return err
		}
		encoder := base32.NewEncoder(rawBase32HexEncoding, di.w)
		if _, err := encoder.Write(val); err != nil {
			return err
		}
		encoder.Close()
		return di.writeByte('\'')

	case ByteStringBase64Encoding:
		if err := di.writeString("b64'"); err != nil {
			return err
		}
		encoder := base64.NewEncoder(base64.RawURLEncoding, di.w)
		if _, err := encoder.Write(val); err != nil {
			return err
		}
		encoder.Close()
		return di.writeByte('\'')

	default:
		return di.dm.byteStringEncoding.valid()
	}
}

var utf16SurrSelf = rune(0x10000)

// quote should be either `'` or `"`
func (di *diagnose) encodeTextString(val string, quote byte) error {
	if err := di.writeByte(quote); err != nil {
		return err
	}

	for i := 0; i < len(val); {
		if b := val[i]; b < utf8.RuneSelf {
			switch {
			case b == '\t', b == '\n', b == '\r', b == '\\', b == quote:
				if err := di.writeByte('\\'); err != nil {
					return err
				}

				switch b {
				case '\t':
					b = 't'
				case '\n':
					b = 'n'
				case '\r':
					b = 'r'
				}
				if err := di.writeByte(b); err != nil {
					return err
				}

			case b >= ' ' && b <= '~':
				if err := di.writeByte(b); err != nil {
					return err
				}

			default:
				if err := di.writeU16(rune(b)); err != nil {
					return err
				}
			}

			i++
			continue
		}

		c, size := utf8.DecodeRuneInString(val[i:])
		switch {
		case c == utf8.RuneError:
			// if err := di.writeU16(rune(val[i])); err != nil {
			// 	return err
			// }
			return &SemanticError{"cbor: invalid UTF-8 string"}

		case c < utf16SurrSelf:
			if err := di.writeU16(c); err != nil {
				return err
			}

		default:
			c1, c2 := utf16.EncodeRune(c)
			if err := di.writeU16(c1); err != nil {
				return err
			}
			if err := di.writeU16(c2); err != nil {
				return err
			}
		}

		i += size
	}

	return di.writeByte(quote)
}

func (di *diagnose) encodeFloat(ai byte, val uint64) error {
	f64 := float64(0)
	switch ai {
	case 25:
		f16 := float16.Frombits(uint16(val))
		switch {
		case f16.IsNaN():
			return di.writeString("NaN")
		case f16.IsInf(1):
			return di.writeString("Infinity")
		case f16.IsInf(-1):
			return di.writeString("-Infinity")
		default:
			f64 = float64(f16.Float32())
		}

	case 26:
		f32 := math.Float32frombits(uint32(val))
		switch {
		case f32 != f32:
			return di.writeString("NaN")
		case f32 > math.MaxFloat32:
			return di.writeString("Infinity")
		case f32 < -math.MaxFloat32:
			return di.writeString("-Infinity")
		default:
			f64 = float64(f32)
		}

	case 27:
		f64 = math.Float64frombits(val)
		switch {
		case f64 != f64:
			return di.writeString("NaN")
		case f64 > math.MaxFloat64:
			return di.writeString("Infinity")
		case f64 < -math.MaxFloat64:
			return di.writeString("-Infinity")
		}
	}
	// Use ES6 number to string conversion which should match most JSON generators.
	// Inspired by https://github.com/golang/go/blob/4df10fba1687a6d4f51d7238a403f8f2298f6a16/src/encoding/json/encode.go#L585
	b := make([]byte, 0, 32)
	if abs := math.Abs(f64); abs != 0 && (abs < 1e-6 || abs >= 1e21) {
		b = strconv.AppendFloat(b, f64, 'e', -1, 64)
		// clean up e-09 to e-9
		n := len(b)
		if n >= 4 && string(b[n-4:n-1]) == "e-0" {
			b = append(b[:n-2], b[n-1])
		}
	} else {
		b = strconv.AppendFloat(b, f64, 'f', -1, 64)
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

	if err := di.writeString(string(b)); err != nil {
		return err
	}

	if di.dm.floatPrecisionIndicator {
		switch ai {
		case 25:
			return di.writeString("_1")
		case 26:
			return di.writeString("_2")
		case 27:
			return di.writeString("_3")
		}
	}

	return nil
}
