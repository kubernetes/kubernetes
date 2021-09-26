// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bitfield converts annotated structs into integer values.
//
// Any field that is marked with a bitfield tag is compacted. The tag value has
// two parts. The part before the comma determines the method name for a
// generated type. If left blank the name of the field is used.
// The part after the comma determines the number of bits to use for the
// representation.
package bitfield

import (
	"bytes"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
)

// Config determines settings for packing and generation. If a Config is used,
// the same Config should be used for packing and generation.
type Config struct {
	// NumBits fixes the maximum allowed bits for the integer representation.
	// If NumBits is not 8, 16, 32, or 64, the actual underlying integer size
	// will be the next largest available.
	NumBits uint

	// If Package is set, code generation will write a package clause.
	Package string

	// TypeName is the name for the generated type. By default it is the name
	// of the type of the value passed to Gen.
	TypeName string
}

var nullConfig = &Config{}

// Pack packs annotated bit ranges of struct x in an integer.
//
// Only fields that have a "bitfield" tag are compacted.
func Pack(x interface{}, c *Config) (packed uint64, err error) {
	packed, _, err = pack(x, c)
	return
}

func pack(x interface{}, c *Config) (packed uint64, nBit uint, err error) {
	if c == nil {
		c = nullConfig
	}
	nBits := c.NumBits
	v := reflect.ValueOf(x)
	v = reflect.Indirect(v)
	t := v.Type()
	pos := 64 - nBits
	if nBits == 0 {
		pos = 0
	}
	for i := 0; i < v.NumField(); i++ {
		v := v.Field(i)
		field := t.Field(i)
		f, err := parseField(field)

		if err != nil {
			return 0, 0, err
		}
		if f.nBits == 0 {
			continue
		}
		value := uint64(0)
		switch v.Kind() {
		case reflect.Bool:
			if v.Bool() {
				value = 1
			}
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			value = v.Uint()
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			x := v.Int()
			if x < 0 {
				return 0, 0, fmt.Errorf("bitfield: negative value for field %q not allowed", field.Name)
			}
			value = uint64(x)
		}
		if value > (1<<f.nBits)-1 {
			return 0, 0, fmt.Errorf("bitfield: value %#x of field %q does not fit in %d bits", value, field.Name, f.nBits)
		}
		shift := 64 - pos - f.nBits
		if pos += f.nBits; pos > 64 {
			return 0, 0, fmt.Errorf("bitfield: no more bits left for field %q", field.Name)
		}
		packed |= value << shift
	}
	if nBits == 0 {
		nBits = posToBits(pos)
		packed >>= (64 - nBits)
	}
	return packed, nBits, nil
}

type field struct {
	name  string
	value uint64
	nBits uint
}

// parseField parses a tag of the form [<name>][:<nBits>][,<pos>[..<end>]]
func parseField(field reflect.StructField) (f field, err error) {
	s, ok := field.Tag.Lookup("bitfield")
	if !ok {
		return f, nil
	}
	switch field.Type.Kind() {
	case reflect.Bool:
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
	default:
		return f, fmt.Errorf("bitfield: field %q is not an integer or bool type", field.Name)
	}
	bits := s
	f.name = ""

	if i := strings.IndexByte(s, ','); i >= 0 {
		bits = s[:i]
		f.name = s[i+1:]
	}
	if bits != "" {
		nBits, err := strconv.ParseUint(bits, 10, 8)
		if err != nil {
			return f, fmt.Errorf("bitfield: invalid bit size for field %q: %v", field.Name, err)
		}
		f.nBits = uint(nBits)
	}
	if f.nBits == 0 {
		if field.Type.Kind() == reflect.Bool {
			f.nBits = 1
		} else {
			f.nBits = uint(field.Type.Bits())
		}
	}
	if f.name == "" {
		f.name = field.Name
	}
	return f, err
}

func posToBits(pos uint) (bits uint) {
	switch {
	case pos <= 8:
		bits = 8
	case pos <= 16:
		bits = 16
	case pos <= 32:
		bits = 32
	case pos <= 64:
		bits = 64
	default:
		panic("unreachable")
	}
	return bits
}

// Gen generates code for unpacking integers created with Pack.
func Gen(w io.Writer, x interface{}, c *Config) error {
	if c == nil {
		c = nullConfig
	}
	_, nBits, err := pack(x, c)
	if err != nil {
		return err
	}

	t := reflect.TypeOf(x)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if c.TypeName == "" {
		c.TypeName = t.Name()
	}
	firstChar := []rune(c.TypeName)[0]

	buf := &bytes.Buffer{}

	print := func(w io.Writer, format string, args ...interface{}) {
		if _, e := fmt.Fprintf(w, format+"\n", args...); e != nil && err == nil {
			err = fmt.Errorf("bitfield: write failed: %v", err)
		}
	}

	pos := uint(0)
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		f, _ := parseField(field)
		if f.nBits == 0 {
			continue
		}
		shift := nBits - pos - f.nBits
		pos += f.nBits

		retType := field.Type.Name()
		print(buf, "\nfunc (%c %s) %s() %s {", firstChar, c.TypeName, f.name, retType)
		if field.Type.Kind() == reflect.Bool {
			print(buf, "\tconst bit = 1 << %d", shift)
			print(buf, "\treturn %c&bit == bit", firstChar)
		} else {
			print(buf, "\treturn %s((%c >> %d) & %#x)", retType, firstChar, shift, (1<<f.nBits)-1)
		}
		print(buf, "}")
	}

	if c.Package != "" {
		print(w, "// Code generated by golang.org/x/text/internal/gen/bitfield. DO NOT EDIT.\n")
		print(w, "package %s\n", c.Package)
	}

	bits := posToBits(pos)

	print(w, "type %s uint%d", c.TypeName, bits)

	if _, err := io.Copy(w, buf); err != nil {
		return fmt.Errorf("bitfield: write failed: %v", err)
	}
	return nil
}
