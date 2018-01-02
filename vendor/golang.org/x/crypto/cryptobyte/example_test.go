// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptobyte_test

import (
	"encoding/asn1"
	"fmt"
	"golang.org/x/crypto/cryptobyte"
)

func ExampleString_lengthPrefixed() {
	// This is an example of parsing length-prefixed data (as found in, for
	// example, TLS). Imagine a 16-bit prefixed series of 8-bit prefixed
	// strings.

	input := cryptobyte.String([]byte{0, 12, 5, 'h', 'e', 'l', 'l', 'o', 5, 'w', 'o', 'r', 'l', 'd'})
	var result []string

	var values cryptobyte.String
	if !input.ReadUint16LengthPrefixed(&values) ||
		!input.Empty() {
		panic("bad format")
	}

	for !values.Empty() {
		var value cryptobyte.String
		if !values.ReadUint8LengthPrefixed(&value) {
			panic("bad format")
		}

		result = append(result, string(value))
	}

	// Output: []string{"hello", "world"}
	fmt.Printf("%#v\n", result)
}

func ExampleString_asn1() {
	// This is an example of parsing ASN.1 data that looks like:
	//    Foo ::= SEQUENCE {
	//      version [6] INTEGER DEFAULT 0
	//      data OCTET STRING
	//    }

	input := cryptobyte.String([]byte{0x30, 12, 0xa6, 3, 2, 1, 2, 4, 5, 'h', 'e', 'l', 'l', 'o'})

	var (
		version                   int64
		data, inner, versionBytes cryptobyte.String
		haveVersion               bool
	)
	if !input.ReadASN1(&inner, cryptobyte.Tag(asn1.TagSequence).Constructed()) ||
		!input.Empty() ||
		!inner.ReadOptionalASN1(&versionBytes, &haveVersion, cryptobyte.Tag(6).Constructed().ContextSpecific()) ||
		(haveVersion && !versionBytes.ReadASN1Integer(&version)) ||
		(haveVersion && !versionBytes.Empty()) ||
		!inner.ReadASN1(&data, asn1.TagOctetString) ||
		!inner.Empty() {
		panic("bad format")
	}

	// Output: haveVersion: true, version: 2, data: hello
	fmt.Printf("haveVersion: %t, version: %d, data: %s\n", haveVersion, version, string(data))
}

func ExampleBuilder_asn1() {
	// This is an example of building ASN.1 data that looks like:
	//    Foo ::= SEQUENCE {
	//      version [6] INTEGER DEFAULT 0
	//      data OCTET STRING
	//    }

	version := int64(2)
	data := []byte("hello")
	const defaultVersion = 0

	var b cryptobyte.Builder
	b.AddASN1(cryptobyte.Tag(asn1.TagSequence).Constructed(), func(b *cryptobyte.Builder) {
		if version != defaultVersion {
			b.AddASN1(cryptobyte.Tag(6).Constructed().ContextSpecific(), func(b *cryptobyte.Builder) {
				b.AddASN1Int64(version)
			})
		}
		b.AddASN1OctetString(data)
	})

	result, err := b.Bytes()
	if err != nil {
		panic(err)
	}

	// Output: 300ca603020102040568656c6c6f
	fmt.Printf("%x\n", result)
}

func ExampleBuilder_lengthPrefixed() {
	// This is an example of building length-prefixed data (as found in,
	// for example, TLS). Imagine a 16-bit prefixed series of 8-bit
	// prefixed strings.
	input := []string{"hello", "world"}

	var b cryptobyte.Builder
	b.AddUint16LengthPrefixed(func(b *cryptobyte.Builder) {
		for _, value := range input {
			b.AddUint8LengthPrefixed(func(b *cryptobyte.Builder) {
				b.AddBytes([]byte(value))
			})
		}
	})

	result, err := b.Bytes()
	if err != nil {
		panic(err)
	}

	// Output: 000c0568656c6c6f05776f726c64
	fmt.Printf("%x\n", result)
}
