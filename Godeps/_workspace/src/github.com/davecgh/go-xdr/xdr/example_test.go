/*
 * Copyright (c) 2012 Dave Collins <dave@davec.name>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

package xdr_test

import (
	"fmt"
	"github.com/davecgh/go-xdr/xdr"
)

// This example demonstrates how to use Marshal to automatically XDR encode
// data using reflection.
func ExampleMarshal() {
	// Hypothetical image header format.
	type ImageHeader struct {
		Signature   [3]byte
		Version     uint32
		IsGrayscale bool
		NumSections uint32
	}

	// Sample image header data.
	h := ImageHeader{[3]byte{0xAB, 0xCD, 0xEF}, 2, true, 10}

	// Use marshal to automatically determine the appropriate underlying XDR
	// types and encode.
	encodedData, err := xdr.Marshal(&h)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("encodedData:", encodedData)

	// Output:
	// encodedData: [171 205 239 0 0 0 0 2 0 0 0 1 0 0 0 10]
}

// This example demonstrates how to use Unmarshal to decode XDR encoded data from
// a byte slice into a struct.
func ExampleUnmarshal() {
	// Hypothetical image header format.
	type ImageHeader struct {
		Signature   [3]byte
		Version     uint32
		IsGrayscale bool
		NumSections uint32
	}

	// XDR encoded data described by the above structure.  Typically this would
	// be read from a file or across the network, but use a manual byte array
	// here as an example.
	encodedData := []byte{
		0xAB, 0xCD, 0xEF, 0x00, // Signature
		0x00, 0x00, 0x00, 0x02, // Version
		0x00, 0x00, 0x00, 0x01, // IsGrayscale
		0x00, 0x00, 0x00, 0x0A} // NumSections

	// Declare a variable to provide Unmarshal with a concrete type and instance
	// to decode into.
	var h ImageHeader
	remainingBytes, err := xdr.Unmarshal(encodedData, &h)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("remainingBytes:", remainingBytes)
	fmt.Printf("h: %+v", h)

	// Output:
	// remainingBytes: []
	// h: {Signature:[171 205 239] Version:2 IsGrayscale:true NumSections:10}
}

// This example demonstrates how to manually decode XDR encoded data from a
// byte slice. Compare this example with the Unmarshal example which performs
// the same task automatically by utilizing a struct type definition and
// reflection.
func ExampleNewDecoder() {

	// XDR encoded data for a hypothetical ImageHeader struct as follows:
	// type ImageHeader struct {
	// 		Signature	[3]byte
	// 		Version		uint32
	// 		IsGrayscale	bool
	// 		NumSections	uint32
	// }
	encodedData := []byte{
		0xAB, 0xCD, 0xEF, 0x00, // Signature
		0x00, 0x00, 0x00, 0x02, // Version
		0x00, 0x00, 0x00, 0x01, // IsGrayscale
		0x00, 0x00, 0x00, 0x0A} // NumSections

	// Get a new decoder for manual decoding.
	dec := xdr.NewDecoder(encodedData)

	signature, err := dec.DecodeFixedOpaque(3)
	if err != nil {
		fmt.Println(err)
		return
	}

	version, err := dec.DecodeUint()
	if err != nil {
		fmt.Println(err)
		return
	}

	isGrayscale, err := dec.DecodeBool()
	if err != nil {
		fmt.Println(err)
		return
	}

	numSections, err := dec.DecodeUint()
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("signature:", signature)
	fmt.Println("version:", version)
	fmt.Println("isGrayscale:", isGrayscale)
	fmt.Println("numSections:", numSections)

	// Output:
	// signature: [171 205 239]
	// version: 2
	// isGrayscale: true
	// numSections: 10
}

// This example demonstrates how to manually encode XDR data from Go variables.
// Compare this example with the Marshal example which performs the same task
// automatically by utilizing a struct type definition and reflection.
func ExampleNewEncoder() {
	// Data for a hypothetical ImageHeader struct as follows:
	// type ImageHeader struct {
	// 		Signature	[3]byte
	//		Version		uint32
	//		IsGrayscale	bool
	//		NumSections	uint32
	// }
	signature := []byte{0xAB, 0xCD, 0xEF}
	version := uint32(2)
	isGrayscale := true
	numSections := uint32(10)

	// Get a new encoder for manual encoding.
	enc := xdr.NewEncoder()

	err := enc.EncodeFixedOpaque(signature)
	if err != nil {
		fmt.Println(err)
		return
	}

	err = enc.EncodeUint(version)
	if err != nil {
		fmt.Println(err)
		return
	}

	err = enc.EncodeBool(isGrayscale)
	if err != nil {
		fmt.Println(err)
		return
	}

	err = enc.EncodeUint(numSections)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("encodedData:", enc.Data())

	// Output:
	// encodedData: [171 205 239 0 0 0 0 2 0 0 0 1 0 0 0 10]
}
