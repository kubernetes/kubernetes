// Copyright 2016 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

import (
	"fmt"
	"log"
)

func ExampleLoad_iso88591() {
	buf := []byte("key = ISO-8859-1 value with unicode literal \\u2318 and umlaut \xE4") // 0xE4 == ä
	p, _ := Load(buf, ISO_8859_1)
	v, ok := p.Get("key")
	fmt.Println(ok)
	fmt.Println(v)
	// Output:
	// true
	// ISO-8859-1 value with unicode literal ⌘ and umlaut ä
}

func ExampleLoad_utf8() {
	p, _ := Load([]byte("key = UTF-8 value with unicode character ⌘ and umlaut ä"), UTF8)
	v, ok := p.Get("key")
	fmt.Println(ok)
	fmt.Println(v)
	// Output:
	// true
	// UTF-8 value with unicode character ⌘ and umlaut ä
}

func ExampleProperties_GetBool() {
	var input = `
	key=1
	key2=On
	key3=YES
	key4=true`
	p, _ := Load([]byte(input), ISO_8859_1)
	fmt.Println(p.GetBool("key", false))
	fmt.Println(p.GetBool("key2", false))
	fmt.Println(p.GetBool("key3", false))
	fmt.Println(p.GetBool("key4", false))
	fmt.Println(p.GetBool("keyX", false))
	// Output:
	// true
	// true
	// true
	// true
	// false
}

func ExampleProperties_GetString() {
	p, _ := Load([]byte("key=value"), ISO_8859_1)
	v := p.GetString("another key", "default value")
	fmt.Println(v)
	// Output:
	// default value
}

func Example() {
	// Decode some key/value pairs with expressions
	p, err := Load([]byte("key=value\nkey2=${key}"), ISO_8859_1)
	if err != nil {
		log.Fatal(err)
	}

	// Get a valid key
	if v, ok := p.Get("key"); ok {
		fmt.Println(v)
	}

	// Get an invalid key
	if _, ok := p.Get("does not exist"); !ok {
		fmt.Println("invalid key")
	}

	// Get a key with a default value
	v := p.GetString("does not exist", "some value")
	fmt.Println(v)

	// Dump the expanded key/value pairs of the Properties
	fmt.Println("Expanded key/value pairs")
	fmt.Println(p)

	// Output:
	// value
	// invalid key
	// some value
	// Expanded key/value pairs
	// key = value
	// key2 = value
}
