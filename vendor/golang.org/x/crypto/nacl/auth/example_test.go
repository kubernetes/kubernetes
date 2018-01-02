// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package auth_test

import (
	"encoding/hex"
	"fmt"

	"golang.org/x/crypto/nacl/auth"
)

func Example() {
	// Load your secret key from a safe place and reuse it across multiple
	// Sum calls. (Obviously don't use this example key for anything
	// real.) If you want to convert a passphrase to a key, use a suitable
	// package like bcrypt or scrypt.
	secretKeyBytes, err := hex.DecodeString("6368616e676520746869732070617373776f726420746f206120736563726574")
	if err != nil {
		panic(err)
	}

	var secretKey [32]byte
	copy(secretKey[:], secretKeyBytes)

	mac := auth.Sum([]byte("hello world"), &secretKey)
	fmt.Printf("%x\n", *mac)
	result := auth.Verify(mac[:], []byte("hello world"), &secretKey)
	fmt.Println(result)
	badResult := auth.Verify(mac[:], []byte("different message"), &secretKey)
	fmt.Println(badResult)
	// Output: eca5a521f3d77b63f567fb0cb6f5f2d200641bc8dada42f60c5f881260c30317
	// true
	// false
}
