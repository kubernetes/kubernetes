// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

// Key exchange tests.

import (
	"crypto/rand"
	"reflect"
	"testing"
)

func TestKexes(t *testing.T) {
	type kexResultErr struct {
		result *kexResult
		err    error
	}

	for name, kex := range kexAlgoMap {
		a, b := memPipe()

		s := make(chan kexResultErr, 1)
		c := make(chan kexResultErr, 1)
		var magics handshakeMagics
		go func() {
			r, e := kex.Client(a, rand.Reader, &magics)
			c <- kexResultErr{r, e}
		}()
		go func() {
			r, e := kex.Server(b, rand.Reader, &magics, testSigners["ecdsa"])
			s <- kexResultErr{r, e}
		}()

		clientRes := <-c
		serverRes := <-s
		if clientRes.err != nil {
			t.Errorf("client: %v", clientRes.err)
		}
		if serverRes.err != nil {
			t.Errorf("server: %v", serverRes.err)
		}
		if !reflect.DeepEqual(clientRes.result, serverRes.result) {
			t.Errorf("kex %q: mismatch %#v, %#v", name, clientRes.result, serverRes.result)
		}
	}
}
