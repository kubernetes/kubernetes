/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package metadata

import (
	"reflect"
	"testing"
)

const binaryValue = string(128)

func TestEncodeKeyValue(t *testing.T) {
	for _, test := range []struct {
		// input
		kin string
		vin string
		// output
		kout string
		vout string
	}{
		{"key", "abc", "key", "abc"},
		{"KEY", "abc", "key", "abc"},
		{"key-bin", "abc", "key-bin", "YWJj"},
		{"key-bin", binaryValue, "key-bin", "woA="},
	} {
		k, v := encodeKeyValue(test.kin, test.vin)
		if k != test.kout || !reflect.DeepEqual(v, test.vout) {
			t.Fatalf("encodeKeyValue(%q, %q) = %q, %q, want %q, %q", test.kin, test.vin, k, v, test.kout, test.vout)
		}
	}
}

func TestDecodeKeyValue(t *testing.T) {
	for _, test := range []struct {
		// input
		kin string
		vin string
		// output
		kout string
		vout string
		err  error
	}{
		{"a", "abc", "a", "abc", nil},
		{"key-bin", "Zm9vAGJhcg==", "key-bin", "foo\x00bar", nil},
		{"key-bin", "woA=", "key-bin", binaryValue, nil},
	} {
		k, v, err := DecodeKeyValue(test.kin, test.vin)
		if k != test.kout || !reflect.DeepEqual(v, test.vout) || !reflect.DeepEqual(err, test.err) {
			t.Fatalf("DecodeKeyValue(%q, %q) = %q, %q, %v, want %q, %q, %v", test.kin, test.vin, k, v, err, test.kout, test.vout, test.err)
		}
	}
}

func TestPairsMD(t *testing.T) {
	for _, test := range []struct {
		// input
		kv []string
		// output
		md   MD
		size int
	}{
		{[]string{}, MD{}, 0},
		{[]string{"k1", "v1", "k2-bin", binaryValue}, New(map[string]string{
			"k1":     "v1",
			"k2-bin": binaryValue,
		}), 2},
	} {
		md := Pairs(test.kv...)
		if !reflect.DeepEqual(md, test.md) {
			t.Fatalf("Pairs(%v) = %v, want %v", test.kv, md, test.md)
		}
		if md.Len() != test.size {
			t.Fatalf("Pairs(%v) generates md of size %d, want %d", test.kv, md.Len(), test.size)
		}
	}
}
