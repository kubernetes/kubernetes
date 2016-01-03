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

// Package metadata define the structure of the metadata supported by gRPC library.
package metadata

import (
	"encoding/base64"
	"fmt"
	"strings"

	"golang.org/x/net/context"
)

const (
	binHdrSuffix = "-bin"
)

// grpc-http2 requires ASCII header key and value (more detail can be found in
// "Requests" subsection in go/grpc-http2).
func isASCII(s string) bool {
	for _, c := range s {
		if c > 127 {
			return false
		}
	}
	return true
}

// encodeKeyValue encodes key and value qualified for transmission via gRPC.
// Transmitting binary headers violates HTTP/2 spec.
// TODO(zhaoq): Maybe check if k is ASCII also.
func encodeKeyValue(k, v string) (string, string) {
	if isASCII(v) {
		return k, v
	}
	key := strings.ToLower(k + binHdrSuffix)
	val := base64.StdEncoding.EncodeToString([]byte(v))
	return key, string(val)
}

// DecodeKeyValue returns the original key and value corresponding to the
// encoded data in k, v.
func DecodeKeyValue(k, v string) (string, string, error) {
	if !strings.HasSuffix(k, binHdrSuffix) {
		return k, v, nil
	}
	key := k[:len(k)-len(binHdrSuffix)]
	val, err := base64.StdEncoding.DecodeString(v)
	if err != nil {
		return "", "", err
	}
	return key, string(val), nil
}

// MD is a mapping from metadata keys to values. Users should use the following
// two convenience functions New and Pairs to generate MD.
type MD map[string][]string

// New creates a MD from given key-value map.
func New(m map[string]string) MD {
	md := MD{}
	for k, v := range m {
		key, val := encodeKeyValue(k, v)
		md[key] = append(md[key], val)
	}
	return md
}

// Pairs returns an MD formed by the mapping of key, value ...
// Pairs panics if len(kv) is odd.
func Pairs(kv ...string) MD {
	if len(kv)%2 == 1 {
		panic(fmt.Sprintf("metadata: Pairs got the odd number of input pairs for metadata: %d", len(kv)))
	}
	md := MD{}
	var k string
	for i, s := range kv {
		if i%2 == 0 {
			k = s
			continue
		}
		key, val := encodeKeyValue(k, s)
		md[key] = append(md[key], val)
	}
	return md
}

// Len returns the number of items in md.
func (md MD) Len() int {
	return len(md)
}

// Copy returns a copy of md.
func (md MD) Copy() MD {
	out := MD{}
	for k, v := range md {
		for _, i := range v {
			out[k] = append(out[k], i)
		}
	}
	return out
}

type mdKey struct{}

// NewContext creates a new context with md attached.
func NewContext(ctx context.Context, md MD) context.Context {
	return context.WithValue(ctx, mdKey{}, md)
}

// FromContext returns the MD in ctx if it exists.
func FromContext(ctx context.Context) (md MD, ok bool) {
	md, ok = ctx.Value(mdKey{}).(MD)
	return
}
