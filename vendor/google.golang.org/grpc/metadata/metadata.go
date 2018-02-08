/*
 *
 * Copyright 2014 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package metadata define the structure of the metadata supported by gRPC library.
// Please refer to https://grpc.io/docs/guides/wire.html for more information about custom-metadata.
package metadata

import (
	"fmt"
	"strings"

	"golang.org/x/net/context"
)

// DecodeKeyValue returns k, v, nil.  It is deprecated and should not be used.
func DecodeKeyValue(k, v string) (string, string, error) {
	return k, v, nil
}

// MD is a mapping from metadata keys to values. Users should use the following
// two convenience functions New and Pairs to generate MD.
type MD map[string][]string

// New creates an MD from a given key-value map.
//
// Only the following ASCII characters are allowed in keys:
//  - digits: 0-9
//  - uppercase letters: A-Z (normalized to lower)
//  - lowercase letters: a-z
//  - special characters: -_.
// Uppercase letters are automatically converted to lowercase.
//
// Keys beginning with "grpc-" are reserved for grpc-internal use only and may
// result in errors if set in metadata.
func New(m map[string]string) MD {
	md := MD{}
	for k, val := range m {
		key := strings.ToLower(k)
		md[key] = append(md[key], val)
	}
	return md
}

// Pairs returns an MD formed by the mapping of key, value ...
// Pairs panics if len(kv) is odd.
//
// Only the following ASCII characters are allowed in keys:
//  - digits: 0-9
//  - uppercase letters: A-Z (normalized to lower)
//  - lowercase letters: a-z
//  - special characters: -_.
// Uppercase letters are automatically converted to lowercase.
//
// Keys beginning with "grpc-" are reserved for grpc-internal use only and may
// result in errors if set in metadata.
func Pairs(kv ...string) MD {
	if len(kv)%2 == 1 {
		panic(fmt.Sprintf("metadata: Pairs got the odd number of input pairs for metadata: %d", len(kv)))
	}
	md := MD{}
	var key string
	for i, s := range kv {
		if i%2 == 0 {
			key = strings.ToLower(s)
			continue
		}
		md[key] = append(md[key], s)
	}
	return md
}

// Len returns the number of items in md.
func (md MD) Len() int {
	return len(md)
}

// Copy returns a copy of md.
func (md MD) Copy() MD {
	return Join(md)
}

// Join joins any number of mds into a single MD.
// The order of values for each key is determined by the order in which
// the mds containing those values are presented to Join.
func Join(mds ...MD) MD {
	out := MD{}
	for _, md := range mds {
		for k, v := range md {
			out[k] = append(out[k], v...)
		}
	}
	return out
}

type mdIncomingKey struct{}
type mdOutgoingKey struct{}

// NewIncomingContext creates a new context with incoming md attached.
func NewIncomingContext(ctx context.Context, md MD) context.Context {
	return context.WithValue(ctx, mdIncomingKey{}, md)
}

// NewOutgoingContext creates a new context with outgoing md attached.
func NewOutgoingContext(ctx context.Context, md MD) context.Context {
	return context.WithValue(ctx, mdOutgoingKey{}, md)
}

// FromIncomingContext returns the incoming metadata in ctx if it exists.  The
// returned MD should not be modified. Writing to it may cause races.
// Modification should be made to copies of the returned MD.
func FromIncomingContext(ctx context.Context) (md MD, ok bool) {
	md, ok = ctx.Value(mdIncomingKey{}).(MD)
	return
}

// FromOutgoingContext returns the outgoing metadata in ctx if it exists.  The
// returned MD should not be modified. Writing to it may cause races.
// Modification should be made to the copies of the returned MD.
func FromOutgoingContext(ctx context.Context) (md MD, ok bool) {
	md, ok = ctx.Value(mdOutgoingKey{}).(MD)
	return
}
