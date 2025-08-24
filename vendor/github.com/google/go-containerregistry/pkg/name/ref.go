// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package name

import (
	"fmt"
)

// Reference defines the interface that consumers use when they can
// take either a tag or a digest.
type Reference interface {
	fmt.Stringer

	// Context accesses the Repository context of the reference.
	Context() Repository

	// Identifier accesses the type-specific portion of the reference.
	Identifier() string

	// Name is the fully-qualified reference name.
	Name() string

	// Scope is the scope needed to access this reference.
	Scope(string) string
}

// ParseReference parses the string as a reference, either by tag or digest.
func ParseReference(s string, opts ...Option) (Reference, error) {
	if t, err := NewTag(s, opts...); err == nil {
		return t, nil
	}
	if d, err := NewDigest(s, opts...); err == nil {
		return d, nil
	}
	return nil, newErrBadName("could not parse reference: %s", s)
}

type stringConst string

// MustParseReference behaves like ParseReference, but panics instead of
// returning an error. It's intended for use in tests, or when a value is
// expected to be valid at code authoring time.
//
// To discourage its use in scenarios where the value is not known at code
// authoring time, it must be passed a string constant:
//
//	const str = "valid/string"
//	MustParseReference(str)
//	MustParseReference("another/valid/string")
//	MustParseReference(str + "/and/more")
//
// These will not compile:
//
//	var str = "valid/string"
//	MustParseReference(str)
//	MustParseReference(strings.Join([]string{"valid", "string"}, "/"))
func MustParseReference(s stringConst, opts ...Option) Reference {
	ref, err := ParseReference(string(s), opts...)
	if err != nil {
		panic(err)
	}
	return ref
}
