// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package util

import (
	"strconv"
)

// TODO(mdlayher): util packages are an anti-pattern and this should be moved
// somewhere else that is more focused in the future.

// A ValueParser enables parsing a single string into a variety of data types
// in a concise and safe way. The Err method must be invoked after invoking
// any other methods to ensure a value was successfully parsed.
type ValueParser struct {
	v   string
	err error
}

// NewValueParser creates a ValueParser using the input string.
func NewValueParser(v string) *ValueParser {
	return &ValueParser{v: v}
}

// PInt64 interprets the underlying value as an int64 and returns a pointer to
// that value.
func (vp *ValueParser) PInt64() *int64 {
	if vp.err != nil {
		return nil
	}

	// A base value of zero makes ParseInt infer the correct base using the
	// string's prefix, if any.
	const base = 0
	v, err := strconv.ParseInt(vp.v, base, 64)
	if err != nil {
		vp.err = err
		return nil
	}

	return &v
}

// PUInt64 interprets the underlying value as an uint64 and returns a pointer to
// that value.
func (vp *ValueParser) PUInt64() *uint64 {
	if vp.err != nil {
		return nil
	}

	// A base value of zero makes ParseInt infer the correct base using the
	// string's prefix, if any.
	const base = 0
	v, err := strconv.ParseUint(vp.v, base, 64)
	if err != nil {
		vp.err = err
		return nil
	}

	return &v
}

// Err returns the last error, if any, encountered by the ValueParser.
func (vp *ValueParser) Err() error {
	return vp.err
}
