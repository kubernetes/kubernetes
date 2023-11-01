//go:build !go1.21
// +build !go1.21

/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package serialize

import (
	"bytes"
	"fmt"

	"github.com/go-logr/logr"
)

// KVFormat serializes one key/value pair into the provided buffer.
// A space gets inserted before the pair.
func (f Formatter) KVFormat(b *bytes.Buffer, k, v interface{}) {
	// This is the version without slog support. Must be kept in sync with
	// the version in keyvalues_slog.go.

	b.WriteByte(' ')
	// Keys are assumed to be well-formed according to
	// https://github.com/kubernetes/community/blob/master/contributors/devel/sig-instrumentation/migration-to-structured-logging.md#name-arguments
	// for the sake of performance. Keys with spaces,
	// special characters, etc. will break parsing.
	if sK, ok := k.(string); ok {
		// Avoid one allocation when the key is a string, which
		// normally it should be.
		b.WriteString(sK)
	} else {
		b.WriteString(fmt.Sprintf("%s", k))
	}

	// The type checks are sorted so that more frequently used ones
	// come first because that is then faster in the common
	// cases. In Kubernetes, ObjectRef (a Stringer) is more common
	// than plain strings
	// (https://github.com/kubernetes/kubernetes/pull/106594#issuecomment-975526235).
	switch v := v.(type) {
	case textWriter:
		writeTextWriterValue(b, v)
	case fmt.Stringer:
		writeStringValue(b, StringerToString(v))
	case string:
		writeStringValue(b, v)
	case error:
		writeStringValue(b, ErrorToString(v))
	case logr.Marshaler:
		value := MarshalerToValue(v)
		// A marshaler that returns a string is useful for
		// delayed formatting of complex values. We treat this
		// case like a normal string. This is useful for
		// multi-line support.
		//
		// We could do this by recursively formatting a value,
		// but that comes with the risk of infinite recursion
		// if a marshaler returns itself. Instead we call it
		// only once and rely on it returning the intended
		// value directly.
		switch value := value.(type) {
		case string:
			writeStringValue(b, value)
		default:
			f.formatAny(b, value)
		}
	case []byte:
		// In https://github.com/kubernetes/klog/pull/237 it was decided
		// to format byte slices with "%+q". The advantages of that are:
		// - readable output if the bytes happen to be printable
		// - non-printable bytes get represented as unicode escape
		//   sequences (\uxxxx)
		//
		// The downsides are that we cannot use the faster
		// strconv.Quote here and that multi-line output is not
		// supported. If developers know that a byte array is
		// printable and they want multi-line output, they can
		// convert the value to string before logging it.
		b.WriteByte('=')
		b.WriteString(fmt.Sprintf("%+q", v))
	default:
		f.formatAny(b, v)
	}
}
