/*
Copyright 2021 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/go-logr/logr"
)

type textWriter interface {
	WriteText(*bytes.Buffer)
}

// WithValues implements LogSink.WithValues. The old key/value pairs are
// assumed to be well-formed, the new ones are checked and padded if
// necessary. It returns a new slice.
func WithValues(oldKV, newKV []interface{}) []interface{} {
	if len(newKV) == 0 {
		return oldKV
	}
	newLen := len(oldKV) + len(newKV)
	hasMissingValue := newLen%2 != 0
	if hasMissingValue {
		newLen++
	}
	// The new LogSink must have its own slice.
	kv := make([]interface{}, 0, newLen)
	kv = append(kv, oldKV...)
	kv = append(kv, newKV...)
	if hasMissingValue {
		kv = append(kv, missingValue)
	}
	return kv
}

// MergeKVs deduplicates elements provided in two key/value slices.
//
// Keys in each slice are expected to be unique, so duplicates can only occur
// when the first and second slice contain the same key. When that happens, the
// key/value pair from the second slice is used. The first slice must be well-formed
// (= even key/value pairs). The second one may have a missing value, in which
// case the special "missing value" is added to the result.
func MergeKVs(first, second []interface{}) []interface{} {
	maxLength := len(first) + (len(second)+1)/2*2
	if maxLength == 0 {
		// Nothing to do at all.
		return nil
	}

	if len(first) == 0 && len(second)%2 == 0 {
		// Nothing to be overridden, second slice is well-formed
		// and can be used directly.
		return second
	}

	// Determine which keys are in the second slice so that we can skip
	// them when iterating over the first one. The code intentionally
	// favors performance over completeness: we assume that keys are string
	// constants and thus compare equal when the string values are equal. A
	// string constant being overridden by, for example, a fmt.Stringer is
	// not handled.
	overrides := map[interface{}]bool{}
	for i := 0; i < len(second); i += 2 {
		overrides[second[i]] = true
	}
	merged := make([]interface{}, 0, maxLength)
	for i := 0; i+1 < len(first); i += 2 {
		key := first[i]
		if overrides[key] {
			continue
		}
		merged = append(merged, key, first[i+1])
	}
	merged = append(merged, second...)
	if len(merged)%2 != 0 {
		merged = append(merged, missingValue)
	}
	return merged
}

type Formatter struct {
	AnyToStringHook AnyToStringFunc
}

type AnyToStringFunc func(v interface{}) string

// MergeKVsInto is a variant of MergeKVs which directly formats the key/value
// pairs into a buffer.
func (f Formatter) MergeAndFormatKVs(b *bytes.Buffer, first, second []interface{}) {
	if len(first) == 0 && len(second) == 0 {
		// Nothing to do at all.
		return
	}

	if len(first) == 0 && len(second)%2 == 0 {
		// Nothing to be overridden, second slice is well-formed
		// and can be used directly.
		for i := 0; i < len(second); i += 2 {
			f.KVFormat(b, second[i], second[i+1])
		}
		return
	}

	// Determine which keys are in the second slice so that we can skip
	// them when iterating over the first one. The code intentionally
	// favors performance over completeness: we assume that keys are string
	// constants and thus compare equal when the string values are equal. A
	// string constant being overridden by, for example, a fmt.Stringer is
	// not handled.
	overrides := map[interface{}]bool{}
	for i := 0; i < len(second); i += 2 {
		overrides[second[i]] = true
	}
	for i := 0; i < len(first); i += 2 {
		key := first[i]
		if overrides[key] {
			continue
		}
		f.KVFormat(b, key, first[i+1])
	}
	// Round down.
	l := len(second)
	l = l / 2 * 2
	for i := 1; i < l; i += 2 {
		f.KVFormat(b, second[i-1], second[i])
	}
	if len(second)%2 == 1 {
		f.KVFormat(b, second[len(second)-1], missingValue)
	}
}

func MergeAndFormatKVs(b *bytes.Buffer, first, second []interface{}) {
	Formatter{}.MergeAndFormatKVs(b, first, second)
}

const missingValue = "(MISSING)"

// KVListFormat serializes all key/value pairs into the provided buffer.
// A space gets inserted before the first pair and between each pair.
func (f Formatter) KVListFormat(b *bytes.Buffer, keysAndValues ...interface{}) {
	for i := 0; i < len(keysAndValues); i += 2 {
		var v interface{}
		k := keysAndValues[i]
		if i+1 < len(keysAndValues) {
			v = keysAndValues[i+1]
		} else {
			v = missingValue
		}
		f.KVFormat(b, k, v)
	}
}

func KVListFormat(b *bytes.Buffer, keysAndValues ...interface{}) {
	Formatter{}.KVListFormat(b, keysAndValues...)
}

func KVFormat(b *bytes.Buffer, k, v interface{}) {
	Formatter{}.KVFormat(b, k, v)
}

// formatAny is the fallback formatter for a value. It supports a hook (for
// example, for YAML encoding) and itself uses JSON encoding.
func (f Formatter) formatAny(b *bytes.Buffer, v interface{}) {
	b.WriteRune('=')
	if f.AnyToStringHook != nil {
		b.WriteString(f.AnyToStringHook(v))
		return
	}
	formatAsJSON(b, v)
}

func formatAsJSON(b *bytes.Buffer, v interface{}) {
	encoder := json.NewEncoder(b)
	l := b.Len()
	if err := encoder.Encode(v); err != nil {
		// This shouldn't happen. We discard whatever the encoder
		// wrote and instead dump an error string.
		b.Truncate(l)
		b.WriteString(fmt.Sprintf(`"<internal error: %v>"`, err))
		return
	}
	// Remove trailing newline.
	b.Truncate(b.Len() - 1)
}

// StringerToString converts a Stringer to a string,
// handling panics if they occur.
func StringerToString(s fmt.Stringer) (ret string) {
	defer func() {
		if err := recover(); err != nil {
			ret = fmt.Sprintf("<panic: %s>", err)
		}
	}()
	ret = s.String()
	return
}

// MarshalerToValue invokes a marshaler and catches
// panics.
func MarshalerToValue(m logr.Marshaler) (ret interface{}) {
	defer func() {
		if err := recover(); err != nil {
			ret = fmt.Sprintf("<panic: %s>", err)
		}
	}()
	ret = m.MarshalLog()
	return
}

// ErrorToString converts an error to a string,
// handling panics if they occur.
func ErrorToString(err error) (ret string) {
	defer func() {
		if err := recover(); err != nil {
			ret = fmt.Sprintf("<panic: %s>", err)
		}
	}()
	ret = err.Error()
	return
}

func writeTextWriterValue(b *bytes.Buffer, v textWriter) {
	b.WriteByte('=')
	defer func() {
		if err := recover(); err != nil {
			fmt.Fprintf(b, `"<panic: %s>"`, err)
		}
	}()
	v.WriteText(b)
}

func writeStringValue(b *bytes.Buffer, v string) {
	data := []byte(v)
	index := bytes.IndexByte(data, '\n')
	if index == -1 {
		b.WriteByte('=')
		// Simple string, quote quotation marks and non-printable characters.
		b.WriteString(strconv.Quote(v))
		return
	}

	// Complex multi-line string, show as-is with indention like this:
	// I... "hello world" key=<
	// <tab>line 1
	// <tab>line 2
	//  >
	//
	// Tabs indent the lines of the value while the end of string delimiter
	// is indented with a space. That has two purposes:
	// - visual difference between the two for a human reader because indention
	//   will be different
	// - no ambiguity when some value line starts with the end delimiter
	//
	// One downside is that the output cannot distinguish between strings that
	// end with a line break and those that don't because the end delimiter
	// will always be on the next line.
	b.WriteString("=<\n")
	for index != -1 {
		b.WriteByte('\t')
		b.Write(data[0 : index+1])
		data = data[index+1:]
		index = bytes.IndexByte(data, '\n')
	}
	if len(data) == 0 {
		// String ended with line break, don't add another.
		b.WriteString(" >")
	} else {
		// No line break at end of last line, write rest of string and
		// add one.
		b.WriteByte('\t')
		b.Write(data)
		b.WriteString("\n >")
	}
}
