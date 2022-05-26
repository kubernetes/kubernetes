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
	"fmt"
	"strconv"
)

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

// TrimDuplicates deduplicates elements provided in multiple key/value tuple
// slices, whilst maintaining the distinction between where the items are
// contained.
func TrimDuplicates(kvLists ...[]interface{}) [][]interface{} {
	// maintain a map of all seen keys
	seenKeys := map[interface{}]struct{}{}
	// build the same number of output slices as inputs
	outs := make([][]interface{}, len(kvLists))
	// iterate over the input slices backwards, as 'later' kv specifications
	// of the same key will take precedence over earlier ones
	for i := len(kvLists) - 1; i >= 0; i-- {
		// initialise this output slice
		outs[i] = []interface{}{}
		// obtain a reference to the kvList we are processing
		// and make sure it has an even number of entries
		kvList := kvLists[i]
		if len(kvList)%2 != 0 {
			kvList = append(kvList, missingValue)
		}

		// start iterating at len(kvList) - 2 (i.e. the 2nd last item) for
		// slices that have an even number of elements.
		// We add (len(kvList) % 2) here to handle the case where there is an
		// odd number of elements in a kvList.
		// If there is an odd number, then the last element in the slice will
		// have the value 'null'.
		for i2 := len(kvList) - 2 + (len(kvList) % 2); i2 >= 0; i2 -= 2 {
			k := kvList[i2]
			// if we have already seen this key, do not include it again
			if _, ok := seenKeys[k]; ok {
				continue
			}
			// make a note that we've observed a new key
			seenKeys[k] = struct{}{}
			// attempt to obtain the value of the key
			var v interface{}
			// i2+1 should only ever be out of bounds if we handling the first
			// iteration over a slice with an odd number of elements
			if i2+1 < len(kvList) {
				v = kvList[i2+1]
			}
			// add this KV tuple to the *start* of the output list to maintain
			// the original order as we are iterating over the slice backwards
			outs[i] = append([]interface{}{k, v}, outs[i]...)
		}
	}
	return outs
}

const missingValue = "(MISSING)"

// KVListFormat serializes all key/value pairs into the provided buffer.
// A space gets inserted before the first pair and between each pair.
func KVListFormat(b *bytes.Buffer, keysAndValues ...interface{}) {
	for i := 0; i < len(keysAndValues); i += 2 {
		var v interface{}
		k := keysAndValues[i]
		if i+1 < len(keysAndValues) {
			v = keysAndValues[i+1]
		} else {
			v = missingValue
		}
		b.WriteByte(' ')
		// Keys are assumed to be well-formed according to
		// https://github.com/kubernetes/community/blob/master/contributors/devel/sig-instrumentation/migration-to-structured-logging.md#name-arguments
		// for the sake of performance. Keys with spaces,
		// special characters, etc. will break parsing.
		if k, ok := k.(string); ok {
			// Avoid one allocation when the key is a string, which
			// normally it should be.
			b.WriteString(k)
		} else {
			b.WriteString(fmt.Sprintf("%s", k))
		}

		// The type checks are sorted so that more frequently used ones
		// come first because that is then faster in the common
		// cases. In Kubernetes, ObjectRef (a Stringer) is more common
		// than plain strings
		// (https://github.com/kubernetes/kubernetes/pull/106594#issuecomment-975526235).
		switch v := v.(type) {
		case fmt.Stringer:
			writeStringValue(b, true, StringerToString(v))
		case string:
			writeStringValue(b, true, v)
		case error:
			writeStringValue(b, true, ErrorToString(v))
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
			writeStringValue(b, false, fmt.Sprintf("%+v", v))
		}
	}
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

func writeStringValue(b *bytes.Buffer, quote bool, v string) {
	data := []byte(v)
	index := bytes.IndexByte(data, '\n')
	if index == -1 {
		b.WriteByte('=')
		if quote {
			// Simple string, quote quotation marks and non-printable characters.
			b.WriteString(strconv.Quote(v))
			return
		}
		// Non-string with no line breaks.
		b.WriteString(v)
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
