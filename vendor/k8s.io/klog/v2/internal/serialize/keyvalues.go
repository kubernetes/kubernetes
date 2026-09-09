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
	"slices"
	"strconv"
	"strings"

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

type Formatter struct {
	AnyToStringHook AnyToStringFunc
}

type AnyToStringFunc func(v interface{}) string

const missingValue = "(MISSING)"

func FormatKVs(b *bytes.Buffer, kvs ...[]interface{}) {
	Formatter{}.FormatKVs(b, kvs...)
}

// FormatKVs formats all key/value pairs such that the output contains no
// duplicates ("last one wins").
func (f Formatter) FormatKVs(b *bytes.Buffer, kvs ...[]interface{}) {
	// De-duplication is done by optimistically formatting all key value
	// pairs and then cutting out the output of those key/value pairs which
	// got overwritten later.
	//
	// In the common case of no duplicates, the only overhead is tracking
	// previous keys. This uses a slice with a simple linear search because
	// the number of entries is typically so low that allocating a map or
	// keeping a sorted slice with binary search aren't justified.
	//
	// Using a fixed size here makes the Go compiler use the stack as
	// initial backing store for the slice, which is crucial for
	// performance.
	existing := make([]obsoleteKV, 0, 32)
	obsolete := make([]interval, 0, 32) // Sorted by start index.
	for _, keysAndValues := range kvs {
		for i := 0; i < len(keysAndValues); i += 2 {
			var v interface{}
			k := keysAndValues[i]
			if i+1 < len(keysAndValues) {
				v = keysAndValues[i+1]
			} else {
				v = missingValue
			}
			var e obsoleteKV
			e.start = b.Len()
			e.key = f.KVFormat(b, k, v)
			e.end = b.Len()
			i := findObsoleteEntry(existing, e.key)
			if i >= 0 {
				data := b.Bytes()
				if bytes.Compare(data[existing[i].start:existing[i].end], data[e.start:e.end]) == 0 {
					// The new entry gets obsoleted because it's identical.
					// This has the advantage that key/value pairs from
					// a WithValues call always come first, even if the same
					// pair gets added again later. This makes different log
					// entries more consistent.
					//
					// The new entry has a higher start index and thus can be appended.
					obsolete = append(obsolete, e.interval)
				} else {
					// The old entry gets obsoleted because it's value is different.
					//
					// Sort order is not guaranteed, we have to insert at the right place.
					index, _ := slices.BinarySearchFunc(obsolete, existing[i].interval, func(a, b interval) int { return a.start - b.start })
					obsolete = slices.Insert(obsolete, index, existing[i].interval)
					existing[i].interval = e.interval
				}
			} else {
				// Instead of appending at the end and doing a
				// linear search in findEntry, we could keep
				// the slice sorted by key and do a binary search.
				//
				// Above:
				//    i, ok := slices.BinarySearchFunc(existing, e, func(a, b entry) int { return strings.Compare(a.key, b.key) })
				// Here:
				//    existing = slices.Insert(existing, i, e)
				//
				// But that adds a dependency on the slices package
				// and made performance slightly worse, presumably
				// because the cost of shifting entries around
				// did not pay of with faster lookups.
				existing = append(existing, e)
			}
		}
	}

	// If we need to remove some obsolete key/value pairs then move the memory.
	if len(obsolete) > 0 {
		// Potentially the next remaining output (might itself be obsolete).
		from := obsolete[0].end
		// Next obsolete entry.
		nextObsolete := 1
		// This is the source buffer, before truncation.
		all := b.Bytes()
		b.Truncate(obsolete[0].start)

		for nextObsolete < len(obsolete) {
			if from == obsolete[nextObsolete].start {
				// Skip also the next obsolete key/value.
				from = obsolete[nextObsolete].end
				nextObsolete++
				continue
			}

			// Preserve some output. Write uses copy, which
			// explicitly allows source and destination to overlap.
			// That could happen here.
			valid := all[from:obsolete[nextObsolete].start]
			b.Write(valid)
			from = obsolete[nextObsolete].end
			nextObsolete++
		}
		// Copy end of buffer.
		valid := all[from:]
		b.Write(valid)
	}
}

type obsoleteKV struct {
	key string
	interval
}

// interval includes the start and excludes the end.
type interval struct {
	start int
	end   int
}

func findObsoleteEntry(entries []obsoleteKV, key string) int {
	for i, entry := range entries {
		if entry.key == key {
			return i
		}
	}
	return -1
}

// formatAny is the fallback formatter for a value. It supports a hook (for
// example, for YAML encoding) and itself uses JSON encoding.
func (f Formatter) formatAny(b *bytes.Buffer, v interface{}) {
	if f.AnyToStringHook != nil {
		str := f.AnyToStringHook(v)
		if strings.Contains(str, "\n") {
			// If it's multi-line, then pass it through writeStringValue to get start/end delimiters,
			// which separates it better from any following key/value pair.
			writeStringValue(b, str)
			return
		}
		// Otherwise put it directly after the separator, on the same lime,
		// The assumption is that the hook returns something where start/end are obvious.
		b.WriteRune('=')
		b.WriteString(str)
		return
	}
	b.WriteRune('=')
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
