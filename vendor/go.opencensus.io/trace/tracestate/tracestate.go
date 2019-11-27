// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package tracestate implements support for the Tracestate header of the
// W3C TraceContext propagation format.
package tracestate

import (
	"fmt"
	"regexp"
)

const (
	keyMaxSize       = 256
	valueMaxSize     = 256
	maxKeyValuePairs = 32
)

const (
	keyWithoutVendorFormat = `[a-z][_0-9a-z\-\*\/]{0,255}`
	keyWithVendorFormat    = `[a-z][_0-9a-z\-\*\/]{0,240}@[a-z][_0-9a-z\-\*\/]{0,13}`
	keyFormat              = `(` + keyWithoutVendorFormat + `)|(` + keyWithVendorFormat + `)`
	valueFormat            = `[\x20-\x2b\x2d-\x3c\x3e-\x7e]{0,255}[\x21-\x2b\x2d-\x3c\x3e-\x7e]`
)

var keyValidationRegExp = regexp.MustCompile(`^(` + keyFormat + `)$`)
var valueValidationRegExp = regexp.MustCompile(`^(` + valueFormat + `)$`)

// Tracestate represents tracing-system specific context in a list of key-value pairs. Tracestate allows different
// vendors propagate additional information and inter-operate with their legacy Id formats.
type Tracestate struct {
	entries []Entry
}

// Entry represents one key-value pair in a list of key-value pair of Tracestate.
type Entry struct {
	// Key is an opaque string up to 256 characters printable. It MUST begin with a lowercase letter,
	// and can only contain lowercase letters a-z, digits 0-9, underscores _, dashes -, asterisks *, and
	// forward slashes /.
	Key string

	// Value is an opaque string up to 256 characters printable ASCII RFC0020 characters (i.e., the
	// range 0x20 to 0x7E) except comma , and =.
	Value string
}

// Entries returns a slice of Entry.
func (ts *Tracestate) Entries() []Entry {
	if ts == nil {
		return nil
	}
	return ts.entries
}

func (ts *Tracestate) remove(key string) *Entry {
	for index, entry := range ts.entries {
		if entry.Key == key {
			ts.entries = append(ts.entries[:index], ts.entries[index+1:]...)
			return &entry
		}
	}
	return nil
}

func (ts *Tracestate) add(entries []Entry) error {
	for _, entry := range entries {
		ts.remove(entry.Key)
	}
	if len(ts.entries)+len(entries) > maxKeyValuePairs {
		return fmt.Errorf("adding %d key-value pairs to current %d pairs exceeds the limit of %d",
			len(entries), len(ts.entries), maxKeyValuePairs)
	}
	ts.entries = append(entries, ts.entries...)
	return nil
}

func isValid(entry Entry) bool {
	return keyValidationRegExp.MatchString(entry.Key) &&
		valueValidationRegExp.MatchString(entry.Value)
}

func containsDuplicateKey(entries ...Entry) (string, bool) {
	keyMap := make(map[string]int)
	for _, entry := range entries {
		if _, ok := keyMap[entry.Key]; ok {
			return entry.Key, true
		}
		keyMap[entry.Key] = 1
	}
	return "", false
}

func areEntriesValid(entries ...Entry) (*Entry, bool) {
	for _, entry := range entries {
		if !isValid(entry) {
			return &entry, false
		}
	}
	return nil, true
}

// New creates a Tracestate object from a parent and/or entries (key-value pair).
// Entries from the parent are copied if present. The entries passed to this function
// are inserted in front of those copied from the parent. If an entry copied from the
// parent contains the same key as one of the entry in entries then the entry copied
// from the parent is removed. See add func.
//
// An error is returned with nil Tracestate if
//  1. one or more entry in entries is invalid.
//  2. two or more entries in the input entries have the same key.
//  3. the number of entries combined from the parent and the input entries exceeds maxKeyValuePairs.
//     (duplicate entry is counted only once).
func New(parent *Tracestate, entries ...Entry) (*Tracestate, error) {
	if parent == nil && len(entries) == 0 {
		return nil, nil
	}
	if entry, ok := areEntriesValid(entries...); !ok {
		return nil, fmt.Errorf("key-value pair {%s, %s} is invalid", entry.Key, entry.Value)
	}

	if key, duplicate := containsDuplicateKey(entries...); duplicate {
		return nil, fmt.Errorf("contains duplicate keys (%s)", key)
	}

	tracestate := Tracestate{}

	if parent != nil && len(parent.entries) > 0 {
		tracestate.entries = append([]Entry{}, parent.entries...)
	}

	err := tracestate.add(entries)
	if err != nil {
		return nil, err
	}
	return &tracestate, nil
}
