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

package tracestate

import (
	"fmt"
	"testing"
)

func checkFront(t *testing.T, tracestate *Tracestate, wantKey, testname string) {
	gotKey := tracestate.entries[0].Key
	if gotKey != wantKey {
		t.Errorf("test:%s: first entry in the list: got %q want %q", testname, gotKey, wantKey)
	}
}

func checkBack(t *testing.T, tracestate *Tracestate, wantKey, testname string) {
	gotKey := tracestate.entries[len(tracestate.entries)-1].Key
	if gotKey != wantKey {
		t.Errorf("test:%s: last entry in the list: got %q want %q", testname, gotKey, wantKey)
	}
}

func checkSize(t *testing.T, tracestate *Tracestate, wantSize int, testname string) {
	if gotSize := len(tracestate.entries); gotSize != wantSize {
		t.Errorf("test:%s: size of the list: got %q want %q", testname, gotSize, wantSize)
	}
}

func (ts *Tracestate) get(key string) (string, bool) {
	if ts == nil {
		return "", false
	}
	for _, entry := range ts.entries {
		if entry.Key == key {
			return entry.Value, true
		}
	}
	return "", false
}

func checkKeyValue(t *testing.T, tracestate *Tracestate, key, wantValue, testname string) {
	wantOk := true
	if wantValue == "" {
		wantOk = false
	}
	gotValue, gotOk := tracestate.get(key)
	if wantOk != gotOk || gotValue != wantValue {
		t.Errorf("test:%s: get value for key=%s failed: got %q want %q", testname, key, gotValue, wantValue)
	}
}

func checkError(t *testing.T, tracestate *Tracestate, err error, testname, msg string) {
	if err != nil {
		t.Errorf("test:%s: %s: tracestate=%v, error= %v", testname, msg, tracestate, err)
	}
}

func wantError(t *testing.T, tracestate *Tracestate, err error, testname, msg string) {
	if err == nil {
		t.Errorf("test:%s: %s: tracestate=%v, error=%v", testname, msg, tracestate, err)
	}
}

func TestCreateWithNullParent(t *testing.T) {
	key1, value1 := "hello", "world"
	testname := "TestCreateWithNullParent"

	entry := Entry{key1, value1}
	tracestate, err := New(nil, entry)
	checkError(t, tracestate, err, testname, "create failed from null parent")
	checkKeyValue(t, tracestate, key1, value1, testname)
}

func TestCreateFromParentWithSingleKey(t *testing.T) {
	key1, value1, key2, value2 := "hello", "world", "foo", "bar"
	testname := "TestCreateFromParentWithSingleKey"

	entry1 := Entry{key1, value1}
	entry2 := Entry{key2, value2}
	parent, _ := New(nil, entry1)
	tracestate, err := New(parent, entry2)

	checkError(t, tracestate, err, testname, "create failed from parent with single key")
	checkKeyValue(t, tracestate, key2, value2, testname)
	checkFront(t, tracestate, key2, testname)
	checkBack(t, tracestate, key1, testname)
}

func TestCreateFromParentWithDoubleKeys(t *testing.T) {
	key1, value1, key2, value2, key3, value3 := "hello", "world", "foo", "bar", "bar", "baz"
	testname := "TestCreateFromParentWithDoubleKeys"

	entry1 := Entry{key1, value1}
	entry2 := Entry{key2, value2}
	entry3 := Entry{key3, value3}
	parent, _ := New(nil, entry2, entry1)
	tracestate, err := New(parent, entry3)

	checkError(t, tracestate, err, testname, "create failed from parent with double keys")
	checkKeyValue(t, tracestate, key3, value3, testname)
	checkFront(t, tracestate, key3, testname)
	checkBack(t, tracestate, key1, testname)
}

func TestCreateFromParentWithExistingKey(t *testing.T) {
	key1, value1, key2, value2, key3, value3 := "hello", "world", "foo", "bar", "hello", "baz"
	testname := "TestCreateFromParentWithExistingKey"

	entry1 := Entry{key1, value1}
	entry2 := Entry{key2, value2}
	entry3 := Entry{key3, value3}
	parent, _ := New(nil, entry2, entry1)
	tracestate, err := New(parent, entry3)

	checkError(t, tracestate, err, testname, "create failed with an existing key")
	checkKeyValue(t, tracestate, key3, value3, testname)
	checkFront(t, tracestate, key3, testname)
	checkBack(t, tracestate, key2, testname)
	checkSize(t, tracestate, 2, testname)
}

func TestImplicitImmutableTracestate(t *testing.T) {
	key1, value1, key2, value2, key3, value3 := "hello", "world", "hello", "bar", "foo", "baz"
	testname := "TestImplicitImmutableTracestate"

	entry1 := Entry{key1, value1}
	entry2 := Entry{key2, value2}
	parent, _ := New(nil, entry1)
	tracestate, err := New(parent, entry2)

	checkError(t, tracestate, err, testname, "create failed")
	checkKeyValue(t, tracestate, key2, value2, testname)
	checkKeyValue(t, parent, key2, value1, testname)

	// Get and update entries.
	entries := tracestate.Entries()
	entry := Entry{key3, value3}
	entries = append(entries, entry)

	// Check Tracestate does not have key3.
	checkKeyValue(t, tracestate, key3, "", testname)
	// Check that we added the key3 in the entries
	tracestate, err = New(nil, entries...)
	checkError(t, tracestate, err, testname, "create failed")
	checkKeyValue(t, tracestate, key3, value3, testname)
}

func TestKeyWithValidChar(t *testing.T) {
	testname := "TestKeyWithValidChar"

	arrayRune := []rune("")
	for c := 'a'; c <= 'z'; c++ {
		arrayRune = append(arrayRune, c)
	}
	for c := '0'; c <= '9'; c++ {
		arrayRune = append(arrayRune, c)
	}
	arrayRune = append(arrayRune, '_')
	arrayRune = append(arrayRune, '-')
	arrayRune = append(arrayRune, '*')
	arrayRune = append(arrayRune, '/')
	key := string(arrayRune)
	entry := Entry{key, "world"}
	tracestate, err := New(nil, entry)

	checkError(t, tracestate, err, testname, "create failed when the key contains all valid characters")
}

func TestKeyWithInvalidChar(t *testing.T) {
	testname := "TestKeyWithInvalidChar"

	keys := []string{"1ab", "1ab2", "Abc", " abc", "a=b"}

	for _, key := range keys {
		entry := Entry{key, "world"}
		tracestate, err := New(nil, entry)
		wantError(t, tracestate, err, testname, fmt.Sprintf(
			"create did not err with invalid key=%q", key))
	}
}

func TestNilKey(t *testing.T) {
	testname := "TestNilKey"

	entry := Entry{"", "world"}
	tracestate, err := New(nil, entry)
	wantError(t, tracestate, err, testname, "create did not err when the key is nil (\"\")")
}

func TestValueWithInvalidChar(t *testing.T) {
	testname := "TestValueWithInvalidChar"

	keys := []string{"A=B", "A,B", "AB "}

	for _, value := range keys {
		entry := Entry{"hello", value}
		tracestate, err := New(nil, entry)
		wantError(t, tracestate, err, testname,
			fmt.Sprintf("create did not err when the value is invalid (%q)", value))
	}
}

func TestNilValue(t *testing.T) {
	testname := "TestNilValue"

	tracestate, err := New(nil, Entry{"hello", ""})
	wantError(t, tracestate, err, testname, "create did not err when the value is nil (\"\")")
}

func TestInvalidKeyLen(t *testing.T) {
	testname := "TestInvalidKeyLen"

	arrayRune := []rune("")
	for i := 0; i <= keyMaxSize+1; i++ {
		arrayRune = append(arrayRune, 'a')
	}
	key := string(arrayRune)
	tracestate, err := New(nil, Entry{key, "world"})

	wantError(t, tracestate, err, testname,
		fmt.Sprintf("create did not err when the length (%d) of the key is larger than max (%d)",
			len(key), keyMaxSize))
}

func TestInvalidValueLen(t *testing.T) {
	testname := "TestInvalidValueLen"

	arrayRune := []rune("")
	for i := 0; i <= valueMaxSize+1; i++ {
		arrayRune = append(arrayRune, 'a')
	}
	value := string(arrayRune)
	tracestate, err := New(nil, Entry{"hello", value})

	wantError(t, tracestate, err, testname,
		fmt.Sprintf("create did not err when the length (%d) of the value is larger than max (%d)",
			len(value), valueMaxSize))
}

func TestCreateFromArrayWithOverLimitKVPairs(t *testing.T) {
	testname := "TestCreateFromArrayWithOverLimitKVPairs"

	entries := []Entry{}
	for i := 0; i <= maxKeyValuePairs; i++ {
		key := fmt.Sprintf("a%db", i)
		entry := Entry{key, "world"}
		entries = append(entries, entry)
	}
	tracestate, err := New(nil, entries...)
	wantError(t, tracestate, err, testname,
		fmt.Sprintf("create did not err when the number (%d) of key-value pairs is larger than max (%d)",
			len(entries), maxKeyValuePairs))
}

func TestCreateFromEmptyArray(t *testing.T) {
	testname := "TestCreateFromEmptyArray"

	tracestate, err := New(nil, nil...)
	checkError(t, tracestate, err, testname,
		"failed to create nil tracestate")
}

func TestCreateFromParentWithOverLimitKVPairs(t *testing.T) {
	testname := "TestCreateFromParentWithOverLimitKVPairs"

	entries := []Entry{}
	for i := 0; i < maxKeyValuePairs; i++ {
		key := fmt.Sprintf("a%db", i)
		entry := Entry{key, "world"}
		entries = append(entries, entry)
	}
	parent, err := New(nil, entries...)

	checkError(t, parent, err, testname, fmt.Sprintf("create failed to add %d key-value pair", maxKeyValuePairs))

	// Add one more to go over the limit
	key := fmt.Sprintf("a%d", maxKeyValuePairs)
	tracestate, err := New(parent, Entry{key, "world"})
	wantError(t, tracestate, err, testname,
		fmt.Sprintf("create did not err when attempted to exceed the key-value pair limit of %d", maxKeyValuePairs))
}

func TestCreateFromArrayWithDuplicateKeys(t *testing.T) {
	key1, value1, key2, value2, key3, value3 := "hello", "world", "foo", "bar", "hello", "baz"
	testname := "TestCreateFromArrayWithDuplicateKeys"

	entry1 := Entry{key1, value1}
	entry2 := Entry{key2, value2}
	entry3 := Entry{key3, value3}
	tracestate, err := New(nil, entry1, entry2, entry3)

	wantError(t, tracestate, err, testname,
		"create did not err when entries contained duplicate keys")
}

func TestEntriesWithNil(t *testing.T) {
	ts, err := New(nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(ts.Entries()), 0; got != want {
		t.Errorf("zero value should have no entries, got %v; want %v", got, want)
	}
}
