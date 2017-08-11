/*
Copyright 2017 The Kubernetes Authors.

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

package badconfig

import (
	"fmt"
	"reflect"
	"testing"
	"time"
)

func TestMarkBad(t *testing.T) {
	// build a map with one entry
	m := map[string]Entry{}
	uid := "uid"
	reason := "reason"
	markBad(m, uid, reason)

	// the entry should exist for uid
	entry, ok := m[uid]
	if !ok {
		t.Fatalf("expect entry for uid %q, but none exists", uid)
	}

	// the entry's reason should match the reason it was marked bad with
	if entry.Reason != reason {
		t.Errorf("expect Entry.Reason %q, but got %q", reason, entry.Reason)
	}

	// the entry's timestamp should be in RFC3339 format
	if err := assertRFC3339(entry.Time); err != nil {
		t.Errorf("expect Entry.Time to use RFC3339 format, but got %q, error: %v", entry.Time, err)
	}

	// it should be the only entry in the map thus far
	if n := len(m); n != 1 {
		t.Errorf("expect one entry in the map, but got %d", n)
	}

}

func TestGetEntry(t *testing.T) {
	nowstamp := time.Now().Format(time.RFC3339)
	uid := "uid"
	expect := &Entry{
		Time:   nowstamp,
		Reason: "reason",
	}
	m := map[string]Entry{uid: *expect}

	// should return nil for entries that don't exist
	bogus := "bogus-uid"
	if e := getEntry(m, bogus); e != nil {
		t.Errorf("expect nil for entries that don't exist (uid: %q), but got %#v", bogus, e)
	}

	// should return non-nil for entries that exist
	if e := getEntry(m, uid); e == nil {
		t.Errorf("expect non-nil for entries that exist (uid: %q), but got nil", uid)
	} else if !reflect.DeepEqual(expect, e) {
		// entry should match what we inserted for the given UID
		t.Errorf("expect entry for uid %q to match %#v, but got %#v", uid, expect, e)
	}
}

func TestEncode(t *testing.T) {
	nowstamp := time.Now().Format(time.RFC3339)
	uid := "uid"
	expect := fmt.Sprintf(`{"%s":{"time":"%s","reason":"reason"}}`, uid, nowstamp)
	m := map[string]Entry{uid: {
		Time:   nowstamp,
		Reason: "reason",
	}}

	data, err := encode(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	json := string(data)

	if json != expect {
		t.Errorf("expect encoding of %#v to match %q, but got %q", m, expect, json)
	}
}

func TestDecode(t *testing.T) {
	nowstamp := time.Now().Format(time.RFC3339)
	uid := "uid"
	valid := []byte(fmt.Sprintf(`{"%s":{"time":"%s","reason":"reason"}}`, uid, nowstamp))
	expect := map[string]Entry{uid: {
		Time:   nowstamp,
		Reason: "reason",
	}}

	// decoding valid json should result in an object with the correct values
	if m, err := decode(valid); err != nil {
		t.Errorf("expect decoding valid json %q to produce a map, but got error: %v", valid, err)
	} else if !reflect.DeepEqual(expect, m) {
		// m should equal expected decoded object
		t.Errorf("expect decoding valid json %q to produce %#v, but got %#v", valid, expect, m)
	}

	// decoding invalid json should return an error
	invalid := []byte(`invalid`)
	if m, err := decode(invalid); err == nil {
		t.Errorf("expect decoding invalid json %q to return an error, but decoded to %#v", invalid, m)
	}
}

func TestRoundTrip(t *testing.T) {
	nowstamp := time.Now().Format(time.RFC3339)
	uid := "uid"
	expect := map[string]Entry{uid: {
		Time:   nowstamp,
		Reason: "reason",
	}}

	// test that encoding and decoding an object results in the same value
	data, err := encode(expect)
	if err != nil {
		t.Fatalf("failed to encode %#v, error: %v", expect, err)
	}
	after, err := decode(data)
	if err != nil {
		t.Fatalf("failed to decode %q, error: %v", string(data), err)
	}
	if !reflect.DeepEqual(expect, after) {
		t.Errorf("expect round-tripping %#v to result in the same value, but got %#v", expect, after)
	}
}

func assertRFC3339(s string) error {
	tm, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return fmt.Errorf("expect RFC3339 format, but failed to parse, error: %v", err)
	}
	// parsing succeeded, now finish round-trip and compare
	rt := tm.Format(time.RFC3339)
	if rt != s {
		return fmt.Errorf("expect RFC3339 format, but failed to round trip unchanged, original %q, round-trip %q", s, rt)
	}
	return nil
}
