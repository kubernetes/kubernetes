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
	"path/filepath"
	"reflect"
	"testing"
	"time"

	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utilfs "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/filesystem"
	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
)

const testTrackingDir = "/test-tracking-dir"

// TODO(mtaufen): this file reuses a lot of test code from badconfig_test.go, should consolidate

func newInitializedFakeFsTracker() (*fsTracker, error) {
	fs := utilfs.NewFakeFs()
	tracker := NewFsTracker(fs, testTrackingDir)
	if err := tracker.Initialize(); err != nil {
		return nil, err
	}
	return tracker.(*fsTracker), nil
}

func TestFsTrackerInitialize(t *testing.T) {
	tracker, err := newInitializedFakeFsTracker()
	if err != nil {
		t.Fatalf("fsTracker.Initialize() failed with error: %v", err)
	}

	// check that testTrackingDir exists
	_, err = tracker.fs.Stat(testTrackingDir)
	if err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", testTrackingDir, err)
	}

	// check that testTrackingDir contains the badConfigsFile
	path := filepath.Join(testTrackingDir, badConfigsFile)
	_, err = tracker.fs.Stat(path)
	if err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", path, err)
	}
}

func TestFsTrackerMarkBad(t *testing.T) {
	tracker, err := newInitializedFakeFsTracker()
	if err != nil {
		t.Fatalf("failed to construct a tracker, error: %v", err)
	}

	// create a bad config entry in the fs
	uid := "uid"
	reason := "reason"
	tracker.MarkBad(uid, reason)

	// load the map from the fs
	m, err := tracker.load()
	if err != nil {
		t.Fatalf("failed to load bad-config data, error: %v", err)
	}

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

func TestFsTrackerEntry(t *testing.T) {
	tracker, err := newInitializedFakeFsTracker()
	if err != nil {
		t.Fatalf("failed to construct a tracker, error: %v", err)
	}

	// manually save a correct entry to fs
	nowstamp := time.Now().Format(time.RFC3339)
	uid := "uid"
	expect := &Entry{
		Time:   nowstamp,
		Reason: "reason",
	}
	m := map[string]Entry{uid: *expect}
	err = tracker.save(m)
	if err != nil {
		t.Fatalf("failed to save bad-config data, error: %v", err)
	}

	// should return nil for entries that don't exist
	bogus := "bogus-uid"
	e, err := tracker.Entry(bogus)
	if err != nil {
		t.Errorf("expect nil for entries that don't exist (uid: %q), but got error: %v", bogus, err)
	} else if e != nil {
		t.Errorf("expect nil for entries that don't exist (uid: %q), but got %#v", bogus, e)
	}

	// should return non-nil for entries that exist
	e, err = tracker.Entry(uid)
	if err != nil {
		t.Errorf("expect non-nil for entries that exist (uid: %q), but got error: %v", uid, err)
	} else if e == nil {
		t.Errorf("expect non-nil for entries that exist (uid: %q), but got nil", uid)
	} else if !reflect.DeepEqual(expect, e) {
		// entry should match what we inserted for the given UID
		t.Errorf("expect entry for uid %q to match %#v, but got %#v", uid, expect, e)
	}
}

// TODO(mtaufen): test loading invalid json (see startups/fstracker_test.go for example)
func TestFsTrackerLoad(t *testing.T) {
	tracker, err := newInitializedFakeFsTracker()
	if err != nil {
		t.Fatalf("failed to construct a tracker, error: %v", err)
	}

	uid := "uid"
	nowstamp := time.Now().Format(time.RFC3339)
	cases := []struct {
		desc   string
		data   []byte
		expect map[string]Entry
		err    string
	}{
		// empty file
		{"empty file", []byte(""), map[string]Entry{}, ""},
		// empty map
		{"empty map", []byte("{}"), map[string]Entry{}, ""},
		// valid json
		{"valid json", []byte(fmt.Sprintf(`{"%s":{"time":"%s","reason":"reason"}}`, uid, nowstamp)),
			map[string]Entry{uid: {
				Time:   nowstamp,
				Reason: "reason",
			}}, ""},
		// invalid json
		{"invalid json", []byte(`*`), map[string]Entry{}, "failed to unmarshal"},
	}

	for _, c := range cases {
		// save a file containing the correct serialization
		utilfiles.ReplaceFile(tracker.fs, filepath.Join(testTrackingDir, badConfigsFile), c.data)

		// loading valid json should result in an object with the correct values
		m, err := tracker.load()
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if !reflect.DeepEqual(c.expect, m) {
			// m should equal expected decoded object
			t.Errorf("case %q, expect %#v but got %#v", c.desc, c.expect, m)
		}
	}
}

func TestFsTrackerSave(t *testing.T) {
	tracker, err := newInitializedFakeFsTracker()
	if err != nil {
		t.Fatalf("failed to construct a tracker, error: %v", err)
	}

	uid := "uid"
	nowstamp := time.Now().Format(time.RFC3339)
	cases := []struct {
		desc   string
		m      map[string]Entry
		expect string
		err    string
	}{
		// empty map
		{"empty map", map[string]Entry{}, "{}", ""},
		// 1-entry map
		{"1-entry map",
			map[string]Entry{uid: {
				Time:   nowstamp,
				Reason: "reason",
			}},
			fmt.Sprintf(`{"%s":{"time":"%s","reason":"reason"}}`, uid, nowstamp), ""},
	}

	for _, c := range cases {
		if err := tracker.save(c.m); utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}

		data, err := tracker.fs.ReadFile(filepath.Join(testTrackingDir, badConfigsFile))
		if err != nil {
			t.Fatalf("failed to read bad-config file, error: %v", err)
		}
		json := string(data)

		if json != c.expect {
			t.Errorf("case %q, expect %q but got %q", c.desc, c.expect, json)
		}
	}
}

func TestFsTrackerRoundTrip(t *testing.T) {
	tracker, err := newInitializedFakeFsTracker()
	if err != nil {
		t.Fatalf("failed to construct a tracker, error: %v", err)
	}

	nowstamp := time.Now().Format(time.RFC3339)
	uid := "uid"
	expect := map[string]Entry{uid: {
		Time:   nowstamp,
		Reason: "reason",
	}}

	// test that saving and loading an object results in the same value
	err = tracker.save(expect)
	if err != nil {
		t.Fatalf("failed to save bad-config data, error: %v", err)
	}
	after, err := tracker.load()
	if err != nil {
		t.Fatalf("failed to load bad-config data, error: %v", err)
	}
	if !reflect.DeepEqual(expect, after) {
		t.Errorf("expect round-tripping %#v to result in the same value, but got %#v", expect, after)
	}
}
