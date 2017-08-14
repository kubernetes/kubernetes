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

package startups

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

// TODO(mtaufen): this file reuses a lot of test code from startups_test.go, should consolidate

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
		t.Fatalf("tracker.Initialize() failed with error: %v", err)
	}

	// check that testTrackingDir exists
	_, err = tracker.fs.Stat(testTrackingDir)
	if err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", testTrackingDir, err)
	}

	// check that testTrackingDir contains the startupsFile
	path := filepath.Join(testTrackingDir, startupsFile)
	_, err = tracker.fs.Stat(path)
	if err != nil {
		t.Fatalf("expect %q to exist, but stat failed with error: %v", path, err)
	}
}

func TestFsTrackerRecordStartup(t *testing.T) {
	tracker, err := newInitializedFakeFsTracker()
	if err != nil {
		t.Fatalf("failed to construct a tracker, error: %v", err)
	}

	now := time.Now()

	fullList := func() []string {
		ls := []string{}
		for i := maxStartups; i > 0; i-- {
			// subtract decreasing amounts so timestamps increase but remain in the past
			ls = append(ls, now.Add(-time.Duration(i)*time.Second).Format(time.RFC3339))
		}
		return ls
	}()
	cases := []struct {
		desc       string
		ls         []string
		expectHead []string // what we expect the first length-1 elements to look like after recording a new timestamp
		expectLen  int      // how long the list should be after recording
	}{
		// start empty
		{
			"start empty",
			[]string{},
			[]string{},
			1,
		},
		// start non-empty
		{
			"start non-empty",
			// subtract 1 so stamps are in the past
			[]string{now.Add(-1 * time.Second).Format(time.RFC3339)},
			[]string{now.Add(-1 * time.Second).Format(time.RFC3339)},
			2,
		},
		// rotate list
		{
			"rotate list",
			// make a slice with len == maxStartups, containing monotonically-increasing timestamps
			fullList,
			fullList[1:],
			maxStartups,
		},
	}

	for _, c := range cases {
		// save the starting point, record a "startup" time, then load list from fs
		if err := tracker.save(c.ls); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if err := tracker.RecordStartup(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		ls, err := tracker.load()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if c.expectLen != len(ls) {
			t.Errorf("case %q, expected list %q to have length %d", c.desc, ls, c.expectLen)
		}
		if !reflect.DeepEqual(c.expectHead, ls[:len(ls)-1]) {
			t.Errorf("case %q, expected elements 0 through n-1 of list %q to equal %q", c.desc, ls, c.expectHead)
		}
		// timestamps should be monotonically increasing (assuming system clock isn't jumping around at least)
		if sorted, err := timestampsSorted(ls); err != nil {
			t.Fatalf("unexpected error: %v", err)
		} else if !sorted {
			t.Errorf("case %q, expected monotonically increasing timestamps, but got %q", c.desc, ls)
		}
	}
}

func TestFsTrackerStartupsSince(t *testing.T) {
	tracker, err := newInitializedFakeFsTracker()
	if err != nil {
		t.Fatalf("failed to construct a tracker, error: %v", err)
	}

	now, err := time.Parse(time.RFC3339, "2017-01-02T15:04:05Z")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []struct {
		desc   string
		ls     []string
		expect int32
		err    string
	}{
		// empty list
		{"empty list", []string{}, 0, ""},
		// no startups since
		{
			"no startups since",
			[]string{"2014-01-02T15:04:05Z", "2015-01-02T15:04:05Z", "2016-01-02T15:04:05Z"},
			0,
			"",
		},
		// 2 startups since
		{
			"some startups since",
			[]string{"2016-01-02T15:04:05Z", "2018-01-02T15:04:05Z", "2019-01-02T15:04:05Z"},
			2,
			"",
		},
		// all startups since
		{
			"all startups since",
			[]string{"2018-01-02T15:04:05Z", "2019-01-02T15:04:05Z", "2020-01-02T15:04:05Z"},
			3,
			"",
		},
		// invalid timestamp
		{"invalid timestamp", []string{"2018-01-02T15:04:05Z08:00"}, 0, "failed to parse"},
	}

	for _, c := range cases {
		if err := tracker.save(c.ls); err != nil {
			t.Fatalf("unexected error: %v", err)
		}
		num, err := tracker.StartupsSince(now)
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if num != c.expect {
			t.Errorf("case %q, expect %d startups but got %d", c.desc, c.expect, num)
		}
	}
}

func TestFsTrackerLoad(t *testing.T) {
	tracker, err := newInitializedFakeFsTracker()
	if err != nil {
		t.Fatalf("failed to construct a tracker, error: %v", err)
	}

	nowstamp := time.Now().Format(time.RFC3339)
	cases := []struct {
		desc   string
		data   []byte
		expect []string
		err    string
	}{
		// empty file
		{"empty file", []byte(""), []string{}, ""},
		// empty list
		{"empty list", []byte("[]"), []string{}, ""},
		// valid json
		{"valid json", []byte(fmt.Sprintf(`["%s"]`, nowstamp)), []string{nowstamp}, ""},
		// invalid json
		{"invalid json", []byte(`*`), []string{}, "failed to unmarshal"},
	}

	for _, c := range cases {
		// save a file containing the correct serialization
		utilfiles.ReplaceFile(tracker.fs, filepath.Join(testTrackingDir, startupsFile), c.data)

		// loading valid json should result in an object with the correct serialization
		ls, err := tracker.load()
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if !reflect.DeepEqual(c.expect, ls) {
			// ls should equal expected decoded object
			t.Errorf("case %q, expect %#v but got %#v", c.desc, c.expect, ls)
		}
	}

}

func TestFsTrackerSave(t *testing.T) {
	tracker, err := newInitializedFakeFsTracker()
	if err != nil {
		t.Fatalf("failed to construct a tracker, error: %v", err)
	}

	nowstamp := time.Now().Format(time.RFC3339)
	cases := []struct {
		desc   string
		ls     []string
		expect string
		err    string
	}{
		// empty list
		{"empty list", []string{}, "[]", ""},
		// 1-entry list
		{"valid json", []string{nowstamp}, fmt.Sprintf(`["%s"]`, nowstamp), ""},
	}

	for _, c := range cases {
		if err := tracker.save(c.ls); utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}

		data, err := tracker.fs.ReadFile(filepath.Join(testTrackingDir, startupsFile))
		if err != nil {
			t.Fatalf("failed to read startups file, error: %v", err)
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
	expect := []string{nowstamp}

	// test that saving and loading an object results in the same value
	err = tracker.save(expect)
	if err != nil {
		t.Fatalf("failed to save startups data, error: %v", err)
	}
	after, err := tracker.load()
	if err != nil {
		t.Fatalf("failed to load startups data, error: %v", err)
	}
	if !reflect.DeepEqual(expect, after) {
		t.Errorf("expect round-tripping %#v to result in the same value, but got %#v", expect, after)
	}
}
