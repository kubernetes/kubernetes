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
	"reflect"
	"testing"
	"time"

	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
)

func TestRecordStartup(t *testing.T) {
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
		ls := recordStartup(c.ls)
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

func TestStartupsSince(t *testing.T) {
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
		num, err := startupsSince(c.ls, now)
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		if num != c.expect {
			t.Errorf("case %q, expect %d startups but got %d", c.desc, c.expect, num)
		}
	}

}

// returns true if the timestamps are monotically increasing, false otherwise
func timestampsSorted(ls []string) (bool, error) {
	if len(ls) < 2 {
		return true, nil
	}
	prev, err := time.Parse(time.RFC3339, ls[0])
	if err != nil {
		return false, err
	}
	for _, stamp := range ls[1:] {
		cur, err := time.Parse(time.RFC3339, stamp)
		if err != nil {
			return false, err
		}
		if !cur.After(prev) {
			return false, nil
		}
		prev = cur
	}
	return true, nil
}
