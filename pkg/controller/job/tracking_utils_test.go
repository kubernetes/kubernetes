/*
Copyright 2020 The Kubernetes Authors.

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

package job

import (
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestUIDTrackingExpectations(t *testing.T) {
	tracks := []struct {
		job         string
		firstRound  []string
		secondRound []string
	}{
		{
			job:         "foo",
			firstRound:  []string{"a", "b", "c", "d"},
			secondRound: []string{"e", "f"},
		},
		{
			job:         "bar",
			firstRound:  []string{"x", "y", "z"},
			secondRound: []string{"u", "v", "w"},
		},
		{
			job:         "baz",
			firstRound:  []string{"w"},
			secondRound: []string{"a"},
		},
	}
	expectations := newUIDTrackingExpectations()

	// Insert first round of keys in parallel.

	var wg sync.WaitGroup
	wg.Add(len(tracks))
	errs := make([]error, len(tracks))
	for i := range tracks {
		track := tracks[i]
		go func(errID int) {
			errs[errID] = expectations.expectFinalizersRemoved(track.job, track.firstRound)
			wg.Done()
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Errorf("Failed adding first round of UIDs for job %s: %v", tracks[i].job, err)
		}
	}

	for _, track := range tracks {
		uids := expectations.getSet(track.job)
		if uids == nil {
			t.Errorf("Set of UIDs is empty for job %s", track.job)
		} else if diff := cmp.Diff(track.firstRound, uids.set.List()); diff != "" {
			t.Errorf("Unexpected keys for job %s (-want,+got):\n%s", track.job, diff)
		}
	}

	// Delete the first round of keys and add the second round in parallel.

	for i, track := range tracks {
		wg.Add(len(track.firstRound) + 1)
		track := track
		for _, uid := range track.firstRound {
			uid := uid
			go func() {
				expectations.finalizerRemovalObserved(track.job, uid)
				wg.Done()
			}()
		}
		go func(errID int) {
			errs[errID] = expectations.expectFinalizersRemoved(track.job, track.secondRound)
			wg.Done()
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("Failed adding second round of UIDs for job %s: %v", tracks[i].job, err)
		}
	}

	for _, track := range tracks {
		uids := expectations.getSet(track.job)
		if uids == nil {
			t.Errorf("Set of UIDs is empty for job %s", track.job)
		} else if diff := cmp.Diff(track.secondRound, uids.set.List()); diff != "" {
			t.Errorf("Unexpected keys for job %s (-want,+got):\n%s", track.job, diff)
		}
	}
}
