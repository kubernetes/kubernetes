// Copyright 2015 The rkt Authors
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

package pod

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/pkg/lock"
)

func TestWalkPods(t *testing.T) {
	tests := [][]*struct {
		uuid      string
		exited    bool
		garbage   bool
		deleting  bool
		expected  bool
		n_matched int
	}{
		{ // nothing
		},
		{ // single executing pod
			{
				uuid:     "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
				exited:   false,
				garbage:  false,
				deleting: false,

				expected: true,
			},
		},
		{ // single exited pod
			{
				uuid:     "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
				exited:   true,
				garbage:  false,
				deleting: false,

				expected: true,
			},
		},
		{ // single garbage pod
			{
				uuid:     "cccccccc-cccc-cccc-cccc-cccccccccccc",
				exited:   true,
				garbage:  true,
				deleting: false,

				expected: true,
			},
		},
		{ // single deleting pod
			{
				uuid:     "dddddddd-dddd-dddd-dddd-dddddddddddd",
				exited:   true,
				garbage:  true,
				deleting: true,

				expected: true,
			},
		},
		{ // one of each
			{ // executing
				uuid:     "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
				exited:   false,
				garbage:  false,
				deleting: false,

				expected: true,
			},
			{ // exited
				uuid:     "ffffffff-ffff-ffff-ffff-ffffffffffff",
				exited:   true,
				garbage:  false,
				deleting: false,

				expected: true,
			},
			{ // garbage
				uuid:     "f0f0f0f0-f0f0-f0f0-f0f0-f0f0f0f0f0f0",
				exited:   true,
				garbage:  true,
				deleting: false,

				expected: true,
			},
			{ // deleting
				uuid:     "f1f1f1f1-f1f1-f1f1-f1f1-f1f1f1f1f1f1",
				exited:   true,
				garbage:  true,
				deleting: true,

				expected: true,
			},
		},
		// TODO(vc): update to test new prepared/prepare-failed/non-exited-garbage states..
	}

	for _, tt := range tests {
		// start every test with a clean slate
		d, err := ioutil.TempDir("", "")
		if err != nil {
			t.Fatalf("error creating tmpdir: %v", err)
		}
		defer os.RemoveAll(d)

		if err := initPods(d); err != nil {
			t.Fatalf("error initializing pods: %v", err)
		}

		var (
			n_expected int
			n_walked   int
			n_matched  int
			included   IncludeMask
		)

		// create the pod dirs as specified by the test
		for _, ct := range tt {
			var cp string
			if ct.garbage {
				cp = filepath.Join(exitedGarbageDir(d), ct.uuid)
				included |= IncludeExitedGarbageDir
			} else {
				cp = filepath.Join(runDir(d), ct.uuid)
				included |= IncludeRunDir
			}

			if err := os.MkdirAll(cp, 0700); err != nil {
				t.Fatalf("error creating pod directory: %v", err)
			}

			if !ct.exited || ct.deleting { // acquire lock to simulate running and deleting pods
				l, err := lock.ExclusiveLock(cp, lock.Dir)
				if err != nil {
					t.Fatalf("error locking pod: %v", err)
				}
				defer l.Close()
			}

			if ct.expected {
				n_expected++
			}
		}

		// match what walk provided against the set in the test
		if err := WalkPods(d, included, func(ch *Pod) {
			n_walked++
			for _, ct := range tt {
				if ch.UUID.String() == ct.uuid &&
					ch.isExitedGarbage == ct.garbage &&
					ch.isExited == ct.exited &&
					ch.isExitedDeleting == ct.deleting {

					ct.n_matched++
					if ct.n_matched > 1 {
						t.Errorf("no pods should match multiple times")
					}
					n_matched++
				}
			}
		}); err != nil {
			t.Fatalf("error walking pods: %v", err)
		}

		if n_expected != n_matched {
			t.Errorf("walked: %d expected: %d matched: %d", n_walked, n_expected, n_matched)
		}

		for _, ct := range tt {
			if ct.expected && ct.n_matched == 0 {
				t.Errorf("pod %q expected but not matched", ct.uuid)
			}

			if !ct.expected && ct.n_matched != 0 {
				t.Errorf("pod %q matched but not expected", ct.uuid)
			}
		}
	}
}

type states struct {
	isEmbryo         bool
	isPreparing      bool
	isAbortedPrepare bool
	isPrepared       bool
	isExited         bool
	isExitedGarbage  bool
	isExitedDeleting bool
	isGarbage        bool
	isDeleting       bool
	isGone           bool
}

type dirFn func(string) string

func TestGetPodAndRefreshState(t *testing.T) {
	testCases := []struct {
		paths    []dirFn
		locks    []dirFn
		expected states
	}{
		{
			paths:    []dirFn{embryoDir},
			expected: states{isEmbryo: true},
		},
		{
			paths:    []dirFn{prepareDir},
			locks:    []dirFn{prepareDir},
			expected: states{isPreparing: true},
		},
		{
			paths:    []dirFn{prepareDir},
			expected: states{isAbortedPrepare: true},
		},
		{
			paths:    []dirFn{runDir},
			locks:    []dirFn{runDir},
			expected: states{},
		},
		{
			paths:    []dirFn{runDir},
			expected: states{isExited: true},
		},
		{
			paths:    []dirFn{garbageDir},
			expected: states{isGarbage: true},
		},
		{
			paths:    []dirFn{garbageDir},
			locks:    []dirFn{garbageDir},
			expected: states{isGarbage: true, isDeleting: true},
		},
	}

	uuid, err := types.NewUUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	if err != nil {
		panic(err)
	}

	for i, tcase := range testCases {
		tmpDir, err := ioutil.TempDir("", "")
		if err != nil {
			panic(err)
		}
		defer os.RemoveAll(tmpDir)

		for _, pfn := range tcase.paths {
			podPath := filepath.Join(pfn(tmpDir), uuid.String())
			if err := os.MkdirAll(podPath, 0777); err != nil {
				panic(err)
			}
		}

		for _, lfn := range tcase.locks {
			podPath := filepath.Join(lfn(tmpDir), uuid.String())
			l, err := lock.NewLock(podPath, lock.Dir)
			if err != nil {
				t.Fatalf("error taking lock on directory: %v", err)
			}
			err = l.ExclusiveLock()
			if err != nil {
				t.Fatalf("could not get exclusive lock on directory: %v", err)
			}
			defer l.Unlock()
		}

		p, err := getPod(tmpDir, uuid)
		if err != nil {
			t.Fatalf("%v: unable to get pod: %v", i, err)
		}

		pstate := podToStates(p)
		if !reflect.DeepEqual(tcase.expected, pstate) {
			t.Errorf("%v: expected %+v == %+v after getPod", i, tcase.expected, pstate)
		}

		err = p.refreshState()
		if err != nil {
			t.Errorf("error refreshing state: %v", err)
			continue
		}

		pstate = podToStates(p)
		if !reflect.DeepEqual(tcase.expected, pstate) {
			t.Errorf("%v: expected %+v == %+v after refrshState", i, tcase.expected, pstate)
		}
	}
}

func TestRefreshPodIsGone(t *testing.T) {
	uuid, err := types.NewUUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	if err != nil {
		panic(err)
	}
	tmpDir, err := ioutil.TempDir("", "")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(tmpDir)
	podPath := filepath.Join(embryoDir(tmpDir), uuid.String())
	os.MkdirAll(podPath, 0777)

	p, err := getPod(tmpDir, uuid)
	if err != nil {
		t.Fatalf("unable to get pod: %v", err)
	}

	os.RemoveAll(tmpDir)

	err = p.refreshState()
	if err != nil {
		t.Fatalf("error refreshing state: %v", err)
	}

	pstate := podToStates(p)
	expected := states{isGone: true}
	if !reflect.DeepEqual(expected, pstate) {
		t.Errorf("expected %+v == %+v after refrshState", expected, pstate)
	}
}

func podToStates(p *Pod) states {
	return states{
		isEmbryo:         p.isEmbryo,
		isPreparing:      p.isPreparing,
		isAbortedPrepare: p.isAbortedPrepare,
		isPrepared:       p.isPrepared,
		isExited:         p.isExited,
		isExitedGarbage:  p.isExitedGarbage,
		isExitedDeleting: p.isExitedDeleting,
		isGarbage:        p.isGarbage,
		isDeleting:       p.isDeleting,
		isGone:           p.isGone,
	}
}
