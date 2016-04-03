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

package main

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

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

		// This will mark the flag as changed, so it will have
		// precedence over the configuration and the default
		// value.
		cmdRkt.PersistentFlags().Set("dir", d)
		if err := initPods(); err != nil {
			t.Fatalf("error initializing pods: %v", err)
		}

		var (
			n_expected int
			n_walked   int
			n_matched  int
			included   includeMask
		)

		// create the pod dirs as specified by the test
		for _, ct := range tt {
			var cp string
			if ct.garbage {
				cp = filepath.Join(exitedGarbageDir(), ct.uuid)
				included |= includeExitedGarbageDir
			} else {
				cp = filepath.Join(runDir(), ct.uuid)
				included |= includeRunDir
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
		if err := walkPods(included, func(ch *pod) {
			n_walked++
			for _, ct := range tt {
				if ch.uuid.String() == ct.uuid &&
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
