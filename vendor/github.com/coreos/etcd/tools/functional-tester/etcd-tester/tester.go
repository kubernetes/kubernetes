// Copyright 2015 CoreOS, Inc.
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
	"sync"
	"time"
)

type tester struct {
	failures []failure
	cluster  *cluster
	limit    int

	status Status
}

func (tt *tester) runLoop() {
	tt.status.Since = time.Now()
	tt.status.RoundLimit = tt.limit
	tt.status.cluster = tt.cluster
	for _, f := range tt.failures {
		tt.status.Failures = append(tt.status.Failures, f.Desc())
	}
	for i := 0; i < tt.limit; i++ {
		tt.status.setRound(i)
		roundTotalCounter.Inc()

		var currentRevision int64
		for j, f := range tt.failures {
			caseTotalCounter.WithLabelValues(f.Desc()).Inc()

			tt.status.setCase(j)

			if err := tt.cluster.WaitHealth(); err != nil {
				plog.Printf("[round#%d case#%d] wait full health error: %v", i, j, err)
				if err := tt.cleanup(i, j); err != nil {
					plog.Printf("[round#%d case#%d] cleanup error: %v", i, j, err)
					return
				}
				continue
			}
			plog.Printf("[round#%d case#%d] start failure %s", i, j, f.Desc())

			plog.Printf("[round#%d case#%d] start injecting failure...", i, j)
			if err := f.Inject(tt.cluster, i); err != nil {
				plog.Printf("[round#%d case#%d] injection error: %v", i, j, err)
				if err := tt.cleanup(i, j); err != nil {
					plog.Printf("[round#%d case#%d] cleanup error: %v", i, j, err)
					return
				}
				continue
			}
			plog.Printf("[round#%d case#%d] injected failure", i, j)

			plog.Printf("[round#%d case#%d] start recovering failure...", i, j)
			if err := f.Recover(tt.cluster, i); err != nil {
				plog.Printf("[round#%d case#%d] recovery error: %v", i, j, err)
				if err := tt.cleanup(i, j); err != nil {
					plog.Printf("[round#%d case#%d] cleanup error: %v", i, j, err)
					return
				}
				continue
			}
			plog.Printf("[round#%d case#%d] recovered failure", i, j)

			if tt.cluster.v2Only {
				plog.Printf("[round#%d case#%d] succeed!", i, j)
				continue
			}

			plog.Printf("[round#%d case#%d] canceling the stressers...", i, j)
			for _, s := range tt.cluster.Stressers {
				s.Cancel()
			}
			plog.Printf("[round#%d case#%d] canceled stressers", i, j)

			plog.Printf("[round#%d case#%d] checking current revisions...", i, j)
			var (
				revs   map[string]int64
				hashes map[string]int64
				rerr   error
				ok     bool
			)
			for k := 0; k < 5; k++ {
				time.Sleep(time.Second)

				revs, hashes, rerr = tt.cluster.getRevisionHash()
				if rerr != nil {
					plog.Printf("[round#%d case#%d.%d] failed to get current revisions (%v)", i, j, k, rerr)
					continue
				}
				if currentRevision, ok = getSameValue(revs); ok {
					break
				}

				plog.Printf("[round#%d case#%d.%d] inconsistent current revisions %+v", i, j, k, revs)
			}
			if !ok || rerr != nil {
				plog.Printf("[round#%d case#%d] checking current revisions failed (%v)", i, j, revs)
				if err := tt.cleanup(i, j); err != nil {
					plog.Printf("[round#%d case#%d] cleanup error: %v", i, j, err)
					return
				}
				continue
			}
			plog.Printf("[round#%d case#%d] all members are consistent with current revisions", i, j)

			plog.Printf("[round#%d case#%d] checking current storage hashes...", i, j)
			if _, ok = getSameValue(hashes); !ok {
				plog.Printf("[round#%d case#%d] checking current storage hashes failed (%v)", i, j, hashes)
				if err := tt.cleanup(i, j); err != nil {
					plog.Printf("[round#%d case#%d] cleanup error: %v", i, j, err)
					return
				}
				continue
			}
			plog.Printf("[round#%d case#%d] all members are consistent with storage hashes", i, j)

			plog.Printf("[round#%d case#%d] restarting the stressers...", i, j)
			for _, s := range tt.cluster.Stressers {
				go s.Stress()
			}

			plog.Printf("[round#%d case#%d] succeed!", i, j)
		}

		revToCompact := max(0, currentRevision-10000)
		plog.Printf("[round#%d] compacting storage at %d (current revision %d)", i, revToCompact, currentRevision)
		if err := tt.cluster.compactKV(revToCompact); err != nil {
			plog.Printf("[round#%d] compactKV error (%v)", i, err)
			if err := tt.cleanup(i, 0); err != nil {
				plog.Printf("[round#%d] cleanup error: %v", i, err)
				return
			}
			continue
		}
		plog.Printf("[round#%d] compacted storage", i)

		// TODO: make sure compaction is finished
		time.Sleep(30 * time.Second)
	}
}

func (tt *tester) cleanup(i, j int) error {
	roundFailedTotalCounter.Inc()
	caseFailedTotalCounter.WithLabelValues(tt.failures[j].Desc()).Inc()

	plog.Printf("[round#%d case#%d] cleaning up...", i, j)
	if err := tt.cluster.Cleanup(); err != nil {
		return err
	}
	return tt.cluster.Bootstrap()
}

type Status struct {
	Since      time.Time
	Failures   []string
	RoundLimit int

	Cluster ClusterStatus
	cluster *cluster

	mu    sync.Mutex // guards Round and Case
	Round int
	Case  int
}

// get gets a copy of status
func (s *Status) get() Status {
	s.mu.Lock()
	got := *s
	cluster := s.cluster
	s.mu.Unlock()
	got.Cluster = cluster.Status()
	return got
}

func (s *Status) setRound(r int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Round = r
}

func (s *Status) setCase(c int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Case = c
}
