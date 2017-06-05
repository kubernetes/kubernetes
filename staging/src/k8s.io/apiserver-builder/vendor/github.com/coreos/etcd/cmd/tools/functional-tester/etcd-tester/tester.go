// Copyright 2015 The etcd Authors
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
	"fmt"
	"time"
)

type tester struct {
	failures         []failure
	cluster          *cluster
	limit            int
	consistencyCheck bool

	status          Status
	currentRevision int64
}

// compactQPS is rough number of compact requests per second.
// Previous tests showed etcd can compact about 60,000 entries per second.
const compactQPS = 50000

func (tt *tester) runLoop() {
	tt.status.Since = time.Now()
	tt.status.RoundLimit = tt.limit
	tt.status.cluster = tt.cluster
	for _, f := range tt.failures {
		tt.status.Failures = append(tt.status.Failures, f.Desc())
	}

	var (
		round          int
		prevCompactRev int64
	)
	for {
		tt.status.setRound(round)
		tt.status.setCase(-1) // -1 so that logPrefix doesn't print out 'case'
		roundTotalCounter.Inc()

		var failed bool
		for j, f := range tt.failures {
			caseTotalCounter.WithLabelValues(f.Desc()).Inc()
			tt.status.setCase(j)

			if err := tt.cluster.WaitHealth(); err != nil {
				plog.Printf("%s wait full health error: %v", tt.logPrefix(), err)
				if err := tt.cleanup(); err != nil {
					return
				}
				failed = true
				break
			}

			plog.Printf("%s injecting failure %q", tt.logPrefix(), f.Desc())
			if err := f.Inject(tt.cluster, round); err != nil {
				plog.Printf("%s injection error: %v", tt.logPrefix(), err)
				if err := tt.cleanup(); err != nil {
					return
				}
				failed = true
				break
			}
			plog.Printf("%s injected failure", tt.logPrefix())

			plog.Printf("%s recovering failure %q", tt.logPrefix(), f.Desc())
			if err := f.Recover(tt.cluster, round); err != nil {
				plog.Printf("%s recovery error: %v", tt.logPrefix(), err)
				if err := tt.cleanup(); err != nil {
					return
				}
				failed = true
				break
			}
			plog.Printf("%s recovered failure", tt.logPrefix())

			if tt.cluster.v2Only {
				plog.Printf("%s succeed!", tt.logPrefix())
				continue
			}

			if !tt.consistencyCheck {
				if err := tt.updateRevision(); err != nil {
					plog.Warningf("%s functional-tester returning with tt.updateRevision error (%v)", tt.logPrefix(), err)
					return
				}
				continue
			}

			var err error
			failed, err = tt.checkConsistency()
			if err != nil {
				plog.Warningf("%s functional-tester returning with tt.checkConsistency error (%v)", tt.logPrefix(), err)
				return
			}
			if failed {
				break
			}
			plog.Printf("%s succeed!", tt.logPrefix())
		}

		// -1 so that logPrefix doesn't print out 'case'
		tt.status.setCase(-1)

		if failed {
			continue
		}

		revToCompact := max(0, tt.currentRevision-10000)
		compactN := revToCompact - prevCompactRev
		timeout := 10 * time.Second
		if prevCompactRev != 0 && compactN > 0 {
			timeout += time.Duration(compactN/compactQPS) * time.Second
		}
		prevCompactRev = revToCompact

		plog.Printf("%s compacting %d entries (timeout %v)", tt.logPrefix(), compactN, timeout)
		if err := tt.compact(revToCompact, timeout); err != nil {
			plog.Warningf("%s functional-tester compact got error (%v)", tt.logPrefix(), err)
			if err := tt.cleanup(); err != nil {
				return
			}
		}
		if round > 0 && round%500 == 0 { // every 500 rounds
			if err := tt.defrag(); err != nil {
				plog.Warningf("%s functional-tester returning with error (%v)", tt.logPrefix(), err)
				return
			}
		}

		round++
		if round == tt.limit {
			plog.Printf("%s functional-tester is finished", tt.logPrefix())
			break
		}
	}
}

func (tt *tester) updateRevision() error {
	revs, _, err := tt.cluster.getRevisionHash()
	for _, rev := range revs {
		tt.currentRevision = rev
		break // just need get one of the current revisions
	}
	return err
}

func (tt *tester) checkConsistency() (failed bool, err error) {
	tt.cancelStressers()
	defer tt.startStressers()

	plog.Printf("%s updating current revisions...", tt.logPrefix())
	var (
		revs   map[string]int64
		hashes map[string]int64
		rerr   error
		ok     bool
	)
	for i := 0; i < 7; i++ {
		time.Sleep(time.Second)

		revs, hashes, rerr = tt.cluster.getRevisionHash()
		if rerr != nil {
			plog.Printf("%s #%d failed to get current revisions (%v)", tt.logPrefix(), i, rerr)
			continue
		}
		if tt.currentRevision, ok = getSameValue(revs); ok {
			break
		}

		plog.Printf("%s #%d inconsistent current revisions %+v", tt.logPrefix(), i, revs)
	}
	plog.Printf("%s updated current revisions with %d", tt.logPrefix(), tt.currentRevision)

	if !ok || rerr != nil {
		plog.Printf("%s checking current revisions failed [revisions: %v]", tt.logPrefix(), revs)
		failed = true
		err = tt.cleanup()
		return
	}
	plog.Printf("%s all members are consistent with current revisions [revisions: %v]", tt.logPrefix(), revs)

	plog.Printf("%s checking current storage hashes...", tt.logPrefix())
	if _, ok = getSameValue(hashes); !ok {
		plog.Printf("%s checking current storage hashes failed [hashes: %v]", tt.logPrefix(), hashes)
		failed = true
		err = tt.cleanup()
		return
	}
	plog.Printf("%s all members are consistent with storage hashes", tt.logPrefix())
	return
}

func (tt *tester) compact(rev int64, timeout time.Duration) error {
	plog.Printf("%s compacting storage (current revision %d, compact revision %d)", tt.logPrefix(), tt.currentRevision, rev)
	if err := tt.cluster.compactKV(rev, timeout); err != nil {
		return err
	}
	plog.Printf("%s compacted storage (compact revision %d)", tt.logPrefix(), rev)

	plog.Printf("%s checking compaction (compact revision %d)", tt.logPrefix(), rev)
	if err := tt.cluster.checkCompact(rev); err != nil {
		plog.Warningf("%s checkCompact error (%v)", tt.logPrefix(), err)
		return err
	}

	plog.Printf("%s confirmed compaction (compact revision %d)", tt.logPrefix(), rev)
	return nil
}

func (tt *tester) defrag() error {
	plog.Printf("%s defragmenting...", tt.logPrefix())
	if err := tt.cluster.defrag(); err != nil {
		plog.Warningf("%s defrag error (%v)", tt.logPrefix(), err)
		if cerr := tt.cleanup(); cerr != nil {
			return fmt.Errorf("%s, %s", err, cerr)
		}
		return err
	}

	plog.Printf("%s defragmented...", tt.logPrefix())
	return nil
}

func (tt *tester) logPrefix() string {
	var (
		rd     = tt.status.getRound()
		cs     = tt.status.getCase()
		prefix = fmt.Sprintf("[round#%d case#%d]", rd, cs)
	)
	if cs == -1 {
		prefix = fmt.Sprintf("[round#%d]", rd)
	}
	return prefix
}

func (tt *tester) cleanup() error {
	roundFailedTotalCounter.Inc()
	desc := "compact/defrag"
	if tt.status.Case != -1 {
		desc = tt.failures[tt.status.Case].Desc()
	}
	caseFailedTotalCounter.WithLabelValues(desc).Inc()

	plog.Printf("%s cleaning up...", tt.logPrefix())
	if err := tt.cluster.Cleanup(); err != nil {
		plog.Warningf("%s cleanup error: %v", tt.logPrefix(), err)
		return err
	}

	if err := tt.cluster.Bootstrap(); err != nil {
		plog.Warningf("%s cleanup Bootstrap error: %v", tt.logPrefix(), err)
		return err
	}

	return nil
}

func (tt *tester) cancelStressers() {
	plog.Printf("%s canceling the stressers...", tt.logPrefix())
	for _, s := range tt.cluster.Stressers {
		s.Cancel()
	}
	plog.Printf("%s canceled stressers", tt.logPrefix())
}

func (tt *tester) startStressers() {
	plog.Printf("%s starting the stressers...", tt.logPrefix())
	for _, s := range tt.cluster.Stressers {
		go s.Stress()
	}
	plog.Printf("%s started stressers", tt.logPrefix())
}
