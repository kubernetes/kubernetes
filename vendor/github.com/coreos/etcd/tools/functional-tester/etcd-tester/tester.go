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
	cluster *cluster
	limit   int

	failures        []failure
	status          Status
	currentRevision int64

	stresserType string
	scfg         stressConfig
	doChecks     bool

	stresser Stresser
	checker  Checker
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

	if err := tt.resetStressCheck(); err != nil {
		plog.Errorf("%s failed to start stresser (%v)", tt.logPrefix(), err)
		return
	}

	var preModifiedKey int64
	for round := 0; round < tt.limit || tt.limit == -1; round++ {
		tt.status.setRound(round)
		roundTotalCounter.Inc()

		if err := tt.doRound(round); err != nil {
			plog.Warningf("%s functional-tester returning with error (%v)", tt.logPrefix(), err)
			if tt.cleanup() != nil {
				return
			}
			// reset preModifiedKey after clean up
			preModifiedKey = 0
			continue
		}
		// -1 so that logPrefix doesn't print out 'case'
		tt.status.setCase(-1)

		revToCompact := max(0, tt.currentRevision-10000)
		currentModifiedKey := tt.stresser.ModifiedKeys()
		modifiedKey := currentModifiedKey - preModifiedKey
		preModifiedKey = currentModifiedKey
		timeout := 10 * time.Second
		timeout += time.Duration(modifiedKey/compactQPS) * time.Second
		plog.Infof("%s compacting %d modifications (timeout %v)", tt.logPrefix(), modifiedKey, timeout)
		if err := tt.compact(revToCompact, timeout); err != nil {
			plog.Warningf("%s functional-tester compact got error (%v)", tt.logPrefix(), err)
			if tt.cleanup() != nil {
				return
			}
			// reset preModifiedKey after clean up
			preModifiedKey = 0
		}
		if round > 0 && round%500 == 0 { // every 500 rounds
			if err := tt.defrag(); err != nil {
				plog.Warningf("%s functional-tester returning with error (%v)", tt.logPrefix(), err)
				return
			}
		}
	}

	plog.Infof("%s functional-tester is finished", tt.logPrefix())
}

func (tt *tester) doRound(round int) error {
	for j, f := range tt.failures {
		caseTotalCounter.WithLabelValues(f.Desc()).Inc()
		tt.status.setCase(j)

		if err := tt.cluster.WaitHealth(); err != nil {
			return fmt.Errorf("wait full health error: %v", err)
		}
		plog.Infof("%s injecting failure %q", tt.logPrefix(), f.Desc())
		if err := f.Inject(tt.cluster, round); err != nil {
			return fmt.Errorf("injection error: %v", err)
		}
		plog.Infof("%s injected failure", tt.logPrefix())

		plog.Infof("%s recovering failure %q", tt.logPrefix(), f.Desc())
		if err := f.Recover(tt.cluster, round); err != nil {
			return fmt.Errorf("recovery error: %v", err)
		}
		plog.Infof("%s recovered failure", tt.logPrefix())
		tt.cancelStresser()
		plog.Infof("%s wait until cluster is healthy", tt.logPrefix())
		if err := tt.cluster.WaitHealth(); err != nil {
			return fmt.Errorf("wait full health error: %v", err)
		}
		plog.Infof("%s cluster is healthy", tt.logPrefix())

		plog.Infof("%s checking consistency and invariant of cluster", tt.logPrefix())
		if err := tt.checkConsistency(); err != nil {
			return fmt.Errorf("tt.checkConsistency error (%v)", err)
		}
		plog.Infof("%s checking consistency and invariant of cluster done", tt.logPrefix())

		plog.Infof("%s succeed!", tt.logPrefix())
	}
	return nil
}

func (tt *tester) updateRevision() error {
	revs, _, err := tt.cluster.getRevisionHash()
	for _, rev := range revs {
		tt.currentRevision = rev
		break // just need get one of the current revisions
	}

	plog.Infof("%s updated current revision to %d", tt.logPrefix(), tt.currentRevision)
	return err
}

func (tt *tester) checkConsistency() (err error) {
	defer func() {
		if err != nil {
			return
		}
		if err = tt.updateRevision(); err != nil {
			plog.Warningf("%s functional-tester returning with tt.updateRevision error (%v)", tt.logPrefix(), err)
			return
		}
		err = tt.startStresser()
	}()
	if err = tt.checker.Check(); err != nil {
		plog.Infof("%s %v", tt.logPrefix(), err)
	}
	return err
}

func (tt *tester) compact(rev int64, timeout time.Duration) (err error) {
	tt.cancelStresser()
	defer func() {
		if err == nil {
			err = tt.startStresser()
		}
	}()

	plog.Infof("%s compacting storage (current revision %d, compact revision %d)", tt.logPrefix(), tt.currentRevision, rev)
	if err = tt.cluster.compactKV(rev, timeout); err != nil {
		return err
	}
	plog.Infof("%s compacted storage (compact revision %d)", tt.logPrefix(), rev)

	plog.Infof("%s checking compaction (compact revision %d)", tt.logPrefix(), rev)
	if err = tt.cluster.checkCompact(rev); err != nil {
		plog.Warningf("%s checkCompact error (%v)", tt.logPrefix(), err)
		return err
	}

	plog.Infof("%s confirmed compaction (compact revision %d)", tt.logPrefix(), rev)
	return nil
}

func (tt *tester) defrag() error {
	plog.Infof("%s defragmenting...", tt.logPrefix())
	if err := tt.cluster.defrag(); err != nil {
		plog.Warningf("%s defrag error (%v)", tt.logPrefix(), err)
		if cerr := tt.cleanup(); cerr != nil {
			return fmt.Errorf("%s, %s", err, cerr)
		}
		return err
	}
	plog.Infof("%s defragmented...", tt.logPrefix())
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

	tt.cancelStresser()
	if err := tt.cluster.Cleanup(); err != nil {
		plog.Warningf("%s cleanup error: %v", tt.logPrefix(), err)
		return err
	}
	if err := tt.cluster.Reset(); err != nil {
		plog.Warningf("%s cleanup Bootstrap error: %v", tt.logPrefix(), err)
		return err
	}
	return tt.resetStressCheck()
}

func (tt *tester) cancelStresser() {
	plog.Infof("%s canceling the stressers...", tt.logPrefix())
	tt.stresser.Cancel()
	plog.Infof("%s canceled stressers", tt.logPrefix())
}

func (tt *tester) startStresser() (err error) {
	plog.Infof("%s starting the stressers...", tt.logPrefix())
	err = tt.stresser.Stress()
	plog.Infof("%s started stressers", tt.logPrefix())
	return err
}

func (tt *tester) resetStressCheck() error {
	plog.Infof("%s resetting stressers and checkers...", tt.logPrefix())
	cs := &compositeStresser{}
	for _, m := range tt.cluster.Members {
		s := NewStresser(tt.stresserType, &tt.scfg, m)
		cs.stressers = append(cs.stressers, s)
	}
	tt.stresser = cs
	if !tt.doChecks {
		tt.checker = newNoChecker()
		return tt.startStresser()
	}
	chk := newHashChecker(hashAndRevGetter(tt.cluster))
	if schk := cs.Checker(); schk != nil {
		chk = newCompositeChecker([]Checker{chk, schk})
	}
	tt.checker = chk
	return tt.startStresser()
}

func (tt *tester) Report() int64 { return tt.stresser.ModifiedKeys() }
