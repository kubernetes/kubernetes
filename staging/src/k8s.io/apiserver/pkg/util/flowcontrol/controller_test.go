/*
Copyright 2019 The Kubernetes Authors.

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

package flowcontrol

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"testing"
	"time"

	fcv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/apimachinery/pkg/util/sets"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	fcfmt "k8s.io/apiserver/pkg/util/flowcontrol/format"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	fcclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1alpha1"
	"k8s.io/klog/v2"
)

func TestMain(m *testing.M) {
	klog.InitFlags(nil)
	os.Exit(m.Run())
}

var mandPLs = func() map[string]*fcv1a1.PriorityLevelConfiguration {
	ans := make(map[string]*fcv1a1.PriorityLevelConfiguration)
	for _, mand := range fcboot.MandatoryPriorityLevelConfigurations {
		ans[mand.Name] = mand
	}
	return ans
}()

type ctlTestState struct {
	t               *testing.T
	cfgCtl          *configController
	fcIfc           fcclient.FlowcontrolV1alpha1Interface
	existingPLs     map[string]*fcv1a1.PriorityLevelConfiguration
	existingFSs     map[string]*fcv1a1.FlowSchema
	heldRequestsMap map[string][]heldRequest
	requestWG       sync.WaitGroup
	lock            sync.Mutex
	queues          map[string]*ctlTestQueueSet
}

type heldRequest struct {
	rd       RequestDigest
	finishCh chan struct{}
}

var _ fq.QueueSetFactory = (*ctlTestState)(nil)

type ctlTestQueueSetCompleter struct {
	cts *ctlTestState
	cqs *ctlTestQueueSet
	qc  fq.QueuingConfig
}

type ctlTestQueueSet struct {
	cts         *ctlTestState
	qc          fq.QueuingConfig
	dc          fq.DispatchingConfig
	countActive int
}

type ctlTestRequest struct {
	cqs            *ctlTestQueueSet
	qsName         string
	descr1, descr2 interface{}
}

func (cts *ctlTestState) BeginConstruction(qc fq.QueuingConfig) (fq.QueueSetCompleter, error) {
	return ctlTestQueueSetCompleter{cts, nil, qc}, nil
}

func (cqs *ctlTestQueueSet) BeginConfigChange(qc fq.QueuingConfig) (fq.QueueSetCompleter, error) {
	return ctlTestQueueSetCompleter{cqs.cts, cqs, qc}, nil
}

func (cqc ctlTestQueueSetCompleter) Complete(dc fq.DispatchingConfig) fq.QueueSet {
	cqc.cts.lock.Lock()
	defer cqc.cts.lock.Unlock()
	qs := cqc.cqs
	if qs == nil {
		qs = &ctlTestQueueSet{cts: cqc.cts, qc: cqc.qc, dc: dc}
		cqc.cts.queues[cqc.qc.Name] = qs
	} else {
		qs.qc, qs.dc = cqc.qc, dc
	}
	return qs
}

func (cqs *ctlTestQueueSet) IsIdle() bool {
	cqs.cts.lock.Lock()
	defer cqs.cts.lock.Unlock()
	klog.V(7).Infof("For %p QS %s, countActive==%d", cqs, cqs.qc.Name, cqs.countActive)
	return cqs.countActive == 0
}

func (cqs *ctlTestQueueSet) StartRequest(ctx context.Context, hashValue uint64, fsName string, descr1, descr2 interface{}) (req fq.Request, idle bool) {
	cqs.cts.lock.Lock()
	defer cqs.cts.lock.Unlock()
	cqs.countActive++
	cqs.cts.t.Logf("Queued %q %#+v %#+v for %p QS=%s, countActive:=%d", fsName, descr1, descr2, cqs, cqs.qc.Name, cqs.countActive)
	return &ctlTestRequest{cqs, cqs.qc.Name, descr1, descr2}, false
}

func (ctr *ctlTestRequest) Finish(execute func()) bool {
	execute()
	ctr.cqs.cts.lock.Lock()
	defer ctr.cqs.cts.lock.Unlock()
	ctr.cqs.countActive--
	ctr.cqs.cts.t.Logf("Finished %#+v %#+v for %p QS=%s, countActive:=%d", ctr.descr1, ctr.descr2, ctr.cqs, ctr.cqs.qc.Name, ctr.cqs.countActive)
	return ctr.cqs.countActive == 0
}

func (cts *ctlTestState) getQueueSetNames() sets.String {
	cts.lock.Lock()
	defer cts.lock.Unlock()
	return sets.StringKeySet(cts.queues)
}

func (cts *ctlTestState) getNonIdleQueueSetNames() sets.String {
	cts.lock.Lock()
	defer cts.lock.Unlock()
	ans := sets.NewString()
	for name, qs := range cts.queues {
		if qs.countActive > 0 {
			ans.Insert(name)
		}
	}
	return ans
}

func (cts *ctlTestState) hasNonIdleQueueSet(name string) bool {
	cts.lock.Lock()
	defer cts.lock.Unlock()
	qs := cts.queues[name]
	return qs != nil && qs.countActive > 0
}

func (cts *ctlTestState) addHeldRequest(plName string, rd RequestDigest, finishCh chan struct{}) {
	cts.lock.Lock()
	defer cts.lock.Unlock()
	hrs := cts.heldRequestsMap[plName]
	hrs = append(hrs, heldRequest{rd, finishCh})
	cts.heldRequestsMap[plName] = hrs
	cts.t.Logf("Holding %#+v for %s, count:=%d", rd, plName, len(hrs))
}

func (cts *ctlTestState) popHeldRequest() (plName string, hr *heldRequest, nCount int) {
	cts.lock.Lock()
	defer cts.lock.Unlock()
	var hrs []heldRequest
	for {
		for plName, hrs = range cts.heldRequestsMap {
			goto GotOne
		}
		return "", nil, 0
	GotOne:
		if nhr := len(hrs); nhr > 0 {
			hrv := hrs[nhr-1]
			hrs = hrs[:nhr-1]
			hr = &hrv
		}
		if len(hrs) == 0 {
			delete(cts.heldRequestsMap, plName)
		} else {
			cts.heldRequestsMap[plName] = hrs
		}
		if hr != nil {
			nCount = len(hrs)
			return
		}
	}
}

var mandQueueSetNames, exclQueueSetNames = func() (sets.String, sets.String) {
	mandQueueSetNames := sets.NewString()
	exclQueueSetNames := sets.NewString()
	for _, mpl := range fcboot.MandatoryPriorityLevelConfigurations {
		if mpl.Spec.Type == fcv1a1.PriorityLevelEnablementExempt {
			exclQueueSetNames.Insert(mpl.Name)
		} else {
			mandQueueSetNames.Insert(mpl.Name)
		}
	}
	return mandQueueSetNames, exclQueueSetNames
}()

func TestConfigConsumer(t *testing.T) {
	rngOuter := rand.New(rand.NewSource(1234567890123456789))
	for i := 1; i <= 20; i++ {
		rng := rand.New(rand.NewSource(int64(rngOuter.Uint64())))
		t.Run(fmt.Sprintf("trial%d:", i), func(t *testing.T) {
			clientset := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(clientset, 0)
			flowcontrolClient := clientset.FlowcontrolV1alpha1()
			cts := &ctlTestState{t: t,
				fcIfc:           flowcontrolClient,
				existingFSs:     map[string]*fcv1a1.FlowSchema{},
				existingPLs:     map[string]*fcv1a1.PriorityLevelConfiguration{},
				heldRequestsMap: map[string][]heldRequest{},
				queues:          map[string]*ctlTestQueueSet{},
			}
			ctl := newTestableController(
				informerFactory,
				flowcontrolClient,
				100,         // server concurrency limit
				time.Minute, // request wait limit
				cts,
			)
			cts.cfgCtl = ctl
			persistingPLNames := sets.NewString()
			trialStep := fmt.Sprintf("trial%d-0", i)
			_, _, desiredPLNames, newBadPLNames := genPLs(rng, trialStep, persistingPLNames, 0)
			_, _, newFTRs, newCatchAlls := genFSs(t, rng, trialStep, desiredPLNames, newBadPLNames, 0)
			for j := 0; ; {
				t.Logf("For %s, desiredPLNames=%#+v", trialStep, desiredPLNames)
				t.Logf("For %s, newFTRs=%#+v", trialStep, newFTRs)
				// Check that the latest digestion did the right thing
				nextPLNames := sets.NewString()
				for oldPLName := range persistingPLNames {
					if mandPLs[oldPLName] != nil || cts.hasNonIdleQueueSet(oldPLName) {
						nextPLNames.Insert(oldPLName)
					}
				}
				persistingPLNames = nextPLNames.Union(desiredPLNames)
				expectedQueueSetNames := persistingPLNames.Union(mandQueueSetNames).Difference(exclQueueSetNames)
				allQueueSetNames := cts.getQueueSetNames()
				missingQueueSetNames := expectedQueueSetNames.Difference(allQueueSetNames)
				if len(missingQueueSetNames) > 0 {
					t.Errorf("Fail: missing QueueSets %v", missingQueueSetNames)
				}
				nonIdleQueueSetNames := cts.getNonIdleQueueSetNames()
				extraQueueSetNames := nonIdleQueueSetNames.Difference(expectedQueueSetNames)
				if len(extraQueueSetNames) > 0 {
					t.Errorf("Fail: unexpected QueueSets %v", extraQueueSetNames)
				}
				for plName, hr, nCount := cts.popHeldRequest(); hr != nil; plName, hr, nCount = cts.popHeldRequest() {
					desired := desiredPLNames.Has(plName) || mandPLs[plName] != nil
					t.Logf("Releasing held request %#+v, desired=%v, plName=%s, count:=%d", hr.rd, desired, plName, nCount)
					close(hr.finishCh)
				}
				cts.requestWG.Wait()
				for _, ftr := range newFTRs {
					checkNewFS(cts, rng, trialStep, ftr, newCatchAlls)
				}

				j++
				if j > 20 {
					break
				}

				// Calculate expected survivors

				// Now create a new config and digest it
				trialStep = fmt.Sprintf("trial%d-%d", i, j)
				var newPLs []*fcv1a1.PriorityLevelConfiguration
				var newFSs []*fcv1a1.FlowSchema
				newPLs, _, desiredPLNames, newBadPLNames = genPLs(rng, trialStep, persistingPLNames, 1+rng.Intn(4))
				newFSs, _, newFTRs, newCatchAlls = genFSs(t, rng, trialStep, desiredPLNames, newBadPLNames, 1+rng.Intn(6))

				for _, newPL := range newPLs {
					t.Logf("For %s, digesting newPL=%s", trialStep, fcfmt.Fmt(newPL))
				}
				for _, newFS := range newFSs {
					t.Logf("For %s, digesting newFS=%s", trialStep, fcfmt.Fmt(newFS))
				}
				_ = ctl.lockAndDigestConfigObjects(newPLs, newFSs)
			}
			for plName, hr, nCount := cts.popHeldRequest(); hr != nil; plName, hr, nCount = cts.popHeldRequest() {
				desired := desiredPLNames.Has(plName) || mandPLs[plName] != nil
				t.Logf("Releasing held request %#+v, desired=%v, plName=%s, count:=%d", hr.rd, desired, plName, nCount)
				close(hr.finishCh)
			}
			cts.requestWG.Wait()
		})
	}
}

func checkNewFS(cts *ctlTestState, rng *rand.Rand, trialName string, ftr *fsTestingRecord, catchAlls map[bool]*fcv1a1.FlowSchema) {
	t := cts.t
	ctl := cts.cfgCtl
	fs := ftr.fs
	expectedPLName := fs.Spec.PriorityLevelConfiguration.Name
	ctx := context.Background()
	// Use this to make sure all these requests have started executing
	// before the next reconfiguration
	var startWG sync.WaitGroup
	for matches, digests1 := range ftr.digests {
		for isResource, digests2 := range digests1 {
			for _, rd := range digests2 {
				finishCh := make(chan struct{})
				rdu := uniqify(rd)
				cts.requestWG.Add(1)
				startWG.Add(1)
				go func(matches, isResource bool, rdu RequestDigest) {
					expectedMatch := matches && ftr.wellFormed && (fsPrecedes(fs, catchAlls[isResource]) || fs.Name == catchAlls[isResource].Name)
					ctl.Handle(ctx, rdu, func(matchFS *fcv1a1.FlowSchema, matchPL *fcv1a1.PriorityLevelConfiguration) {
						matchIsExempt := matchPL.Spec.Type == fcv1a1.PriorityLevelEnablementExempt
						t.Logf("Considering FlowSchema %s, expectedMatch=%v, isResource=%v: Handle(%#+v) => note(fs=%s, pl=%s, isExempt=%v)", fs.Name, expectedMatch, isResource, rdu, matchFS.Name, matchPL.Name, matchIsExempt)
						if e, a := expectedMatch, matchFS.Name == fs.Name; e != a {
							t.Errorf("Fail at %s/%s: rd=%#+v, expectedMatch=%v, actualMatch=%v, matchFSName=%q, catchAlls=%#+v", trialName, fs.Name, rdu, e, a, matchFS.Name, catchAlls)
						}
						if matchFS.Name == fs.Name {
							if e, a := fs.Spec.PriorityLevelConfiguration.Name, matchPL.Name; e != a {
								t.Errorf("Fail at %s/%s: e=%v, a=%v", trialName, fs.Name, e, a)
							}
						}
					}, func() {
						startWG.Done()
						_ = <-finishCh
					})
					cts.requestWG.Done()
				}(matches, isResource, rdu)
				if rng.Float32() < 0.8 {
					t.Logf("Immediate request %#+v, plName=%s", rdu, expectedPLName)
					close(finishCh)
				} else {
					cts.addHeldRequest(expectedPLName, rdu, finishCh)
				}
			}
		}
	}
	startWG.Wait()
}

func genPLs(rng *rand.Rand, trial string, oldPLNames sets.String, n int) (pls []*fcv1a1.PriorityLevelConfiguration, plMap map[string]*fcv1a1.PriorityLevelConfiguration, goodNames, badNames sets.String) {
	pls = make([]*fcv1a1.PriorityLevelConfiguration, 0, n)
	plMap = make(map[string]*fcv1a1.PriorityLevelConfiguration, n)
	goodNames = sets.NewString()
	badNames = sets.NewString(trial+"-nopl1", trial+"-nopl2")
	addGood := func(pl *fcv1a1.PriorityLevelConfiguration) {
		pls = append(pls, pl)
		plMap[pl.Name] = pl
		goodNames.Insert(pl.Name)
	}
	for i := 1; i <= n; i++ {
		pl := genPL(rng, fmt.Sprintf("%s-pl%d", trial, i))
		addGood(pl)
	}
	for oldPLName := range oldPLNames {
		if _, has := mandPLs[oldPLName]; has {
			continue
		}
		if rng.Float32() < 0.67 {
			pl := genPL(rng, oldPLName)
			addGood(pl)
		}
	}
	for _, pl := range mandPLs {
		if n > 0 && rng.Float32() < 0.5 && !(goodNames.Has(pl.Name) || badNames.Has(pl.Name)) {
			addGood(pl)
		}
	}
	return
}

func genFSs(t *testing.T, rng *rand.Rand, trial string, goodPLNames, badPLNames sets.String, n int) (newFSs []*fcv1a1.FlowSchema, newFSMap map[string]*fcv1a1.FlowSchema, newFTRs map[string]*fsTestingRecord, catchAlls map[bool]*fcv1a1.FlowSchema) {
	newFTRs = map[string]*fsTestingRecord{}
	catchAlls = map[bool]*fcv1a1.FlowSchema{
		false: fcboot.MandatoryFlowSchemaCatchAll,
		true:  fcboot.MandatoryFlowSchemaCatchAll}
	newFSMap = map[string]*fcv1a1.FlowSchema{}
	add := func(ftr *fsTestingRecord) {
		newFSs = append(newFSs, ftr.fs)
		newFSMap[ftr.fs.Name] = ftr.fs
		newFTRs[ftr.fs.Name] = ftr
		if ftr.wellFormed {
			if ftr.matchesAllNonResourceRequests && fsPrecedes(ftr.fs, catchAlls[false]) {
				catchAlls[false] = ftr.fs
			}
			if ftr.matchesAllResourceRequests && fsPrecedes(ftr.fs, catchAlls[true]) {
				catchAlls[true] = ftr.fs
			}
		}
		t.Logf("For trial %s, adding wf=%v FlowSchema %s", trial, ftr.wellFormed, fcfmt.Fmt(ftr.fs))
	}
	if n == 0 || rng.Float32() < 0.5 {
		add(mandFTRCatchAll)
	}
	for i := 1; i <= n; i++ {
		ftr := genFS(t, rng, fmt.Sprintf("%s-fs%d", trial, i), false, goodPLNames, badPLNames)
		add(ftr)
	}
	if n == 0 || rng.Float32() < 0.5 {
		add(mandFTRExempt)
	}
	return
}

func fsPrecedes(a, b *fcv1a1.FlowSchema) bool {
	if a.Spec.MatchingPrecedence < b.Spec.MatchingPrecedence {
		return true
	}
	if a.Spec.MatchingPrecedence == b.Spec.MatchingPrecedence {
		return a.Name < b.Name
	}
	return false
}
