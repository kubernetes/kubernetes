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

package filterconfig

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"testing"
	"time"

	fcv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	fcfmt "k8s.io/apiserver/pkg/util/flowcontrol/format"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	fcclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1alpha1"
	"k8s.io/klog"
)

type flowSchemaDigests struct {
	matches    []RequestDigest
	mismatches []RequestDigest
}

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

var mandFSs = func() map[string]*fcv1a1.FlowSchema {
	ans := make(map[string]*fcv1a1.FlowSchema)
	for _, mand := range fcboot.MandatoryFlowSchemas {
		ans[mand.Name] = mand
	}
	return ans
}()

type ctlTestState struct {
	t           *testing.T
	cfgCtl      *configController
	fcIfc       fcclient.FlowcontrolV1alpha1Interface
	existingPLs map[string]*fcv1a1.PriorityLevelConfiguration
	existingFSs map[string]*fcv1a1.FlowSchema
	tracer      *fcv1a1.FlowSchema
	ftrs        map[string]*fsTestingRecord
	lock        sync.Mutex
	queues      map[string]*ctlTestQueueSet
}

var _ fq.QueueSetFactory = (*ctlTestState)(nil)

type ctlTestQueueSetCompleter struct {
	cts *ctlTestState
	qc  fq.QueuingConfig
}

type ctlTestQueueSet struct {
	cts *ctlTestState
	qc  fq.QueuingConfig
	dc  fq.DispatchingConfig
}

type ctlTestRequest struct {
	cts            *ctlTestState
	qsName         string
	descr1, descr2 interface{}
}

func (cts *ctlTestState) BeginConstruction(qc fq.QueuingConfig) (fq.QueueSetCompleter, error) {
	return ctlTestQueueSetCompleter{cts, qc}, nil
}

func (cqs *ctlTestQueueSet) BeginConfigChange(qc fq.QueuingConfig) (fq.QueueSetCompleter, error) {
	return ctlTestQueueSetCompleter{cqs.cts, qc}, nil
}

func (cqc ctlTestQueueSetCompleter) Complete(dc fq.DispatchingConfig) fq.QueueSet {
	cqc.cts.lock.Lock()
	defer cqc.cts.lock.Unlock()
	qs := &ctlTestQueueSet{cqc.cts, cqc.qc, dc}
	cqc.cts.queues[cqc.qc.Name] = qs
	return qs
}

func (cqs *ctlTestQueueSet) IsIdle() bool {
	return false
}

func (cqs *ctlTestQueueSet) StartRequest(ctx context.Context, hashValue uint64, descr1, descr2 interface{}) (req fq.Request, idle bool) {
	return &ctlTestRequest{cqs.cts, cqs.qc.Name, descr1, descr2}, false
}

func (ctr *ctlTestRequest) Wait() (execute, idle bool, afterExecution func() (idle bool)) {
	return true, false, func() (idle bool) { return false }
}

func TestDigestConfig(t *testing.T) {
	rngOuter := rand.New(rand.NewSource(1234567890123456789))
	for i := 1; i <= 10; i++ {
		rng := rand.New(rand.NewSource(int64(rngOuter.Uint64())))
		t.Run(fmt.Sprintf("trial%d:", i), func(t *testing.T) {
			clientset := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(clientset, 0)
			flowcontrolClient := clientset.FlowcontrolV1alpha1()
			ctl := NewTestableController(
				informerFactory,
				flowcontrolClient,
				100,         // server concurrency limit
				time.Minute, // request wait limit
				noRestraintQSF,
			).(*configController)
			cts := &ctlTestState{t: t,
				cfgCtl:      ctl,
				fcIfc:       flowcontrolClient,
				existingFSs: map[string]*fcv1a1.FlowSchema{},
				existingPLs: map[string]*fcv1a1.PriorityLevelConfiguration{},
			}
			stopCh := make(chan struct{})
			// informerFactory.Start(stopCh)
			go informerFactory.Flowcontrol().V1alpha1().FlowSchemas().Informer().Run(stopCh)
			go informerFactory.Flowcontrol().V1alpha1().PriorityLevelConfigurations().Informer().Run(stopCh)
			go ctl.Run(stopCh)
			oldPLStates := map[string]*priorityLevelState{}
			trialStep := fmt.Sprintf("trial%d-0", i)
			_, newGoodPLMap, newGoodPLNames, newBadPLNames := genPLs(rng, trialStep, sets.NewString(), 0)
			_, newFSMap, newGoodFSMap, newFSDigestses := genFSs(t, rng, trialStep, newGoodPLNames, newBadPLNames, 0)
			for j := 0; ; {
				t.Logf("For %s, newGoodPLNames=%#+v", trialStep, newGoodPLNames)
				t.Logf("For %s, newGoodFSMap=%#+v", trialStep, newGoodFSMap)
				// Check that the latest digestion did the right thing
				expectedPLNames := newGoodPLNames.Union(sets.StringKeySet(oldPLStates))
				expectedPLNames = expectedPLNames.Union(sets.StringKeySet(mandPLs))
				priorityLevelStates, fss := ctl.copyState()
				if e, a := expectedPLNames, sets.StringKeySet(priorityLevelStates); !e.Equal(a) {
					t.Errorf("Fail at %s: e=%v, a=%v", trialStep, e, a)
				}
				for plName := range expectedPLNames {
					plState := priorityLevelStates[plName]
					checkNewPLState(t, trialStep, plName, plState, oldPLStates, newGoodPLMap)
				}
				testFSNames := sets.StringKeySet(newGoodFSMap).Union(sets.StringKeySet(mandFSs))
				for _, fs := range fss {
					checkNewFS(t, ctl, trialStep, fs, newGoodFSMap, newFSDigestses[fs.Name])
					testFSNames.Delete(fs.Name)
				}
				if len(testFSNames) > 0 {
					t.Errorf("Fail at %s: ctl did not retain FlowSchemas %#+v", trialStep, testFSNames)
				}

				j++
				if j > 10 {
					break
				}

				// Calculate expected survivors
				nextPLStates := map[string]*priorityLevelState{}
				for oldPLName, oldPLState := range priorityLevelStates {
					if mandPLs[oldPLName] != nil || oldPLState.queues != nil && !(oldPLState.quiescing && oldPLState.queues.IsIdle()) {
						oldState := *oldPLState
						nextPLStates[oldPLName] = &oldState
					}
				}
				oldPLStates = nextPLStates

				// Now create a new config and digest it
				trialStep = fmt.Sprintf("trial%d-%d", i, j)
				var newPLs []*fcv1a1.PriorityLevelConfiguration
				var newFSs []*fcv1a1.FlowSchema
				newPLs, newGoodPLMap, newGoodPLNames, newBadPLNames = genPLs(rng, trialStep, expectedPLNames, 1+rng.Intn(4))
				newFSs, newFSMap, newGoodFSMap, newFSDigestses = genFSs(t, rng, trialStep, newGoodPLNames, newBadPLNames, 1+rng.Intn(6))

				if true {
					cts.establishAPIObjects(trialStep, int32(i*10+j), newGoodPLMap, newFSMap)
				} else {
					for _, newPL := range newPLs {
						t.Logf("For %s, digesting newPL=%#+v", trialStep, fcfmt.Fmt(newPL))
					}
					for _, newFS := range newFSs {
						t.Logf("For %s, digesting newFS=%#+v", trialStep, fcfmt.Fmt(newFS))
					}
					ctl.digestConfigObjects(newPLs, newFSs)
				}
			}
			close(stopCh)
		})
	}
}

func (cfgCtl *configController) copyState() (priorityLevelStates map[string]*priorityLevelState, fss []*fcv1a1.FlowSchema) {
	cfgCtl.lock.Lock()
	defer cfgCtl.lock.Unlock()
	fss = append(fss, cfgCtl.flowSchemas...)
	priorityLevelStates = make(map[string]*priorityLevelState, len(cfgCtl.priorityLevelStates))
	for k, v := range cfgCtl.priorityLevelStates {
		priorityLevelStates[k] = v
	}
	klog.V(7).Infof("Copied state pls=%#+v, fss=%#+v", priorityLevelStates, fss)
	return
}

func (cts *ctlTestState) establishAPIObjects(trialStep string, traceValue int32, newPLs map[string]*fcv1a1.PriorityLevelConfiguration, newFSs map[string]*fcv1a1.FlowSchema) {
	plIfc := cts.fcIfc.PriorityLevelConfigurations()
	for plName, _ := range cts.existingPLs {
		if newPLs[plName] == nil {
			err := plIfc.Delete(plName, nil)
			if err == nil {
				cts.t.Logf("%s: deleted undesired PriorityLevelConfiguration %s", trialStep, plName)
			} else {
				cts.t.Errorf("%s: failed to delete undesired PriorityLevelConfiguration %s: %s", trialStep, plName, err.Error())
			}
		}
	}
	newExistingPLs := map[string]*fcv1a1.PriorityLevelConfiguration{}
	for plName, pl := range newPLs {
		oldPL := cts.existingPLs[plName]
		if oldPL != nil {
			modPL := oldPL.DeepCopy()
			modPL.Spec = pl.Spec
			retPL, err := plIfc.Update(modPL)
			if err == nil {
				cts.t.Logf("%s: updated PriorityLevelConfiguration %#+v", trialStep, fcfmt.Fmt(modPL))
			} else {
				cts.t.Errorf("%s: failed to update PriorityLevelConfiguration %#+v: %s", trialStep, fcfmt.Fmt(modPL), err.Error())
			}
			newExistingPLs[plName] = retPL
		} else {
			retPL, err := plIfc.Create(pl)
			if err == nil {
				cts.t.Logf("%s: created PriorityLevelConfiguration %#+v", trialStep, fcfmt.Fmt(pl))
			} else {
				cts.t.Errorf("%s: failed to create PriorityLevelConfiguration %#+v: %s", trialStep, fcfmt.Fmt(pl), err.Error())
			}
			newExistingPLs[plName] = retPL
		}
	}
	cts.existingPLs = newExistingPLs

	fsIfc := cts.fcIfc.FlowSchemas()
	for fsName, _ := range cts.existingFSs {
		if newFSs[fsName] == nil {
			err := fsIfc.Delete(fsName, nil)
			if err == nil {
				cts.t.Logf("%s: deleted undesired FlowSchema %s", trialStep, fsName)
			} else {
				cts.t.Errorf("%s: failed to delete undesired FlowSchema %s: %s", trialStep, fsName, err.Error())
			}
		}
	}
	newExistingFSs := map[string]*fcv1a1.FlowSchema{}
	for fsName, fs := range newFSs {
		oldFS := cts.existingFSs[fsName]
		if oldFS != nil {
			modFS := oldFS.DeepCopy()
			modFS.Spec = fs.Spec
			retFS, err := fsIfc.Update(modFS)
			if err == nil {
				cts.t.Logf("%s: updated FlowSchema %#+v", trialStep, fcfmt.Fmt(modFS))
			} else {
				cts.t.Errorf("%s: failed to update FlowSchema %#+v: %s", trialStep, fcfmt.Fmt(modFS), err.Error())
			}
			newExistingFSs[fsName] = retFS
		} else {
			retFS, err := fsIfc.Create(fs)
			if err == nil {
				cts.t.Logf("%s: created FlowSchema %#+v", trialStep, fcfmt.Fmt(fs))
			} else {
				cts.t.Errorf("%s: failed to create FlowSchema %#+v: %s", trialStep, fcfmt.Fmt(fs), err.Error())
			}
			newExistingFSs[fsName] = retFS
		}
	}
	cts.existingFSs = newExistingFSs

	// Now we have to wait for all those changes to be processed.
	time.Sleep(500 * time.Millisecond)
	// Wait for the controller to think it has nothing to do
	wait.PollImmediate(250*time.Millisecond, 2*time.Second, func() (bool, error) {
		return !cts.cfgCtl.hasWork(), nil
	})
	// Fire a tracer
	var err error
	if cts.tracer == nil {
		cts.tracer, err = fsIfc.Create(&fcv1a1.FlowSchema{
			ObjectMeta: metav1.ObjectMeta{Name: tracerName},
			Spec: fcv1a1.FlowSchemaSpec{
				PriorityLevelConfiguration: fcv1a1.PriorityLevelConfigurationReference{"exempt"},
				MatchingPrecedence:         traceValue},
			Status: fcv1a1.FlowSchemaStatus{
				Conditions: []fcv1a1.FlowSchemaCondition{{
					Type:   fcv1a1.FlowSchemaConditionDangling,
					Status: fcv1a1.ConditionFalse}},
			}})
	} else {
		t2 := cts.tracer.DeepCopy()
		t2.Spec.MatchingPrecedence = traceValue
		cts.tracer, err = fsIfc.Update(t2)
	}
	if err != nil {
		panic(err)
	}
	cts.t.Logf("Created tracer %#+v", fcfmt.Fmt(cts.tracer))
	if false {
		return
	}
	// Wait for it to be processed
	cts.cfgCtl.waitForTracedValue(traceValue)
}

func checkNewPLState(t *testing.T, trialName, plName string, plState *priorityLevelState, oldPLStates map[string]*priorityLevelState, newGoodPLMap map[string]*fcv1a1.PriorityLevelConfiguration) {
	var expectedSpec *fcv1a1.PriorityLevelConfigurationSpec
	pl, inNew := newGoodPLMap[plName]
	if inNew {
		expectedSpec = &pl.Spec
	}
	ost, inOld := oldPLStates[plName]
	if expectedSpec == nil && inOld {
		expectedSpec = &ost.config
	}
	var isMand bool
	for _, mpl := range fcboot.MandatoryPriorityLevelConfigurations {
		if plName == mpl.Name {
			isMand = true
			if expectedSpec == nil {
				expectedSpec = &mpl.Spec
			}
		}
	}
	if expectedSpec == nil {
		t.Errorf("Fail at %s/%s: Inexplicable entry %#+v", trialName, plName, plState)
		return
	}
	if plState == nil {
		t.Errorf("Fail at %s/%s: missing new priorityLevelState", trialName, plName)
		return
	}
	if e, a := *expectedSpec, plState.config; !apiequality.Semantic.DeepEqual(e, a) {
		t.Errorf("Fail at %s/%s: e=%#+v, a=%#+v", trialName, plName, fcfmt.Fmt(e), fcfmt.Fmt(a))
	}
	isExempt := expectedSpec.Type == fcv1a1.PriorityLevelEnablementExempt
	if e, a := isExempt, (plState.qsCompleter == nil); e != a {
		t.Errorf("Fail at %s/%s: e=%v, a=%v", trialName, plName, e, a)
	}
	if e, a := isExempt, (plState.queues == nil); e != a {
		t.Errorf("Fail at %s/%s: e=%v, a=%v", trialName, plName, e, a)
	}
	if inOld {
		if e, a := ost.queues, plState.queues; e != a {
			t.Errorf("Fail at %s/%s: e=%p, a=%p", trialName, plName, e, a)
		}
	}
	if e, a := !(inNew || isMand), plState.quiescing; e != a {
		t.Errorf("Fail at %s/%s: e=%v, a=%v", trialName, plName, e, a)
	}
	if e, a := 0, plState.numPending; e != a {
		t.Errorf("Fail at %s/%s: e=%v, a=%v", trialName, plName, e, a)
	}
}

func checkNewFS(t *testing.T, ctl *configController, trialName string, fs *fcv1a1.FlowSchema, newGoodFSMap map[string]*fcv1a1.FlowSchema, digests *flowSchemaDigests) {
	if fs.Name == tracerName {
		return
	}
	orig := newGoodFSMap[fs.Name]
	for _, mfs := range fcboot.MandatoryFlowSchemas {
		if fs.Name == mfs.Name {
			if orig == nil {
				orig = mfs
			}
		}
	}
	if orig == nil {
		t.Errorf("Fail at %s: Inexplicable entry %#+v", trialName, *fs)
		return
	}
	if e, a := orig.Spec, fs.Spec; !apiequality.Semantic.DeepEqual(e, a) {
		t.Errorf("Fail at %s/%s: e=%#+v, a=%#+v", trialName, fs.Name, fcfmt.Fmt(e), fcfmt.Fmt(a))
	}
	if digests != nil {
		for _, rd := range digests.matches {
			matchFSName, matchDistMethod, matchPLName, startFn := ctl.Match(rd)
			t.Logf("Considering FlowSchema %s, Match(%#+v) => %q, %#+v, %q, %v", fs.Name, rd, matchFSName, matchDistMethod, matchPLName, startFn)
			if matchFSName == orig.Name {
				if e, a := orig.Spec.DistinguisherMethod, matchDistMethod; !apiequality.Semantic.DeepEqual(e, a) {
					t.Errorf("Fail at %s/%s: Got DistinguisherMethod %#+v for %#+v", trialName, fs.Name, a, fcfmt.Fmt(orig))
				}
				if e, a := orig.Spec.PriorityLevelConfiguration.Name, matchPLName; e != a {
					t.Errorf("Fail at %s/%s: e=%v, a=%v", trialName, fs.Name, e, a)
				}
			} else if matchFSName == fcv1a1.FlowSchemaNameCatchAll && fs.Spec.MatchingPrecedence < fcv1a1.FlowSchemaMaxMatchingPrecedence {
				t.Errorf("Fail at %s/%s: Failed expected match for %#+v against %#+v", trialName, fs.Name, rd, fcfmt.Fmt(orig))
			}
			if startFn != nil {
				exec, after := startFn(context.Background(), 47)
				t.Logf("startFn(..) => %v, %p", exec, after)
				if exec {
					after()
					t.Log("called after()")
				}
			}
		}
		for _, rd := range digests.mismatches {
			matchFSName, _, _, startFn := ctl.Match(rd)
			if matchFSName == orig.Name {
				t.Errorf("Fail at %s/%s: Digest %#+v unexpectedly matched schema %#+v", trialName, fs.Name, rd, fcfmt.Fmt(fs))
			}
			if startFn != nil {
				exec, after := startFn(context.Background(), 74)
				t.Logf("startFn(..) => %v, %p", exec, after)
				if exec {
					after()
					t.Log("called after()")
				}
			}
		}
	}
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
		if n == 0 || rng.Float32() < 0.5 && !(goodNames.Has(pl.Name) || badNames.Has(pl.Name)) {
			addGood(pl)
		}
	}
	return
}

func genFSs(t *testing.T, rng *rand.Rand, trial string, goodPLNames, badPLNames sets.String, n int) (newFSs []*fcv1a1.FlowSchema, newFSMap, newGoodFSMap map[string]*fcv1a1.FlowSchema, newFSDigestses map[string]*flowSchemaDigests) {
	newFSMap = map[string]*fcv1a1.FlowSchema{}
	newGoodFSMap = map[string]*fcv1a1.FlowSchema{}
	newFSDigestses = map[string]*flowSchemaDigests{}
	add := func(fs *fcv1a1.FlowSchema) {
		newFSs = append(newFSs, fs)
		newFSMap[fs.Name] = fs
	}
	addGood := func(fs *fcv1a1.FlowSchema, matches, mismatches []RequestDigest) {
		add(fs)
		newGoodFSMap[fs.Name] = fs
		newFSDigestses[fs.Name] = &flowSchemaDigests{matches: matches, mismatches: mismatches}
		t.Logf("For trial %s, adding good FlowSchema %#+v", trial, fcfmt.Fmt(fs))
	}
	if n == 0 || rng.Float32() < 0.5 {
		addGood(fcboot.MandatoryFlowSchemaCatchAll, nil, nil)
	}
	for i := 1; i <= n; i++ {
		ftr, wellFormed := genFS(t, rng, fmt.Sprintf("%s-fs%d", trial, i), false, goodPLNames, badPLNames)
		if wellFormed {
			addGood(ftr.fs, append(ftr.matchingRDigests, ftr.matchingNDigests...), append(ftr.skippingRDigests, ftr.skippingNDigests...))
		} else {
			add(ftr.fs)
			t.Logf("For trial %s, adding bad FlowSchema %#+v", trial, fcfmt.Fmt(ftr.fs))
		}
	}
	if n == 0 || rng.Float32() < 0.5 {
		addGood(fcboot.MandatoryFlowSchemaExempt, nil, nil)
	}
	return
}
