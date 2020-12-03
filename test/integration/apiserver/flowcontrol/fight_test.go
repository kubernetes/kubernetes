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
	"sync"
	"testing"
	"time"

	flowcontrol "k8s.io/api/flowcontrol/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/wait"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfc "k8s.io/apiserver/pkg/util/flowcontrol"
	fqtesting "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
)

const timeFmt = "2006-01-02T15:04:05.999"

type fightTest struct {
	t              *testing.T
	ctx            context.Context
	loopbackConfig *rest.Config
	size           int
	stopCh         chan struct{}
	now            time.Time
	clk            *clock.FakeClock
	ctlrs          map[bool][]utilfc.TestableInterface
	lastRVs        map[string]string // FlowSchema key to ResourceVersion
	rvMutex        sync.Mutex        // hold when accessing notifiedRVs
	// maps invert -> i -> FS key -> last notified ResourceVersion
	notifiedRVs  map[bool][]map[string]string
	someTimedOut bool
	iteration    int
	// maps invert -> i -> FS name -> write times (oldest first)
	writeHistories map[bool][]map[string][]time.Time
}

func newFightTest(t *testing.T, loopbackConfig *rest.Config, size int) *fightTest {
	now := time.Now()
	ft := &fightTest{
		t:              t,
		ctx:            context.Background(),
		loopbackConfig: loopbackConfig,
		size:           size,
		stopCh:         make(chan struct{}),
		now:            now,
		clk:            clock.NewFakeClock(now),
		ctlrs: map[bool][]utilfc.TestableInterface{
			false: make([]utilfc.TestableInterface, size),
			true:  make([]utilfc.TestableInterface, size)},
		notifiedRVs: map[bool][]map[string]string{
			false: make([]map[string]string, size),
			true:  make([]map[string]string, size)},
		writeHistories: map[bool][]map[string][]time.Time{
			false: make([]map[string][]time.Time, size),
			true:  make([]map[string][]time.Time, size)},
	}
	ft.foreach(func(invert bool, i int) {
		ft.rvMutex.Lock()
		defer ft.rvMutex.Unlock()
		ft.notifiedRVs[invert][i] = map[string]string{}
		ft.writeHistories[invert][i] = map[string][]time.Time{}
	})
	return ft
}

func (ft *fightTest) createController(invert bool, i int) {
	myConfig := rest.CopyConfig(ft.loopbackConfig)
	myConfig = rest.AddUserAgent(myConfig, fmt.Sprintf("invert=%v, i=%d", invert, i))
	myClientset := clientset.NewForConfigOrDie(myConfig)
	fcIfc := myClientset.FlowcontrolV1beta1()
	if ft.lastRVs == nil {
		var err error
		ft.lastRVs, err = getResourceVersionOfEachFlowSchema(ft.t, fcIfc.FlowSchemas())
		if err != nil {
			ft.t.Fatal(err)
		}
	}
	informerFactory := informers.NewSharedInformerFactory(myClientset, 0)
	fieldMgr := utilfc.ConfigConsumerAsFieldManager
	foundToDangling := func(found bool) bool { return !found }
	if invert {
		fieldMgr = fieldMgr + "x"
		foundToDangling = func(found bool) bool { return found }
	}
	ctlr := utilfc.NewTestable(utilfc.TestableConfig{
		Name:  fmt.Sprintf("Controller%d[invert=%v]", i, invert),
		Clock: ft.clk,
		FinishHandlingNotification: func(wq workqueue.RateLimitingInterface, obj interface{}) {
			ft.t.Logf("For invert=%v, i=%v, notified of %#+v", invert, i, obj)
			obj = peel(obj)
			switch typed := obj.(type) {
			case *flowcontrol.FlowSchema:
				key, _ := cache.MetaNamespaceKeyFunc(obj)
				ft.rvMutex.Lock()
				defer ft.rvMutex.Unlock()
				ft.notifiedRVs[invert][i][key] = typed.ResourceVersion
			}
		},
		AsFieldManager:         fieldMgr,
		FoundToDangling:        foundToDangling,
		InformerFactory:        informerFactory,
		FlowcontrolClient:      fcIfc,
		ServerConcurrencyLimit: 200,         // server concurrency limit
		RequestWaitLimit:       time.Minute, // request wait limit
		ObsPairGenerator:       metrics.PriorityLevelConcurrencyObserverPairGenerator,
		QueueSetFactory:        fqtesting.NewNoRestraintFactory(),
	})
	ft.ctlrs[invert][i] = ctlr
	informerFactory.Start(ft.stopCh)
	if ctlr.WaitForCacheSync(ft.stopCh) {
		ft.t.Logf("Achieved initial sync for invert=%v, i=%v", invert, i)
	} else {
		ft.t.Fatalf("Never achieved initial sync for invert=%v, i=%v", invert, i)
	}
}

func (ft *fightTest) waitForLastRVs() {
	AOK := false
	// wait until notifiedRVs[invert][i] covers lastRVs for all invert, i
	for k := 1; k < 11 && !AOK; k++ {
		ft.t.Logf("For size=%d, iteration=%d, k=%d, starting to wait for lastRVs=%v", ft.size, ft.iteration, k, ft.lastRVs)
		time.Sleep(time.Millisecond * time.Duration(k*100))
		AOK = true
		ft.foreach(func(invert bool, i int) {
			ft.rvMutex.Lock()
			defer ft.rvMutex.Unlock()
			for key, rv := range ft.lastRVs {
				if ft.notifiedRVs[invert][i][key] != rv {
					AOK = false
				}
			}
		})
	}
	if !AOK {
		func() {
			ft.rvMutex.Lock()
			defer ft.rvMutex.Unlock()
			ft.t.Logf("For size=%d, iteration=%d, lastRVs=%v but notifiedRVs=%#+v", ft.size, ft.iteration, ft.lastRVs, ft.notifiedRVs)
		}()
		ft.someTimedOut = true
	}
}

func (ft *fightTest) updateAndCheckHistories(invert bool, i int, ctlrRVs map[string]string) int {
	timeLimit := ft.now.Add(-time.Minute)
	var nWrites int
	for key, rv := range ctlrRVs {
		ft.t.Logf("For invert=%v, i=%v, iteration=%d, wrote to %q", invert, i, ft.iteration, key)
		ft.lastRVs[key] = rv
		nWrites++
		hist := ft.writeHistories[invert][i][key]
		hist = append(hist, ft.now)
		for idx, updateTime := range hist {
			if updateTime.After(timeLimit) {
				hist = hist[idx:]
				break
			}
		}
		ft.writeHistories[invert][i][key] = hist
		if len(hist) > 6 {
			ft.t.Errorf("For invert=%v, i=%v, iteration=%v, FlowSchema %q, found %d updates in the last minute (%#+v)", invert, i, ft.iteration, key, len(hist), hist)
		}
	}
	return nWrites
}

func TestConfigConsumerFight(t *testing.T) {
	// Disable the APF FeatureGate so that the normal config consumer
	// controller does not interfere
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIPriorityAndFairness, false)()
	_, loopbackConfig, closeFn := setup(t, 100, 100)
	defer closeFn()
	const size = 3
	ft := newFightTest(t, loopbackConfig, size)
	ft.foreach(ft.createController)
	t.Logf("After initial sync, lastRVs=%v", ft.lastRVs)
	nextTime := ft.clk.Now()
	lastTime := nextTime.Add(601 * time.Second)
	for ft.iteration = 0; lastTime.After(nextTime); ft.iteration++ {
		ft.waitForLastRVs()
		ft.now = nextTime
		ft.clk.SetTime(ft.now)
		t.Logf("Syncing[size=%d, iteration=%d] at %s", ft.size, ft.iteration, ft.now.Format(timeFmt))
		klog.V(3).Infof("Syncing size=%d, iteration=%d at %s", ft.size, ft.iteration, ft.now.Format(timeFmt))
		ft.lastRVs = make(map[string]string)
		const highTime = 2 * time.Minute
		wait := highTime
		ft.foreach(func(invert bool, i int) {
			ctlr := ft.ctlrs[invert][i]
			ctlrRVs := make(map[string]string)
			report := ctlr.SyncOne(ctlrRVs)
			nWrites := ft.updateAndCheckHistories(invert, i, ctlrRVs)
			if report.NeedRetry {
				t.Errorf("Error for invert=%v, i=%d", invert, i)
			}
			t.Logf("For invert=%v, i=%d: nWrites=%d, NeededSpecificWait=%s", invert, i, nWrites, report.NeededSpecificWait)
			if report.NeededSpecificWait > 0 {
				wait = durationMin(wait, report.NeededSpecificWait)
			}
		})
		t.Logf("For size=%d, iteration=%d at %s, lastRVs = %v", size, ft.iteration, ft.now.Format(timeFmt), ft.lastRVs)
		if wait == highTime {
			wait = time.Second * 4
		}
		nextTime = ft.now.Add(wait)
	}
	close(ft.stopCh)
	if ft.someTimedOut {
		t.Error("Some timed out")
	}
}

func (ft *fightTest) foreach(visit func(invert bool, i int)) {
	for i := 0; i < ft.size; i++ {
		// The order of the following iteration is not deterministic,
		// and that is good.
		invert := rand.Intn(2) == 0
		visit(invert, i)
		visit(!invert, i)
	}
}

func getResourceVersionOfEachFlowSchema(t *testing.T, fsIfc flowcontrolclient.FlowSchemaInterface) (map[string]string, error) {
	lastRVs := make(map[string]string)
	// Wait until every FlowSchema is defined, and record its RV
	allFlowSchemas := append(fcboot.MandatoryFlowSchemas, fcboot.SuggestedFlowSchemas...)
	err := wait.Poll(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		for _, fs := range allFlowSchemas {
			fs2, err := fsIfc.Get(context.Background(), fs.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			t.Logf("Got initial FlowSchema %#+v", fs2)
			key, _ := cache.MetaNamespaceKeyFunc(fs2)
			lastRVs[key] = fs2.ResourceVersion
		}
		return true, nil
	})
	return lastRVs, err
}

func peel(obj interface{}) interface{} {
	if d, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		return d.Obj
	}
	return obj
}

// durationMin computes the minimum of two time.Duration values
func durationMin(x, y time.Duration) time.Duration {
	if x < y {
		return x
	}
	return y
}
