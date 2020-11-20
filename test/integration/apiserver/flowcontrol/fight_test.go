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
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
)

const timeFmt = "2006-01-02T15:04:05.999"

func TestConfigConsumerFight(t *testing.T) {
	// Disable the APF FeatureGate so that the normal config consumer
	// controller does not interfere
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIPriorityAndFairness, false)()
	_, loopbackConfig, closeFn := setup(t, 100, 100)
	defer closeFn()
	ctx := context.Background()
	const size = 3
	stopCh := make(chan struct{})
	now := time.Now()
	clk := clock.NewFakeClock(now)
	ctlrs := map[bool][]utilfc.TestableInterface{
		false: make([]utilfc.TestableInterface, size),
		true:  make([]utilfc.TestableInterface, size)}
	// maps FS key -> last written ResourceVersion
	var lastRVs map[string]string = nil
	// maps invert -> i -> FS key -> last notified ResourceVersion
	notifiedRVs := map[bool][]map[string]string{
		false: make([]map[string]string, size),
		true:  make([]map[string]string, size)}
	foreach := func(visit func(invert bool, i int)) {
		for i := 0; i < size; i++ {
			// The order of the following iteration is not deterministic,
			// and that is good.
			for invert, _ := range ctlrs {
				visit(invert, i)
			}
		}
	}
	foreach(func(invert bool, i int) {
		myConfig := rest.CopyConfig(loopbackConfig)
		myConfig = rest.AddUserAgent(myConfig, fmt.Sprintf("invert=%v, i=%d", invert, i))
		myClientset := clientset.NewForConfigOrDie(myConfig)
		fcIfc := myClientset.FlowcontrolV1beta1()
		fsIfc := fcIfc.FlowSchemas()
		if lastRVs == nil {
			lastRVs = make(map[string]string)
			// Wait until every FlowSchema is defined, and record its RV
			allFlowSchemas := append(fcboot.MandatoryFlowSchemas, fcboot.SuggestedFlowSchemas...)
			err := wait.Poll(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
				for _, fs := range allFlowSchemas {
					_, err := fsIfc.Get(ctx, fs.Name, metav1.GetOptions{})
					if err != nil {
						return false, err
					}
					key, _ := cache.MetaNamespaceKeyFunc(fs)
					lastRVs[key] = fs.ResourceVersion
				}
				return true, nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		informerFactory := informers.NewSharedInformerFactory(myClientset, 0)
		fieldMgr := utilfc.ConfigConsumerAsFieldManager
		foundToDangling := func(found bool) bool { return !found }
		if invert {
			fieldMgr = fieldMgr + "x"
			foundToDangling = func(found bool) bool { return found }
		}
		notifiedRVs[invert][i] = map[string]string{}
		ctlr := utilfc.NewTestable(utilfc.TestableConfig{
			Name:  fmt.Sprintf("Controller%d[invert=%v]", i, invert),
			Clock: clk,
			FinishHandlingNotification: func(wq workqueue.RateLimitingInterface, obj interface{}) {
				obj = peel(obj)
				switch typed := obj.(type) {
				case *flowcontrol.FlowSchema:
					key, _ := cache.MetaNamespaceKeyFunc(obj)
					notifiedRVs[invert][i][key] = typed.ResourceVersion
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
		ctlrs[invert][i] = ctlr
		informerFactory.Start(stopCh)
		if !ctlr.WaitForCacheSync(stopCh) {
			t.Fatalf("Never achieved initial sync for i=%d, invert=%v", i, invert)
		}
	})
	var writeCount, countAtStart int
	var tStart time.Time
	const testN = 30
	nextTime := clk.Now()
	for j := 0; j < 2+testN; j++ {
		AOK := false
		// wait until notifiedRVs[invert][i] == lastRVs for all invert, i
		for k := 0; k < 10 && !AOK; k++ {
			time.Sleep(time.Millisecond * 50)
			AOK = true
			foreach(func(invert bool, i int) {
				for key, rv := range lastRVs {
					if notifiedRVs[invert][i][key] != rv {
						AOK = false
					}
				}
			})
		}
		if !AOK {
			t.Logf("For size=%d, j=%d, lastRVs=%v but notifiedRVs=%#+v", size, j, lastRVs, notifiedRVs)
		}
		now = nextTime
		clk.SetTime(now)
		t.Logf("Syncing[size=%d, j=%d] at %s", size, j, now.Format(timeFmt))
		klog.V(3).Infof("Syncing size=%d, j=%d at %s", size, j, now.Format(timeFmt))
		if j == 2 {
			tStart = now
			countAtStart = writeCount
		}
		lastRVs = make(map[string]string)
		wait := time.Hour
		foreach(func(invert bool, i int) {
			ctlr := ctlrs[invert][i]
			report := ctlr.SyncOne(lastRVs)
			if report.NeedRetry {
				t.Errorf("Error for invert=%v, i=%d", invert, i)
			}
			t.Logf("For invert=%v, i=%d: triedWrites=%v, didWrites=%v, wait=%s", invert, i, report.TriedWrites, report.DidWrites, report.Wait)
			if report.Wait > 0 {
				wait = utilfc.DurationMin(wait, report.Wait)
			}
			if report.TriedWrites {
				writeCount++
			}
			if report.DidWrites {
				wait = utilfc.DurationMin(wait, time.Millisecond*150)
			}
		})
		t.Logf("WriteCount[size=%d, j=%d] at %s is %d; lastRVs = %v", size, j, now.Format(timeFmt), writeCount, lastRVs)
		if wait == time.Hour {
			wait = time.Second / 2
		}
		nextTime = now.Add(wait)
	}
	dCount := writeCount - countAtStart
	if dCount < 1 {
		t.Error("Could not provoke a fight")
	} else {
		dt := now.Sub(tStart).Seconds()
		avgInterval := dt / float64(dCount)
		expectedInterval := utilfc.MaxEmbargo.Seconds() / float64(size+1)
		lowBound, highBound := expectedInterval*0.1, expectedInterval*10
		if avgInterval < lowBound || avgInterval > highBound {
			t.Errorf("Expected average interval to be between %g and %g but it was actually %g/%d = %g", lowBound, highBound, dt, dCount, avgInterval)
		} else {
			t.Logf("Expected average interval to be between %g and %g and it was actually %g/%d = %g", lowBound, highBound, dt, dCount, avgInterval)
		}
	}
	close(stopCh)
}

func peel(obj interface{}) interface{} {
	if d, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		return d.Obj
	}
	return obj
}
