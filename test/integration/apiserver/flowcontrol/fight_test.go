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
	fsName := fcboot.MandatoryPriorityLevelConfigurationCatchAll.Name
	stopCh := make(chan struct{})
	now := time.Now()
	clk := clock.NewFakeClock(now)
	ctlrs := map[bool][]utilfc.TestableInterface{
		false: make([]utilfc.TestableInterface, size),
		true:  make([]utilfc.TestableInterface, size)}
	foreach := func(visit func(invert bool, i int, ctlr utilfc.TestableInterface)) {
		for i := 0; i < size; i++ {
			// The order of the following iteration is not deterministic,
			// and that is good.
			for invert, slice := range ctlrs {
				visit(invert, i, slice[i])
			}
		}
	}
	foreach(func(invert bool, i int, _ utilfc.TestableInterface) {
		myConfig := rest.CopyConfig(loopbackConfig)
		myConfig = rest.AddUserAgent(myConfig, fmt.Sprintf("invert=%v, i=%d", invert, i))
		myClientset := clientset.NewForConfigOrDie(myConfig)
		fcIfc := myClientset.FlowcontrolV1beta1()
		// Wait until at least one FlowSchema has been defined by the config producer
		err := wait.Poll(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
			_, err := fcIfc.FlowSchemas().Get(ctx, fsName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return true, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		informerFactory := informers.NewSharedInformerFactory(myClientset, 0)
		fieldMgr := utilfc.ConfigConsumerAsFieldManager
		foundToDangling := func(found bool) bool { return !found }
		if invert {
			fieldMgr = fieldMgr + "x"
			foundToDangling = func(found bool) bool { return found }
		}
		ctlr := utilfc.NewTestable(
			fmt.Sprintf("Controller%d[invert=%v]", i, invert),
			clk,
			func(workqueue.RateLimitingInterface) {},
			fieldMgr,
			foundToDangling,
			informerFactory,
			fcIfc,
			200,         // server concurrency limit
			time.Minute, // request wait limit
			metrics.PriorityLevelConcurrencyObserverPairGenerator,
			fqtesting.NewNoRestraintFactory(),
		)
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
		time.Sleep(time.Millisecond * 150)
		now = nextTime
		clk.SetTime(now)
		t.Logf("Syncing[size=%d, j=%d] at %s", size, j, now.Format(timeFmt))
		klog.V(3).Infof("Syncing size=%d, j=%d at %s", size, j, now.Format(timeFmt))
		if j == 2 {
			tStart = now
			countAtStart = writeCount
		}
		wait := time.Hour
		foreach(func(invert bool, i int, ctlr utilfc.TestableInterface) {
			report := ctlr.SyncOne()
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
		t.Logf("WriteCount[size=%d, j=%d] at %s is %d", size, j, now.Format(timeFmt), writeCount)
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
