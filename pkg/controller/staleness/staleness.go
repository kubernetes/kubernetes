/*
Copyright The Kubernetes Authors.

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

package staleness

import (
	"context"
	"fmt"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

var (
	apiProbePeriod      = 1 * time.Minute
	informerProbePeriod = 1 * time.Second

	bucketCount         = 10
	watchDelayHistogram = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Subsystem:      "controller_manager",
			Name:           "watch_delay_seconds",
			Help:           "Watch delay seconds, for now calculated only for pods",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        compbasemetrics.ExponentialBuckets(informerProbePeriod.Seconds(), 2, bucketCount),
		},
		[]string{"group", "resource"},
	)
	watchDelayGauge = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Subsystem:      "controller_manager",
			Name:           "current_watch_delay_seconds",
			Help:           "Last observed watch delay seconds, for now calculated only for pods",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "resource"},
	)
)

const fakeProberNamespace = "staleness_probe_fake_namespace!"

func init() {
	legacyregistry.MustRegister(watchDelayGauge)
	legacyregistry.MustRegister(watchDelayHistogram)
}

type PodProber struct {
	// client interface
	kubeClient clientset.Interface
	// podStore is the shared pod cache
	podStore cache.Store

	probeInfo ProbeInfo
}

type ProbeInfo struct {
	mux     *sync.Mutex
	cond    *sync.Cond
	stopped bool

	clock clock.Clock

	// latestRV and latestProbedTime are the resource version and timestamp
	// of the last successful probe. Updated by the prober loop.
	latestRV         string
	latestProbedTime time.Time
	// caughtUp is used to tell the controller whether to
	// run the prober or informer loop.
	caughtUp bool
	// currentStaleness is the staleness of the informer.
	currentStaleness time.Duration
	// lastInformerRV and smallestInformerInterval are used to calculate
	// the currentStaleness. They are updated by the informer loop.
	lastInformerRV   string
	lastInformerTime time.Time
}

func (p *PodProber) runProbeLoop(ctx context.Context) {
	logger := klog.FromContext(ctx)
	for {
		// If we haven't caught up to the real RV yet from our previous probe, we know that we are still stale.
		// The previous probe should still be usable to calculate the staleness.
		if !p.probeInfo.waitForCaughtUp(ctx) {
			return
		}

		start := p.probeInfo.clock.Now()
		// Create ListOptions with a fake namespace that will not match any pods.
		resp, err := p.kubeClient.CoreV1().Pods(fakeProberNamespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			logger.Error(err, "unable to list for prober")
			continue
		}
		if len(resp.Items) != 0 {
			logger.Error(fmt.Errorf("list response not empty"), "The list obtained while probing returned a non-empty value", "length", len(resp.Items))
		}
		p.probeInfo.setRVAndTimestamp(resp.ResourceVersion, start)
		p.probeInfo.signal()

		select {
		case <-ctx.Done():
			return
		case <-time.After(apiProbePeriod):
		}
	}
}

func (p *PodProber) metricsLoop(ctx context.Context) {
	logger := klog.FromContext(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(informerProbePeriod):
		}
		if !p.probeInfo.waitForFreshProbe(ctx) {
			return
		}

		rv := p.podStore.LastStoreSyncResourceVersion()
		_, err := resourceversion.CompareResourceVersion(rv, rv)
		if rv == "" {
			logger.Error(err, "lister was unable to obtain a parseable rv")
			continue
		}

		changed, err := p.probeInfo.UpdateStaleness(rv)
		if err != nil {
			logger.Error(err, "Unable to query staleness for rv", "rv", rv)
			continue
		}

		// If the informer resource version hasn't changed since the last time we checked,
		// we can't update the metrics. This is because we need to see the informer in the
		// process of catching up to the API server to calculate the staleness.
		if !changed {
			continue
		}
		staleness := p.probeInfo.currentStaleness.Seconds()

		watchDelayGauge.WithLabelValues("", "pods").Set(staleness)
		watchDelayHistogram.WithLabelValues("", "pods").Observe(staleness)
	}
}

func (p *ProbeInfo) signal() {
	p.cond.Broadcast()
}

func (p *ProbeInfo) waitForCaughtUp(ctx context.Context) bool {
	return p.waitForCondition(ctx, true)
}

func (p *ProbeInfo) waitForFreshProbe(ctx context.Context) bool {
	return p.waitForCondition(ctx, false)
}

func (p *ProbeInfo) waitForCondition(ctx context.Context, condition bool) bool {
	p.mux.Lock()
	defer p.mux.Unlock()

	for p.caughtUp != condition {
		if p.stopped {
			return false
		}
		p.cond.Wait()
	}
	return true
}

func (p *ProbeInfo) setRVAndTimestamp(rv string, timestamp time.Time) {
	p.mux.Lock()
	defer p.mux.Unlock()
	p.latestRV = rv
	p.latestProbedTime = timestamp
	p.caughtUp = false
}

func NewPodProber(ctx context.Context, client clientset.Interface, store cache.Store) *PodProber {
	probeMutex := &sync.Mutex{}
	newProber := &PodProber{
		kubeClient: client,
		podStore:   store,
		probeInfo: ProbeInfo{
			mux:      probeMutex,
			cond:     sync.NewCond(probeMutex),
			caughtUp: true,
			clock:    clock.RealClock{},
		},
	}
	return newProber
}

func (p *PodProber) Run(ctx context.Context) {
	logger := klog.FromContext(ctx)
	logger.Info("Starting staleness controller")
	defer func() {
		p.probeInfo.mux.Lock()
		p.probeInfo.stopped = true
		p.probeInfo.cond.Broadcast()
		p.probeInfo.mux.Unlock()
		logger.Info("Shutting down staleness controller")
	}()
	go p.runProbeLoop(ctx)
	go p.metricsLoop(ctx)
	<-ctx.Done()
}

func (p *ProbeInfo) UpdateStaleness(podRV string) (bool, error) {
	p.mux.Lock()
	defer p.mux.Unlock()
	if p.latestRV == "" {
		return false, fmt.Errorf("unable to query when no probe has happened")
	}
	if p.lastInformerRV == podRV {
		return false, nil
	}
	p.lastInformerRV = podRV

	cmp, err := resourceversion.CompareResourceVersion(p.latestRV, podRV)
	if err != nil {
		return false, err
	}

	var currentStaleness time.Duration
	if p.lastInformerTime.IsZero() {
		currentStaleness = 0
	} else {
		currentStaleness = p.lastInformerTime.Sub(p.latestProbedTime)
	}

	// if latestRV > podRV, then we are stale, otherwise we are caught up.
	if cmp > 0 {
		// We are stale, update the current staleness and return it.
		if p.clock != nil {
			p.lastInformerTime = p.clock.Now()
		} else {
			p.lastInformerTime = time.Now()
		}
		// The current staleness will be the value used and returned, and should only increase from the last
		// time we were caught up since we do not want to record a staleness value that is less than the last observed staleness
		// while the informer is still catching up. We can only observe increases in staleness while in this state.
		if currentStaleness > p.currentStaleness {
			p.currentStaleness = currentStaleness
		}
		return true, nil
	} else {
		// We are caught up, return the last known staleness. Signal that we are caught up so probers can run another probe.
		// This should pause the informer metrics loop until the next api probe. At this point we know the exact staleness
		// since the informer has caught up to the api server within the informer's probe period.
		p.currentStaleness = currentStaleness
		p.clearInformerProberStateLocked()
		p.signal()
		return true, nil
	}
}

func (p *ProbeInfo) clearInformerProberStateLocked() {
	p.lastInformerTime = time.Time{}
	p.lastInformerRV = ""
	p.caughtUp = true
}
