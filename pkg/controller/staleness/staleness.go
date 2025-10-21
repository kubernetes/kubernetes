/*
Copyright 2025 The Kubernetes Authors.

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
	"math"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
)

var (
	probePeriod              = 5 * time.Second
	maxDelayMeasured         = 5 * time.Minute
	queueSize                = int(maxDelayMeasured/probePeriod) + 1
	minSample                = probePeriod / 2
	bucketCount              = 10
	bucketsExponentialFactor = math.Pow(float64(maxDelayMeasured)/float64(minSample), 1.0/float64(bucketCount-1))
	watchDelayHistogram      = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Subsystem:      "controller_manager",
			Name:           "watch_delay_seconds",
			Help:           "Watch delay seconds, for now calculated only for pods",
			StabilityLevel: compbasemetrics.ALPHA,
			Buckets:        compbasemetrics.ExponentialBuckets(minSample.Seconds(), bucketsExponentialFactor, bucketCount),
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

type RVTime struct {
	rv  string
	now time.Time
}

type PodProber struct {
	logger klog.Logger
	// client interface
	kubeClient clientset.Interface
	// podStore is the shared pod cache
	podStore cache.Store

	mux     sync.Mutex
	rvQueue []RVTime
}

func (p *PodProber) runProbeLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(probePeriod):
		}
		start := time.Now()

		// Create ListOptions with a fake namespace that will not match any pods.
		resp, err := p.kubeClient.CoreV1().Pods(fakeProberNamespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			p.logger.Error(err, "unable to list for prober")
			continue
		}
		if len(resp.Items) != 0 {
			p.logger.Error(fmt.Errorf("list response not empty"), "The list obtained while probing returned a non-empty value", "length", len(resp.Items))
		}
		rv := resp.ResourceVersion
		p.mux.Lock()
		p.rvQueue = append(p.rvQueue, RVTime{
			rv:  rv,
			now: start,
		})
		if len(p.rvQueue) > queueSize {
			p.rvQueue = p.rvQueue[len(p.rvQueue)-queueSize:]
		}
		p.mux.Unlock()

	}
}

func (p *PodProber) metricsLoop(ctx context.Context) {
	curRV := ""
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(probePeriod / 2):
		}
		rv := p.podStore.LastStoreSyncResourceVersion()
		_, err := resourceversion.CompareResourceVersion(rv, rv)
		if rv == "" {
			p.logger.Error(err, "lister was unable to obtain a parseable rv")
			continue
		}

		// We cannot easily tell whether the resource version is not changing
		// due to staleness or because the set of pods is no longer changing. We
		// trade off not seeing stuck informers to prevent an infinite delay
		// being recorded when there are no changes occurring to pods.
		if curRV == rv {
			continue
		}
		curRV = rv

		time, err := p.QueryStaleness(rv)
		if err != nil {
			p.logger.Error(err, "Unable to query staleness for rv", "rv", rv)
		}

		watchDelayGauge.WithLabelValues("", "pods").Set(time.Seconds())
		watchDelayHistogram.WithLabelValues("", "pods").Observe(time.Seconds())
	}
}

func NewPodProber(ctx context.Context, client clientset.Interface, store cache.Store) *PodProber {
	logger := klog.FromContext(ctx)
	newProber := &PodProber{
		kubeClient: client,
		podStore:   store,
		logger:     logger,
	}
	return newProber
}

func (p *PodProber) Run(ctx context.Context) {
	go p.runProbeLoop(ctx)
	go p.metricsLoop(ctx)
	<-ctx.Done()
}

func (p *PodProber) QueryStaleness(podRV string) (time.Duration, error) {
	p.mux.Lock()
	defer p.mux.Unlock()
	for i := len(p.rvQueue) - 1; i >= 0; i-- {
		probedRV := p.rvQueue[i].rv
		cmp, err := resourceversion.CompareResourceVersion(probedRV, podRV)
		if err != nil {
			return 0, err
		}
		if cmp <= 0 {
			// We pick the optimistic case to better show the normal case, where
			// we will likely have a cache update time in the milliseconds. If
			// we choose the pessimistic case then we will log a staleness time
			// in the seconds when normally that is not the case. When cache
			// updates come in time we will log the time as zero to show that
			// the cache is as up to date as we can see with our probe interval.
			var t time.Duration
			if i == len(p.rvQueue)-1 {
				t = 0
			} else {
				t = time.Since(p.rvQueue[i+1].now)
			}
			return t, nil
		}
	}

	if len(p.rvQueue) == 0 {
		return 0, fmt.Errorf("unable to query when queue is empty")
	}
	// Return the oldest time we have in our queue if we are older, we can't be
	// any more accurate than that for staleness.
	return time.Since(p.rvQueue[0].now), nil
}
