/*
Copyright 2023 The Kubernetes Authors.

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

package progress

import (
	"context"
	"sync"
	"time"

	"google.golang.org/grpc/metadata"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

const (
	// progressRequestPeriod determines period of requesting progress
	// from etcd when there is a request waiting for watch cache to be fresh.
	progressRequestPeriod = 100 * time.Millisecond
)

func NewConditionalProgressRequester(requestWatchProgress WatchProgressRequester, clock TickerFactory, contextMetadata metadata.MD) *ConditionalProgressRequester {
	pr := &ConditionalProgressRequester{
		clock:                clock,
		requestWatchProgress: requestWatchProgress,
		contextMetadata:      contextMetadata,
	}
	pr.cond = sync.NewCond(&pr.mux)
	return pr
}

type WatchProgressRequester func(ctx context.Context) error

type TickerFactory interface {
	NewTimer(time.Duration) clock.Timer
}

// ConditionalProgressRequester will request progress notification if there
// is a request waiting for watch cache to be fresh.
type ConditionalProgressRequester struct {
	clock                TickerFactory
	requestWatchProgress WatchProgressRequester
	contextMetadata      metadata.MD

	mux     sync.Mutex
	cond    *sync.Cond
	waiting int
	stopped bool
}

func (pr *ConditionalProgressRequester) Run(stopCh <-chan struct{}) {
	ctx := wait.ContextForChannel(stopCh)
	if pr.contextMetadata != nil {
		ctx = metadata.NewOutgoingContext(ctx, pr.contextMetadata)
	}
	go func() {
		defer utilruntime.HandleCrash()
		<-stopCh
		pr.mux.Lock()
		defer pr.mux.Unlock()
		pr.stopped = true
		pr.cond.Signal()
	}()
	timer := pr.clock.NewTimer(progressRequestPeriod)
	defer timer.Stop()
	for {
		stopped := func() bool {
			pr.mux.Lock()
			defer pr.mux.Unlock()
			for pr.waiting == 0 && !pr.stopped {
				pr.cond.Wait()
			}
			return pr.stopped
		}()
		if stopped {
			return
		}

		select {
		case <-timer.C():
			shouldRequest := func() bool {
				pr.mux.Lock()
				defer pr.mux.Unlock()
				return pr.waiting > 0 && !pr.stopped
			}()
			if !shouldRequest {
				timer.Reset(0)
				continue
			}
			timer.Reset(progressRequestPeriod)
			err := pr.requestWatchProgress(ctx)
			if err != nil {
				klog.V(4).InfoS("Error requesting bookmark", "err", err)
			}
		case <-stopCh:
			return
		}
	}
}

func (pr *ConditionalProgressRequester) Add() {
	pr.mux.Lock()
	defer pr.mux.Unlock()
	pr.waiting += 1
	pr.cond.Signal()
}

func (pr *ConditionalProgressRequester) Remove() {
	pr.mux.Lock()
	defer pr.mux.Unlock()
	pr.waiting -= 1
}
