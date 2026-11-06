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

package cacher

import (
	"context"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	"k8s.io/apiserver/pkg/storage/cacher/progress"
	"k8s.io/utils/clock"
)

const (
	// blockTimeout determines how long we're willing to block the request
	// to wait for a given resource version to be propagated to cache,
	// before terminating request and returning Timeout error with retry
	// after suggestion.
	blockTimeout = 3 * time.Second

	// resourceVersionTooHighRetrySeconds is the seconds before a operation should be retried by the client
	// after receiving a 'too high resource version' error.
	resourceVersionTooHighRetrySeconds = 1
)

func newFreshnessWaiter(rLocker sync.Locker, resourceVersion *uint64, c clock.Clock, groupResource schema.GroupResource, pr *progress.ConditionalProgressRequester) *freshnessWaiter {
	return &freshnessWaiter{
		cond:              sync.NewCond(rLocker),
		resourceVersion:   resourceVersion,
		clock:             c,
		groupResource:     groupResource,
		progressRequester: pr,
	}
}

type freshnessWaiter struct {
	cond              *sync.Cond
	resourceVersion   *uint64
	clock             clock.Clock
	groupResource     schema.GroupResource
	progressRequester *progress.ConditionalProgressRequester
}

func (w *freshnessWaiter) broadcast() {
	w.cond.Broadcast()
}

func (w *freshnessWaiter) waitUntilFreshLocked(ctx context.Context, resourceVersion uint64) error {
	if resourceVersion == 0 || resourceVersion <= *w.resourceVersion {
		return nil
	}
	w.progressRequester.Add()
	defer w.progressRequester.Remove()
	startTime := w.clock.Now()
	defer func() {
		if resourceVersion > 0 {
			metrics.WatchCacheReadWait.WithContext(ctx).WithLabelValues(w.groupResource.Group, w.groupResource.Resource).Observe(w.clock.Since(startTime).Seconds())
		}
	}()

	// In case resourceVersion is 0, we accept arbitrarily stale result.
	// As a result, the condition in the below for loop will never be
	// satisfied (w.resourceVersion is never negative), this call will
	// never hit the w.cond.Wait().
	// As a result - we can optimize the code by not firing the wakeup
	// function (and avoid starting a gorotuine), especially given that
	// resourceVersion=0 is the most common case.
	if resourceVersion > 0 {
		go func() {
			// Wake us up when the time limit has expired.  The docs
			// promise that time.After (well, NewTimer, which it calls)
			// will wait *at least* the duration given. Since this go
			// routine starts sometime after we record the start time, and
			// it will wake up the loop below sometime after the broadcast,
			// we don't need to worry about waking it up before the time
			// has expired accidentally.
			<-w.clock.After(blockTimeout)
			w.cond.Broadcast()
		}()
	}

	for *w.resourceVersion < resourceVersion {
		if w.clock.Since(startTime) >= blockTimeout {
			// Request that the client retry after 'resourceVersionTooHighRetrySeconds' seconds.
			return storage.NewTooLargeResourceVersionError(resourceVersion, *w.resourceVersion, resourceVersionTooHighRetrySeconds)
		}
		w.cond.Wait()
	}
	return nil
}
