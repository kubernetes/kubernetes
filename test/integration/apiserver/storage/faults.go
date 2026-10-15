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

package storage

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/testing/correctness"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

func hasUnreadyListError(ops []correctness.Operation) bool {
	for _, op := range ops {
		if op.Request.Op == correctness.OpList && apierrors.IsTooManyRequests(op.Response.Err) && !storage.IsTooLargeResourceVersion(op.Response.Err) {
			return true
		}
	}
	return false
}

func hasUnaryTimeoutError(ops []correctness.Operation) bool {
	for _, op := range ops {
		if storage.IsTooLargeResourceVersion(op.Response.Err) {
			return true
		}
	}
	return false
}

// hasConsistentListFallback checks whether any consistent List (ResourceVersion == "")
// waited for cacher's 3s blockTimeout before falling back to etcd and succeeding.
func hasConsistentListFallback(ops []correctness.Operation) bool {
	for _, op := range ops {
		if op.Request.Op == correctness.OpList && op.Request.List.Options.ResourceVersion == "" && op.Response.Err == nil && op.End.Sub(op.Start) >= 3*time.Second {
			return true
		}
	}
	return false
}

func hasWatchTooManyRequestsError(watches []correctness.WatchOperation) bool {
	for _, w := range watches {
		if apierrors.IsTooManyRequests(w.Response.Err) {
			return true
		}
	}
	return false
}

func hasWatchEventReason(watches []correctness.WatchOperation, reason metav1.StatusReason) bool {
	for _, w := range watches {
		for _, ev := range w.Response.Events {
			if ev.Type != watch.Error {
				continue
			}
			if status, ok := ev.Object.(*metav1.Status); ok && status.Reason == reason {
				return true
			}
		}
	}
	return false
}

type watchBreakingStore struct {
	storage.Interface
	expireAfter   time.Duration
	failNextWatch atomic.Int32
}

func (s *watchBreakingStore) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	for {
		n := s.failNextWatch.Load()
		if n <= 0 {
			break
		}
		if s.failNextWatch.CompareAndSwap(n, n-1) {
			return nil, errors.New("injected watch error")
		}
	}
	w, err := s.Interface.Watch(ctx, key, opts)
	if err != nil {
		return nil, err
	}
	ch := make(chan watch.Event)
	pw := watch.NewProxyWatcher(ch)
	go s.expireWatchAfterSync(w, pw, ch, opts.SendInitialEvents != nil && *opts.SendInitialEvents)
	return pw, nil
}

func (s *watchBreakingStore) expireWatchAfterSync(w watch.Interface, pw *watch.ProxyWatcher, out chan<- watch.Event, waitForInitialEventsEnd bool) {
	defer close(out)
	defer w.Stop()

	synced := !waitForInitialEventsEnd
	var timerCh <-chan time.Time
	var timer *time.Timer
	defer func() {
		if timer != nil {
			timer.Stop()
		}
	}()

	for {
		select {
		case <-pw.StopChan():
			return
		case <-timerCh:
			// Fail both the WatchList attempt and the fallback Watch attempt in the
			// next Reflector.ListAndWatch cycle so that:
			// 1. Reflector falls back to listerWatcher.List + watchCache.Replace (waking blocked readers).
			// 2. The subsequent Watch fails, incrementing WatchCacheInitializationErrors and keeping
			//    the cacher in Pending state while unary workers are active.
			s.failNextWatch.Store(2)
			errEvent := watch.Event{
				Type:   watch.Error,
				Object: &apierrors.NewResourceExpired("watch expired").ErrStatus,
			}
			select {
			case <-pw.StopChan():
			case out <- errEvent:
			}
			return
		case ev, ok := <-w.ResultChan():
			if !ok {
				return
			}
			if !synced {
				if ev.Type == watch.Bookmark {
					if annotated, err := storage.HasInitialEventsEndBookmarkAnnotation(ev.Object); err == nil && annotated {
						synced = true
					}
				}
			} else if timer == nil && s.expireAfter > 0 {
				timer = time.NewTimer(s.expireAfter)
				timerCh = timer.C
			}
			select {
			case <-pw.StopChan():
				return
			case out <- ev:
			}
		}
	}
}

type watchBlockTracker struct {
	stalling atomic.Bool
	once     sync.Once
	ch       chan struct{}
}

func (t *watchBlockTracker) start() {
	t.ch = make(chan struct{})
	t.stalling.Store(true)
}

// observe reports whether the stall is active and, when ctx is a per-watch context from
// runWatch (~2s deadline), closes t.ch if ctx remains open past a normal 500ms watch cycle
// because cacher.Watch is blocked in waitUntilFreshLocked (3s blockTimeout).
func (t *watchBlockTracker) observe(ctx context.Context) bool {
	if !t.stalling.Load() {
		return false
	}
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < 5*time.Second {
		go func() {
			select {
			case <-ctx.Done():
				return
			case <-time.After(1000 * time.Millisecond):
				t.once.Do(func() {
					close(t.ch)
				})
			}
		}()
	}
	return true
}

func (t *watchBlockTracker) wait(stopCh <-chan struct{}, minDuration time.Duration) {
	defer t.stalling.Store(false)
	minTimer := time.NewTimer(minDuration)
	defer minTimer.Stop()

	select {
	case <-stopCh:
		return
	case <-time.After(10 * time.Second):
		return
	case <-t.ch:
		t.stalling.Store(false)
	}

	// t.ch closed after the WatchList request had already been blocked for 1s;
	// wait 2.2s more (3.2s total > 3s blockTimeout) and ensure minDuration elapsed.
	select {
	case <-stopCh:
		return
	case <-time.After(2200 * time.Millisecond):
	}
	select {
	case <-stopCh:
	case <-minTimer.C:
	}
}

type watchDelayingStore struct {
	storage.Interface
	delay   time.Duration
	stalled atomic.Bool
	tracker watchBlockTracker
}

func (s *watchDelayingStore) GetCurrentResourceVersion(ctx context.Context) (uint64, error) {
	s.tracker.observe(ctx)
	return s.Interface.GetCurrentResourceVersion(ctx)
}

func (s *watchDelayingStore) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	w, err := s.Interface.Watch(ctx, key, opts)
	if err != nil {
		return nil, err
	}
	ch := make(chan watch.Event)
	pw := watch.NewProxyWatcher(ch)
	go s.delayWatchEvents(w, pw, ch, opts.SendInitialEvents != nil && *opts.SendInitialEvents)
	return pw, nil
}

func (s *watchDelayingStore) delayWatchEvents(w watch.Interface, pw *watch.ProxyWatcher, out chan<- watch.Event, waitForInitialEventsEnd bool) {
	defer close(out)
	defer w.Stop()

	synced := !waitForInitialEventsEnd
	queue := make(chan watch.Event, 2048)
	go func() {
		defer close(queue)
		for {
			select {
			case <-pw.StopChan():
				return
			case ev, ok := <-w.ResultChan():
				if !ok {
					return
				}
				select {
				case <-pw.StopChan():
					return
				case queue <- ev:
				}
			}
		}
	}()

	postSyncEvents := 0
	for ev := range queue {
		if !synced {
			if ev.Type == watch.Bookmark {
				if annotated, err := storage.HasInitialEventsEndBookmarkAnnotation(ev.Object); err == nil && annotated {
					synced = true
				}
			}
		} else if ev.Type != watch.Bookmark {
			postSyncEvents++
			// After initial sync and enough write traffic to populate RVCached/RVPast, stall
			// watch delivery once so concurrent consistent reads, cached reads, and WatchList
			// requests time out in waitUntilFreshLocked (> 3s blockTimeout).
			if postSyncEvents == 100 && s.delay > 0 && s.stalled.CompareAndSwap(false, true) {
				s.tracker.start()
				s.tracker.wait(pw.StopChan(), s.delay)
			}
		}
		select {
		case <-pw.StopChan():
			return
		case out <- ev:
		}
	}
}

type bookmarkDroppingStore struct {
	storage.Interface
	stallDuration time.Duration
	writesSeen    atomic.Int64
	stalled       atomic.Bool
	tracker       watchBlockTracker
	progressCalls atomic.Int64
	writeMu       sync.RWMutex
}

func (s *bookmarkDroppingStore) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	s.writeMu.RLock()
	defer s.writeMu.RUnlock()
	err := s.Interface.Create(ctx, key, obj, out, ttl)
	if err == nil {
		s.writesSeen.Add(1)
	}
	return err
}

func (s *bookmarkDroppingStore) Delete(ctx context.Context, key string, out runtime.Object, preconditions *storage.Preconditions, validateDeletion storage.ValidateObjectFunc, cachedExistingObject runtime.Object, opts storage.DeleteOptions) error {
	s.writeMu.RLock()
	defer s.writeMu.RUnlock()
	err := s.Interface.Delete(ctx, key, out, preconditions, validateDeletion, cachedExistingObject, opts)
	if err == nil {
		s.writesSeen.Add(1)
	}
	return err
}

func (s *bookmarkDroppingStore) GuaranteedUpdate(ctx context.Context, key string, destination runtime.Object, ignoreNotFound bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, cachedExistingObject runtime.Object) error {
	s.writeMu.RLock()
	defer s.writeMu.RUnlock()
	err := s.Interface.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, cachedExistingObject)
	if err == nil {
		s.writesSeen.Add(1)
	}
	return err
}

func (s *bookmarkDroppingStore) GetCurrentResourceVersion(ctx context.Context) (uint64, error) {
	if s.stallDuration > 0 && s.writesSeen.Load() >= 100 && s.stalled.CompareAndSwap(false, true) {
		ready := make(chan struct{})
		go func() {
			s.writeMu.Lock()
			defer s.writeMu.Unlock()
			// Let watchCache drain any watch events from writes completed right before writeMu.Lock().
			time.Sleep(20 * time.Millisecond)
			s.tracker.start()
			close(ready)
			s.tracker.wait(nil, s.stallDuration)
		}()
		<-ready
	}
	rv, err := s.Interface.GetCurrentResourceVersion(ctx)
	if err != nil {
		return 0, err
	}
	// While "/pods" writes are paused, report currentRV+1 to simulate etcd's global
	// revision advancing on another resource prefix without mutating "/pods".
	// Because progress bookmarks are dropped, watchCache cannot advance to rv+1
	// via RequestWatchProgress and consistent reads time out after blockTimeout (3s).
	if s.tracker.observe(ctx) {
		return rv + 1, nil
	}
	return rv, nil
}

func (s *bookmarkDroppingStore) RequestWatchProgress(ctx context.Context) error {
	if s.progressCalls.Add(1)%2 == 1 {
		return errors.New("injected watch progress error")
	}
	return s.Interface.RequestWatchProgress(ctx)
}

func (s *bookmarkDroppingStore) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
	w, err := s.Interface.Watch(ctx, key, opts)
	if err != nil {
		return nil, err
	}
	ch := make(chan watch.Event)
	pw := watch.NewProxyWatcher(ch)
	go s.dropProgressBookmarks(w, pw, ch)
	return pw, nil
}

func (s *bookmarkDroppingStore) dropProgressBookmarks(w watch.Interface, pw *watch.ProxyWatcher, out chan<- watch.Event) {
	defer close(out)
	defer w.Stop()

	for {
		select {
		case <-pw.StopChan():
			return
		case ev, ok := <-w.ResultChan():
			if !ok {
				return
			}
			if ev.Type == watch.Bookmark {
				if annotated, err := storage.HasInitialEventsEndBookmarkAnnotation(ev.Object); err != nil || !annotated {
					continue
				}
			}
			select {
			case <-pw.StopChan():
				return
			case out <- ev:
			}
		}
	}
}
