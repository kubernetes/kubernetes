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
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

type watchBreakingStore struct {
	storage.Interface
	expireAfter time.Duration
}

func (s *watchBreakingStore) Watch(ctx context.Context, key string, opts storage.ListOptions) (watch.Interface, error) {
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
