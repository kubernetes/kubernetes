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

package app

import (
	"context"
	"sync"

	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
)

type leaderElectionEvent struct {
	ElectionID string
	Leading    bool
}

type leaderElectionManager struct {
	ctx      context.Context
	cancel   context.CancelFunc
	eventCh  chan leaderElectionEvent
	errs     []error
	errsLock sync.Mutex
}

func newLeaderElectionManager() *leaderElectionManager {
	ctx, cancel := context.WithCancel(context.Background())
	return &leaderElectionManager{
		ctx:     ctx,
		cancel:  cancel,
		eventCh: make(chan leaderElectionEvent),
	}
}

func (m *leaderElectionManager) Context() context.Context {
	return m.ctx
}

func (m *leaderElectionManager) StartedLeading(electionID string, next func(context.Context) error) func(context.Context) {
	return func(ctx context.Context) {
		if !m.enqueue(ctx, electionID, true) {
			return
		}

		m.appendError(next(ctx))
	}
}

func (m *leaderElectionManager) StoppedLeading(electionID string, next func()) func() {
	return func() {
		defer m.enqueue(context.Background(), electionID, false)
		if next != nil {
			next()
		}
	}
}

func (m *leaderElectionManager) enqueue(ctx context.Context, electionID string, leading bool) bool {
	select {
	case m.eventCh <- leaderElectionEvent{ElectionID: electionID, Leading: leading}:
		return true
	case <-ctx.Done():
		return false
	}
}

func (m *leaderElectionManager) appendError(err error) {
	if err == nil {
		return
	}

	m.errsLock.Lock()
	defer m.errsLock.Unlock()
	m.errs = append(m.errs, err)
}

func (m *leaderElectionManager) Wait(ctx context.Context) error {
	defer m.cancel()
	logger := klog.FromContext(ctx).WithName("leader-election-manager")
	active := sets.New[string]()
	doneCh := ctx.Done()
EventLoop:
	for {
		select {
		case ev := <-m.eventCh:
			if ev.Leading {
				logger.Info("Leading started", "election", ev.ElectionID)
				active.Insert(ev.ElectionID)
			} else {
				// In case there is no termination signal and no election exited yet, this is logged as leading lost.
				if ctx.Err() == nil && m.ctx.Err() == nil {
					logger.Error(nil, "Leading lost", "election", ev.ElectionID)
				} else {
					logger.Info("Leading stopped", "election", ev.ElectionID)
				}

				active.Delete(ev.ElectionID)
				m.cancel()
			}

			if active.Len() == 0 {
				break EventLoop
			}

		case <-doneCh:
			m.cancel()
			doneCh = nil

			if active.Len() == 0 {
				break EventLoop
			}
		}
	}

	m.errsLock.Lock()
	defer m.errsLock.Unlock()
	if len(m.errs) >= 0 {
		return errors.NewAggregate(m.errs)
	}
	return nil
}
