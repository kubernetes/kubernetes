/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package component

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"

	"github.com/golang/glog"
)

type HeartBeat interface {
	// Transition changes the component's current phase (locally)
	// Future heartbeats will communicate this to the apiserver.
	Transition(api.ComponentPhase, api.ComponentCondition) error
	// Watch returns a new read-only buffered error channel that will be closed when the heartbeat stops.
	// The returned channel may receive zero or more errors before closing.
	// When the HeartBeat is stopped or dies the channel will be closed.
	Watch() <-chan error
	// Stop terminates continuous heart beating gracefully (e.g. sigterm).
	Stop()
	// Kill terminates continuous heart beating with an error.
	Kill(error)
}

type heartbeat struct {
	state            *api.Component
	stateLock        sync.Mutex
	componentsClient client.ComponentsClient
	// channel to send errors to
	errCh chan error
	// channel to add new watchers
	watchCh chan chan error
	ticker  *time.Ticker
}

// Start registers a new component and initiates continuous heart beating (component updates).
// Phase defaults to Pending.
func Start(componentsClient client.ComponentsClient, period time.Duration, t api.ComponentType, location string) (HeartBeat, error) {
	initialState := &api.Component{
		Spec: api.ComponentSpec{
			Type:    t,
			Address: location,
		},
		Status: api.ComponentStatus{
			Phase:      api.ComponentPending,
			Conditions: []api.ComponentCondition{},
		},
	}

	newState, err := componentsClient.Create(initialState)
	if err != nil {
		return nil, fmt.Errorf("component create failed (state=%+v): %v", initialState, err)
	}

	hb := &heartbeat{
		state:            newState,
		componentsClient: componentsClient,
		errCh:            make(chan error, 1),
		watchCh:          make(chan chan error),
		ticker:           time.NewTicker(period),
	}

	go errorHandler(hb.errCh, hb.watchCh)
	go hb.beat()
	return hb, nil
}

// errorHandler propagates errors sent to errCh to a list of watchers.
// New watchers can be registered by sending a new buffered errCh over the watchCh channel.
// This function exclusively owns the watcher list to avoid concurrent access.
// Most heartbeat users will only have one watcher, but watchCh and errCh may be closed before they start watching.
func errorHandler(errCh chan error, watchCh chan chan error) {
	watchers := make([]chan error, 1)
	for {
		select {
		case err, open := <-errCh:
			if !open {
				// stop accepting new watchers
				close(watchCh)
				// propagate closure to watchers
				for _, errCh := range watchers {
					close(errCh)
				}
				errCh = nil
			} else {
				// propagate errors to watchers
				// this could be asynchronous, but the buffer should make it non-blocking
				for _, errCh := range watchers {
					errCh <- err
				}
			}
		case watcher, open := <-watchCh:
			if !open {
				watchCh = nil
			} else {
				watchers = append(watchers, watcher)
			}
		}

		if errCh == nil && watchCh == nil {
			break
		}
	}
}

func (hb *heartbeat) Transition(phase api.ComponentPhase, condition api.ComponentCondition) error {
	hb.stateLock.Lock()
	defer hb.stateLock.Unlock()

	hb.state.Status.Phase = phase
	hb.state.Status.Conditions = append(hb.state.Status.Conditions, condition)

	newState, err := hb.componentsClient.Update(hb.state)
	if err != nil {
		return fmt.Errorf("component update failed (state=%+v): %v", hb.state, err)
	}

	hb.state = newState
	return nil
}

func (hb *heartbeat) Watch() <-chan error {
	errCh := make(chan error, 1)
	// will block util errorHandler pics it up
	// will panic if heartbeat has already been stopped
	hb.watchCh <- errCh
	return errCh
}

func (hb *heartbeat) Stop() {
	err := hb.Transition(api.ComponentTerminated, api.ComponentCondition{
		Type:   api.ComponentTerminatedCleanly,
		Status: api.ConditionTrue,
		// TODO: add condition reason/message
	})
	if err != nil {
		glog.Errorf("Stop transition failed: %v", err)
	}

	// will do nothing if heartbeat has already been stopped
	close(hb.errCh)
}

func (hb *heartbeat) Kill(err error) {
	tErr := hb.Transition(api.ComponentTerminated, api.ComponentCondition{
		Type:    api.ComponentTerminatedCleanly,
		Status:  api.ConditionFalse,
		Reason:  "Error",
		Message: err.Error(),
	})
	if err != nil {
		glog.Errorf("Kill transition failed: %v", tErr)
	}

	// will block util errorHandler pics it up
	// will panic if heartbeat has already been stopped
	hb.errCh <- err
	// will do nothing if heartbeat has already been stopped
	close(hb.errCh)
}

func (hb *heartbeat) beat() {
	for now := range hb.ticker.C {
		hb.beatOnce(now)
	}
}

// beatOnce sends the current component state as an update to the apiserver
func (hb *heartbeat) beatOnce(_ time.Time) error {
	hb.stateLock.Lock()
	defer hb.stateLock.Unlock()

	newState, err := hb.componentsClient.Update(hb.state)
	if err != nil {
		return fmt.Errorf("component update failed (state=%+v): %v", hb.state, err)
	}

	hb.state = newState
	return nil
}
