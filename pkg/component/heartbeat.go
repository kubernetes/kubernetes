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
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

type HeartBeat interface {
	// Transition changes the component's current phase
	Transition(api.ComponentPhase, api.ComponentCondition)
	// Watch returns a new read-only buffered error channel that will be closed when the heartbeat stops.
	// The returned channel may receive zero or more errors before closing.
	// When the HeartBeat is stopped or dies the channel will be closed.
	Watch() <-chan error
	// Stop terminates continuous heart beating.
	Stop()
}

type heartbeat struct {
	componentsClient client.ComponentsClient
	// channel to send errors to
	errCh chan error
	// channel to add new watchers
	watchCh         chan chan error
	ticker          *time.Ticker
	componentBuffer chan api.Component
}

// Start registers a new component and initiates continuous heart beating (component updates).
// Phase defaults to Pending.
func Start(componentsClient client.ComponentsClient, period time.Duration, t api.ComponentType, location string) HeartBeat {
	hb := &heartbeat{
		componentsClient: componentsClient,
		errCh:            make(chan error, 1),
		watchCh:          make(chan chan error),
		ticker:           time.NewTicker(period),
		componentBuffer:  make(chan api.Component, 1),
	}

	// init buffer
	hb.componentBuffer <- api.Component{
		Spec: api.ComponentSpec{
			Type:    t,
			Address: location,
		},
		Status: api.ComponentStatus{
			Phase:      api.ComponentPending,
			Conditions: []api.ComponentCondition{},
		},
	}

	go errorHandler(hb.errCh, hb.watchCh)

	// initial registration blocks
	err := hb.create()
	if err != nil {
		hb.kill(err)
		return nil
	}

	go hb.run()
	return hb
}

// errorHandler propegates errors sent to errCh to a list of watchers.
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

func (hb *heartbeat) Transition(phase api.ComponentPhase, condition api.ComponentCondition) {
	hb.updateBuffer(func(component api.Component) (api.Component, error) {
		component.Status.Phase = phase
		component.Status.Conditions = append(component.Status.Conditions, condition)
		return component, nil
	})
	//TODO(karlkfi): immediately do an update?
}

func (hb *heartbeat) Watch() <-chan error {
	errCh := make(chan error, 1)
	// will block util errorHandler pics it up
	// will panic if heartbeat has already been stopped
	hb.watchCh <- errCh
	return errCh
}

func (hb *heartbeat) Stop() {
	// will do nothing if heartbeat has already been stopped
	close(hb.errCh)
}

func (hb *heartbeat) kill(err error) {
	// will block util errorHandler pics it up
	// will panic if heartbeat has already been stopped
	hb.errCh <- err
	hb.Stop()
}

func (hb *heartbeat) run() {
	for now := range hb.ticker.C {
		hb.update(now)
	}
}

func (hb *heartbeat) create() error {
	return hb.updateBuffer(func(component api.Component) (api.Component, error) {
		newC, err := hb.componentsClient.Create(&component)
		if err != nil {
			return component, fmt.Errorf("heartbeat component create failed (component=%+v): %v", component, err)
		}
		return *newC, nil
	})
}

func (hb *heartbeat) update(_ time.Time) error {
	return hb.updateBuffer(func(component api.Component) (api.Component, error) {
		newC, err := hb.componentsClient.Update(&component)
		if err != nil {
			return component, fmt.Errorf("heartbeat component update failed (component=%+v): %v", component, err)
		}
		return *newC, nil
	})
}

// updateBuffer updates the buffered component in a synchronized way
// TODO(karlkfi) replace this complexity with a pipeline goroutine that owns the current component state.
func (hb *heartbeat) updateBuffer(updater func(api.Component) (api.Component, error)) error {
	component := <-hb.componentBuffer
	defer func() { hb.componentBuffer <- component }()
	newC, err := updater(component)
	if err != nil {
		// ignore update on error
		return err
	}
	component = newC
	return nil
}
