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

package heartbeat

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

type HeartBeat interface {
	// Transition changes the component's current phase
	Transition(api.ComponentPhase)
	// Stop terminates continuous heart beating.
	Stop()
}

type heartbeat struct {
	componentsClient client.ComponentsClient
	errCh            chan error
	ticker           *time.Ticker
	componentBuffer  chan api.Component
}

// Start registers a new component and initiates continuous heart beating (component updates).
// The returned channel may receive zero or more errors before closing.
// When the HeartBeat is stopped or dies the channel will be closed.
// Phase initialized to Pending, use Transition(phase) to change to Running or Stopped.
func Start(componentsClient client.ComponentsClient, period time.Duration, t api.ComponentType, location string) (HeartBeat, <-chan error) {
	hb := &heartbeat{
		componentsClient: componentsClient,
		errCh:            make(chan error, 1),
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
			Phase: api.ComponentPending,
		},
	}

	// initial registration blocks
	err := hb.create()
	if err != nil {
		hb.kill(err)
		return nil, hb.errCh
	}

	go hb.run()
	return hb, hb.errCh
}

func (hb *heartbeat) Transition(phase api.ComponentPhase) {
	hb.updateBuffer(func(component api.Component) (api.Component, error) {
		component.Status.Phase = phase
		return component, nil
	})
	//TODO(karlkfi): immediately do an update?
}

func (hb *heartbeat) Stop() {
	close(hb.errCh)
}

func (hb *heartbeat) kill(err error) {
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
