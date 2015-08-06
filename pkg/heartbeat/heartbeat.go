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
	// Stop terminates continuous heart beating.
	Stop()
}

type heartbeat struct {
	componentsClient client.ComponentsClient
	errCh            chan error
	ticker           *time.Ticker
	component        *api.Component
}

// Start registers a new component and initiates continuous heart beating (component updates).
// The returned channel may receive zero or more errors before closing.
// When the HeartBeat is stopped or dies the channel will be closed.
func Start(componentsClient client.ComponentsClient, period time.Duration, t api.ComponentType, location string) (HeartBeat, <-chan error) {
	errCh := make(chan error, 1)
	ticker := time.NewTicker(period)
	hb := &heartbeat{
		componentsClient: componentsClient,
		errCh:            errCh,
		ticker:           ticker,
		component: &api.Component{
			Type: t,
			URL:  location,
		},
	}

	// initial registration blocks
	err := hb.create()
	if err != nil {
		hb.kill(err)
		return nil, errCh
	}

	go hb.run()
	return hb, errCh
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
	newC, err := hb.componentsClient.Create(hb.component)
	if err != nil {
		return fmt.Errorf("heartbeat component create failed (component=%+v): %v", hb.component, err)
	}
	hb.component = newC
	return nil
}

func (hb *heartbeat) update(_ time.Time) error {
	newC, err := hb.componentsClient.Update(hb.component)
	if err != nil {
		return fmt.Errorf("heartbeat component update failed (component=%+v): %v", hb.component, err)
	}
	hb.component = newC
	return nil
}
