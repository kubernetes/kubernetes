/*
Copyright 2018 The Kubernetes Authors.

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

package audit

import (
	"fmt"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

type backend struct {
	// name of the backend
	name string
	// channel to buffer events before sending to user
	buffer chan *auditinternal.Event
	// shutdowd channel is closed when the backend is closed by registry
	shutdown chan struct{}
	// conClosed channeld is closed when the http connection is closed
	conClosed chan struct{}
}

var _ audit.Backend = &backend{}

func newBackend(name string) *backend {
	return &backend{
		name:      name,
		buffer:    make(chan *auditinternal.Event, 100),
		shutdown:  make(chan struct{}),
		conClosed: make(chan struct{}),
	}
}

func (b *backend) String() string {
	return b.name
}

func (b *backend) Run(stopCh <-chan struct{}) error {
	return nil
}

func (b *backend) Shutdown() {
	close(b.shutdown)
}

func (b *backend) ProcessEvents(ev ...*auditinternal.Event) {
	select {
	case <-b.shutdown:
		return
	case <-b.conClosed:
		return
	default:
	}

	var sendErr error
	var evIndex int
	bufferBlock := fmt.Errorf("audit buffer queue blocked")

	// If buffer channel was closed, an attempt to add an event to it will result in
	// panic that we should recover from.
	defer func() {
		if err := recover(); err != nil {
			sendErr = fmt.Errorf("%v", err)
		}
		if sendErr != nil {
			if sendErr == bufferBlock {
				// if the backend is closed, ignore the block error
				select {
				case <-b.shutdown:
					return
				case <-b.conClosed:
					return
				default:
				}
			}
			audit.HandlePluginError(b.name, sendErr, ev[evIndex:]...)
		}
	}()

	for i, event := range ev {
		evIndex = i

		select {
		case b.buffer <- event:
		default:
			sendErr = bufferBlock
			return
		}
	}
}
