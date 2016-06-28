/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package framework

import (
	"fmt"
	"sync"

	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
)

// queuedEventHandler is an implementation of ResourceEventHandler.
// It queues events and calls user-registered handler asynchronously so that
// it doesn't block informer from calling other handlers.
type queuedEventHandler struct {
	internalHandler ResourceEventHandler
	// pendingNotifications is an unbounded slice that holds all notifications not yet distributed
	// there is one per listener, but a failing/stalled listener will have infinite pendingNotifications
	// added until we OOM.
	// TODO: make handler closeable and close it when pending queue is too long.
	pendingNotifications []interface{}
	// mu/cond protects access to 'pendingNotifications'.
	mu   sync.RWMutex
	cond sync.Cond

	nextCh  chan interface{}
	queueCh chan interface{}
}

type updateNotification struct {
	oldObj interface{}
	newObj interface{}
}

type addNotification struct {
	newObj interface{}
}

type deleteNotification struct {
	oldObj interface{}
}

func newQueuedEventHandler(hd ResourceEventHandler) *queuedEventHandler {
	h := &queuedEventHandler{
		internalHandler: hd,
		nextCh:          make(chan interface{}),
		queueCh:         make(chan interface{}),
	}
	h.cond.L = &h.mu
	return h
}

func (h *queuedEventHandler) run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	go h.eventQueuing(stopCh)

	for {
		select {
		case next := <-h.nextCh:
			switch msg := next.(type) {
			case updateNotification:
				h.internalHandler.OnUpdate(msg.oldObj, msg.newObj)
			case addNotification:
				h.internalHandler.OnAdd(msg.newObj)
			case deleteNotification:
				h.internalHandler.OnDelete(msg.oldObj)
			default:
				panic(fmt.Errorf("unrecognized notification: %#v", next))
			}
		case <-stopCh:
			return
		}
	}
}

func (h *queuedEventHandler) eventQueuing(stopCh <-chan struct{}) {
	for {
		h.mu.Lock()
		for len(h.pendingNotifications) == 0 {
			select {
			case <-stopCh:
				return
			default:
			}
			h.cond.Wait()
		}
		nt := h.pendingNotifications[0]
		h.pendingNotifications = h.pendingNotifications[1:]
		h.mu.Unlock()

		select {
		case h.nextCh <- nt:
		case <-stopCh:
			return
		}
	}
}

func (h *queuedEventHandler) queue(msg interface{}) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.pendingNotifications = append(h.pendingNotifications, msg)
	h.cond.Signal()
}

// OnAdd calls AddFunc if it's not nil.
func (h *queuedEventHandler) OnAdd(obj interface{}) {
	h.queue(addNotification{newObj: obj})
}

// OnUpdate calls UpdateFunc if it's not nil.
func (h *queuedEventHandler) OnUpdate(oldObj, newObj interface{}) {
	h.queue(updateNotification{oldObj: oldObj, newObj: newObj})
}

// OnDelete calls DeleteFunc if it's not nil.
func (h *queuedEventHandler) OnDelete(obj interface{}) {
	h.queue(deleteNotification{oldObj: obj})
}
