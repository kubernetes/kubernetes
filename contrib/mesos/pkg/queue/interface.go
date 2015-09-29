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

package queue

import (
	"time"

	"k8s.io/kubernetes/pkg/client/cache"
)

type EventType int

const (
	ADD_EVENT EventType = 1 << iota
	UPDATE_EVENT
	DELETE_EVENT
	POP_EVENT
)

type UniqueID interface {
	GetUID() string
}

type FIFO interface {
	cache.Store

	// Pop waits until an item is ready and returns it. If multiple items are
	// ready, they are returned in the order in which they were added/updated.
	// The item is removed from the queue (and the store) before it is returned,
	// so if you don't successfully process it, you need to add it back with Add().
	Pop() interface{}

	// Await attempts to Pop within the given interval; upon success the non-nil
	// item is returned, otherwise nil
	Await(timeout time.Duration) interface{}

	// Is there an entry for the id that matches the event mask?
	Poll(id string, types EventType) bool
}
