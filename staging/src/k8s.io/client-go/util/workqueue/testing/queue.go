/*
Copyright 2022 The Kubernetes Authors.

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

package testing

import (
	"time"

	"k8s.io/client-go/util/workqueue"
)

// SpyWorkQueue implements a work queue and adds the ability to inspect processed
// items for testing purposes.
type SpyWorkQueue struct {
	workqueue.RateLimitingInterface
	items []SpyQueueItem
}

// SpyQueueItem represents an item that was being processed.
type SpyQueueItem struct {
	Key interface{}
	// Delay represents the delayed duration if and only if AddAfter was invoked.
	Delay time.Duration
}

// AddAfter is like workqueue.RateLimitingInterface.AddAfter but records the
// added key and delay internally.
func (f *SpyWorkQueue) AddAfter(key interface{}, delay time.Duration) {
	f.items = append(f.items, SpyQueueItem{
		Key:   key,
		Delay: delay,
	})

	f.RateLimitingInterface.AddAfter(key, delay)
}

// GetItems returns all items that were recorded.
func (f *SpyWorkQueue) GetItems() []SpyQueueItem {
	return f.items
}
