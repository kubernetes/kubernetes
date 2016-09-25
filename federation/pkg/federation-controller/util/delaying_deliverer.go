/*
Copyright 2016 The Kubernetes Authors.

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

// TODO: consider moving it to a more generic package.
package util

import (
	"container/heap"
	"time"
)

const (
	// TODO: Investigate what capacity is right.
	delayingDelivererUpdateChanCapacity = 1000
)

// DelayingDelivererItem is structure delivered by DelayingDeliverer to the
// target channel.
type DelayingDelivererItem struct {
	// Key under which the value was added to deliverer.
	Key string
	// Value of the item.
	Value interface{}
	// When the item should be delivered.
	DeliveryTime time.Time
}

type delivererHeap struct {
	keyPosition map[string]int
	data        []*DelayingDelivererItem
}

// Functions required by container.Heap.

func (dh *delivererHeap) Len() int { return len(dh.data) }
func (dh *delivererHeap) Less(i, j int) bool {
	return dh.data[i].DeliveryTime.Before(dh.data[j].DeliveryTime)
}
func (dh *delivererHeap) Swap(i, j int) {
	dh.keyPosition[dh.data[i].Key] = j
	dh.keyPosition[dh.data[j].Key] = i
	dh.data[i], dh.data[j] = dh.data[j], dh.data[i]
}

func (dh *delivererHeap) Push(x interface{}) {
	item := x.(*DelayingDelivererItem)
	dh.data = append(dh.data, item)
	dh.keyPosition[item.Key] = len(dh.data) - 1
}

func (dh *delivererHeap) Pop() interface{} {
	n := len(dh.data)
	item := dh.data[n-1]
	dh.data = dh.data[:n-1]
	delete(dh.keyPosition, item.Key)
	return item
}

// A structure that pushes the items to the target channel at a given time.
type DelayingDeliverer struct {
	// Channel to deliver the data when their time comes.
	targetChannel chan *DelayingDelivererItem
	// Store for data
	heap *delivererHeap
	// Channel to feed the main goroutine with updates.
	updateChannel chan *DelayingDelivererItem
	// To stop the main goroutine.
	stopChannel chan struct{}
}

func NewDelayingDeliverer() *DelayingDeliverer {
	return NewDelayingDelivererWithChannel(make(chan *DelayingDelivererItem, 100))
}

func NewDelayingDelivererWithChannel(targetChannel chan *DelayingDelivererItem) *DelayingDeliverer {
	return &DelayingDeliverer{
		targetChannel: targetChannel,
		heap: &delivererHeap{
			keyPosition: make(map[string]int),
			data:        make([]*DelayingDelivererItem, 0),
		},
		updateChannel: make(chan *DelayingDelivererItem, delayingDelivererUpdateChanCapacity),
		stopChannel:   make(chan struct{}),
	}
}

// Deliver all items due before or equal to timestamp.
func (d *DelayingDeliverer) deliver(timestamp time.Time) {
	for d.heap.Len() > 0 {
		if timestamp.Before(d.heap.data[0].DeliveryTime) {
			return
		}
		item := heap.Pop(d.heap).(*DelayingDelivererItem)
		d.targetChannel <- item
	}
}

func (d *DelayingDeliverer) run() {
	for {
		now := time.Now()
		d.deliver(now)

		nextWakeUp := now.Add(time.Hour)
		if d.heap.Len() > 0 {
			nextWakeUp = d.heap.data[0].DeliveryTime
		}
		sleepTime := nextWakeUp.Sub(now)

		select {
		case <-time.After(sleepTime):
			break // just wake up and process the data
		case item := <-d.updateChannel:
			if position, found := d.heap.keyPosition[item.Key]; found {
				if item.DeliveryTime.Before(d.heap.data[position].DeliveryTime) {
					d.heap.data[position] = item
					heap.Fix(d.heap, position)
				}
				// Ignore if later.
			} else {
				heap.Push(d.heap, item)
			}
		case <-d.stopChannel:
			return
		}
	}
}

// Starts the DelayingDeliverer.
func (d *DelayingDeliverer) Start() {
	go d.run()
}

// Stops the DelayingDeliverer. Undelivered items are discarded.
func (d *DelayingDeliverer) Stop() {
	close(d.stopChannel)
}

// Delivers value at the given time.
func (d *DelayingDeliverer) DeliverAt(key string, value interface{}, deliveryTime time.Time) {
	d.updateChannel <- &DelayingDelivererItem{
		Key:          key,
		Value:        value,
		DeliveryTime: deliveryTime,
	}
}

// Delivers value after the given delay.
func (d *DelayingDeliverer) DeliverAfter(key string, value interface{}, delay time.Duration) {
	d.DeliverAt(key, value, time.Now().Add(delay))
}

// Gets target channel of the deliverer.
func (d *DelayingDeliverer) GetTargetChannel() chan *DelayingDelivererItem {
	return d.targetChannel
}

// Starts Delaying deliverer with a handler listening on the target channel.
func (d *DelayingDeliverer) StartWithHandler(handler func(*DelayingDelivererItem)) {
	go func() {
		for {
			select {
			case item := <-d.targetChannel:
				handler(item)
			case <-d.stopChannel:
				return
			}
		}
	}()
	d.Start()
}
