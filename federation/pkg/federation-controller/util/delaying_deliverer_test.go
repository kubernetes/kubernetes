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

package util

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestDelayingDeliverer(t *testing.T) {
	targetChannel := make(chan *DelayingDelivererItem)
	now := time.Now()
	d := NewDelayingDelivererWithChannel(targetChannel)
	d.Start()
	defer d.Stop()
	startupDelay := time.Second
	d.DeliverAt("a", "aaa", now.Add(startupDelay+2*time.Millisecond))
	d.DeliverAt("b", "bbb", now.Add(startupDelay+3*time.Millisecond))
	d.DeliverAt("c", "ccc", now.Add(startupDelay+1*time.Millisecond))
	d.DeliverAt("e", "eee", now.Add(time.Hour))
	d.DeliverAt("e", "eee", now)

	d.DeliverAt("d", "ddd", now.Add(time.Hour))

	i0 := <-targetChannel
	assert.Equal(t, "e", i0.Key)
	assert.Equal(t, "eee", i0.Value.(string))
	assert.Equal(t, now, i0.DeliveryTime)

	i1 := <-targetChannel
	received1 := time.Now()
	assert.True(t, received1.Sub(now).Nanoseconds() > startupDelay.Nanoseconds())
	assert.Equal(t, "c", i1.Key)

	i2 := <-targetChannel
	assert.Equal(t, "a", i2.Key)

	i3 := <-targetChannel
	assert.Equal(t, "b", i3.Key)

	select {
	case <-targetChannel:
		t.Fatalf("Nothing should be received")
	case <-time.After(time.Second):
		// Ok. Expected
	}
}
